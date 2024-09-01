import os
import sys
import time
import gymnasium as gym
from collections import namedtuple
import itertools
from itertools import count
import sumo_rl
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from utils.highway_utils import train_SAC_agent, read_ckp, ReplayBuffer
import numpy as np
import pandas as pd
import collections
import random
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='SAC(离散) 任务')
parser.add_argument('--model_name', default="SAC", type=str, help='基本算法名称')
parser.add_argument('-t', '--task', default="highway", type=str, help='任务名称')
parser.add_argument('-n', '--net', default="env/big-intersection/big-intersection.net.xml", type=str, help='SUMO路网文件路径')
parser.add_argument('-f', '--flow', default="env/big-intersection/big-intersection.rou.xml", type=str, help='SUMO车流文件路径')
parser.add_argument('-w', '--writer', default=0, type=int, help='存档等级, 0: 不存，1: 本地 2: 本地 + wandb本地, 3. 本地 + wandb云存档')
parser.add_argument('-o', '--online', action="store_true", help='是否上传wandb云')
parser.add_argument('--sta', action="store_true", help='是否利用sta辅助')
parser.add_argument('--sta_kind', default=False, help='sta 预训练模型类型，"expert"或"regular"')
parser.add_argument('-e', '--episodes', default=500, type=int, help='运行回合数')
parser.add_argument('-r', '--reward', default='diff-waiting-time', type=str, help='奖励函数')
parser.add_argument('--begin_time', default=1000, type=int, help='回合开始时间')
parser.add_argument('--duration', default=2000, type=int, help='单回合运行时间')
parser.add_argument('--begin_seed', default=1, type=int, help='起始种子')
parser.add_argument('--end_seed', default=7, type=int, help='结束种子')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.h_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.h_1(F.relu(self.fc1(x))))
        return F.softmax(self.fc2(x), dim=1)  # 直接输出softmax


class QValueNet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.h_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.h_1(F.relu(self.fc1(x))))
        return self.fc2(x)
    
class SAC:
    ''' 处理离散动作的SAC算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 第一个Q网络
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 第二个Q网络
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)  # 动作不再是float类型
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.int).view(-1, 1).to(self.device)
        truncated = torch.tensor(transition_dict['truncated'], dtype=torch.int).view(-1, 1).to(self.device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones | truncated)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新alpha值
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        
        
    
# * --------------------- 参数 -------------------------
if __name__ == '__main__':
    # 环境相关
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.task == 'sumo':
        os.environ['LIBSUMO_AS_TRACI'] = '1'  # 终端运行加速
        env = gym.make('sumo-rl-v0',
                    net_file=args.net,
                    route_file=args.flow,
                    use_gui=False,
                    begin_time=args.begin_time,
                    num_seconds=args.duration,
                    reward_fn=args.reward,
                    sumo_seed=args.begin_seed,
                    sumo_warnings=False,
                    additional_sumo_cmd='--no-step-log')
    elif args.task == 'highway': 
        env = gym.make('highway-fast-v0')
        env.configure({
            "lanes_count": 4,
            "vehicles_density": 2,
            "duration": 100,
        })

    # SAC
    actor_lr = 5e-4
    critic_lr = 5e-3
    alpha_lr = 1e-3
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 20000
    target_entropy = 0.98 * (-np.log(1 / env.action_space.n))
    model_alpha = 0.01  # 模型损失函数中的加权权重
    state_dim = env.observation_space.shape[0] if args.task == 'sumo' else torch.multiply(*env.observation_space.shape)
    action_dim = env.action_space.n
    num_actions = env.action_space.n
    total_epochs = 1
    minimal_size = 500
    batch_size = 64

    # 任务相关
    system_type = sys.platform  # 操作系统
    # args.model_name = args.model_name + '~' +  args.cvae_kind
    print('device:', device)

    # * ----------------------- 训练 ----------------------------
    for seed in trange(args.begin_seed, args.end_seed + 1, mininterval=40, ncols=100):
        CKP_PATH = f'ckpt/{args.task}/{args.model_name}/{seed}_{system_type}.pt'
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        replay_buffer = ReplayBuffer(buffer_size)
        agent = SAC(state_dim, hidden_dim, num_actions, actor_lr,
                    critic_lr, alpha_lr, target_entropy, tau, gamma, device)
        s_epoch, s_episode, return_list, time_list, seed_list = read_ckp(CKP_PATH, agent, args.model_name)
        print('开始训练')
        return_list, train_time = train_SAC_agent(env, agent, args.writer, s_epoch, total_epochs,
                                                  s_episode, args.episodes, replay_buffer, minimal_size, 
                                                  batch_size, return_list, time_list, seed_list,
                                                  seed, CKP_PATH,
                                                  )