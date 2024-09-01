'''
只跑 highway
'''

import os
import sys
import random
import gymnasium as gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.highway_utils import train_PPO_agent, compute_advantage, read_ckp
from utils.STA import CVAE, cvae_train
# from dynamic_model.train_Ensemble_dynamic_model import *
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
import argparse
import warnings
warnings.filterwarnings('ignore')


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.h_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.h_1(F.relu(self.fc1(x))))
        return F.softmax(self.fc2(x), dim=-1)
    
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.h_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.h_1(F.relu(self.fc1(x))))
        return self.fc2(x)


class PPO:
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        actor_lr: float=1e-4,
        critic_lr: float=5e-3,
        gamma: float=0.9,
        lmbda: float=0.9,
        epochs: int=20,
        eps: float=0.2,
        device: str='cpu',
    ):
        
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma  # 时序差分学习率
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state) -> list:
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.int).view(-1, 1).to(self.device)
        truncated = torch.tensor(np.array(transition_dict['truncated']), dtype=torch.int).view(-1, 1).to(self.device)

        target_q = self.critic(next_states)   
        td_target = rewards + self.gamma * target_q * (1 - dones | truncated)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        # 所谓的另一个演员就是原来的演员的初始状态
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)  # 重要性采样系数
            surr1 = ratio * advantage  # 重要性采样
            surr2 = torch.clip(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
    
    
# * --------------------- 参数 -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO 任务')
    parser.add_argument('--model_name', default="PPO", type=str, help='任务_基本算法名称')
    parser.add_argument('-t', '--task', default="cliff", type=str, help='任务名称')
    parser.add_argument('-w', '--writer', default=0, type=int, help='存档等级, 0: 不存，1: 本地')
    parser.add_argument('-o', '--online', action="store_true", help='是否上传wandb云')
    parser.add_argument('-e', '--episodes', default=100, type=int, help='运行回合数')
    parser.add_argument('-s', '--seed', nargs='+', default=[1, 7], type=int, help='起始种子')
    args = parser.parse_args()
    
    # 环境相关
    task = args.model_name.split('_')[0]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 环境相关
    if args.task == 'sumo':
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
    elif args.task == 'intersection':
        env = gym.make("intersection-v0")
    elif args.task == 'cliff':
        from utils.gym_wraaper import CliffWalkingWrapper
        from gymnasium.envs.toy_text import CliffWalkingEnv
        env =  CliffWalkingWrapper(CliffWalkingEnv())
    # PPO相关
    actor_lr = 3e-4
    critic_lr = 3e-4
    lmbda = 0.95  # 似乎可以去掉，这一项仅用于调整计算优势advantage时，额外调整折算奖励的系数
    gamma = 0.99  # 时序差分学习率，也作为折算奖励的系数之一
    total_epochs = 1  # 迭代轮数
    total_episodes = 100  # 一轮训练多少次游戏

    eps = 0.2  # 截断范围参数, 1-eps ~ 1+eps
    epochs = 10  # PPO中一条序列训练多少轮，和迭代算法无关

    # 神经网络相关
    hidden_dim = 128
    state_dim = torch.multiply(*env.observation_space.shape) if len(env.observation_space.shape) > 1 else env.observation_space.shape[0]
    try:
        action_dim = env.action_space.n
    except:
        action_dim = env.action_space.shape[0]

    # 任务相关
    system_type = sys.platform  # 操作系统
    # args.model_name = args.model_name + '~' +  args.cvae_kind

    # * ----------------------- 训练 ----------------------------
    print(f'[ 开始训练, 任务: {args.task}, 模型: {args.model_name}, 设备: {device} ]')
    for seed in trange(args.seed[0], args.seed[-1] + 1, mininterval=40, ncols=100):
        CKP_PATH = f'ckpt/{args.task}/{args.model_name}/{seed}_{system_type}.pt'
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, 
                    critic_lr, gamma, lmbda, epochs, eps, device)
        s_epoch, s_episode, return_list, time_list, seed_list = read_ckp(CKP_PATH, agent, 'PPO')

        print(f'开始训练，任务: {args.task}')
        return_list, train_time = train_PPO_agent(env, agent, args.writer, s_epoch, total_epochs,
                                                  s_episode, args.episodes, return_list, time_list, seed_list,
                                                  seed, CKP_PATH,
                                                  )

        # * ----------------- 绘图 ---------------------

        sns.lineplot(return_list, label=f'{seed}')
        plt.title(f'{args.model_name}, training time: {train_time} min')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.savefig(f'image/tmp/train_{args.model_name}_{system_type}.pdf')
        plt.close()
