import os
import sys
import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.common_utils import train_PPO_agent, compute_advantage, read_ckp
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
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.low = low
        self.high = high

    def forward(self, x):
        # 也可以直接用tanh激活输出一个固定值, 再缩放为所需动作值
        # 输出均值方差的好处是可以创建一个正态分布, 再采样一次, 还有探索空间
        x = F.relu(self.fc1(x))
        mu = torch.tanh(self.fc_mu(x))
        mu = self.low + (mu + 1.0) * (self.high - self.low) / 2
        std = F.softplus(self.fc_std(x))
        return mu, std
    
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
        return x


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

    def take_action(self, state) -> torch.Tensor:
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.int).view(-1, 1).to(self.device)
        truncated = torch.tensor(transition_dict['truncated'], dtype=torch.int).view(-1, 1).to(self.device)
        
        target_q = self.critic(next_states)
        td_target = rewards + self.gamma * target_q * (1 - dones | truncated)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        # 所谓的另一个演员就是原来的演员的初始状态
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(actions)
        
        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
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
    parser.add_argument('-m', '--model_name', default="PPO", type=str, help='算法名称')
    parser.add_argument('-t', '--task', default="lunar", type=str, help='任务名称')
    parser.add_argument('-w', '--writer', default=0, type=int, help='存档等级, 0: 不存，1: 本地')
    parser.add_argument('-e', '--episodes', default=200, type=int, help='运行回合数')
    parser.add_argument('-s', '--seed', nargs='+', default=[1, 7], type=int, help='起始种子')
    args = parser.parse_args()
    
    # 环境相关
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 环境相关
    # 连续动作空间环境基准测试
    if args.task == 'pendulum':
        env = gym.make('Pendulum-v1')
    elif args.task == 'lunar':
        env = gym.make("LunarLander-v2", continuous=True)
    elif args.task == 'walker':
        env = gym.make("BipedalWalker-v3")
    
    # PPO相关
    actor_lr = 2e-4
    critic_lr = 2e-4
    lmbda = 0.9  # 似乎可以去掉，这一项仅用于调整计算优势advantage时，额外调整折算奖励的系数
    gamma = 0.9  # 时序差分学习率，也作为折算奖励的系数之一
    total_epochs = 1  # 迭代轮数

    eps = 0.2  # 截断范围参数, 1-eps ~ 1+eps
    epochs = 30  # PPO中一条序列训练多少轮，和迭代算法无关

    # 神经网络相关
    hidden_dim = 128
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    if (abs(env.action_space.high) != abs(env.action_space.low)).any():
        print('WARNING: 动作空间不对称')
    low = torch.tensor(env.action_space.low, device=device)
    high = torch.tensor(env.action_space.high, device=device)
    
    # 任务相关
    system_type = sys.platform  # 操作系统

    print(f'[ 开始训练, 任务: {args.task}, 模型: {args.model_name}, 设备: {device} ]')
    # * ----------------------- 训练 ----------------------------
    for seed in trange(args.seed[0], args.seed[-1] + 1, mininterval=40, ncols=100):
        CKP_PATH = f'ckpt/{args.task}/{args.model_name}/{seed}_{system_type}.pt'
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, 
                    critic_lr, gamma, lmbda, epochs, eps, device)
        s_epoch, s_episode, return_list, time_list, seed_list = read_ckp(CKP_PATH, agent, 'PPO')
        return_list, train_time = train_PPO_agent(env, agent, args.writer, s_epoch, total_epochs,
                                                  s_episode, args.episodes, return_list, time_list, seed_list,
                                                  seed, CKP_PATH,
                                                  )

        # * ----------------- 绘图 ---------------------

        # sns.lineplot(return_list, label=f'{seed}')
        # plt.title(f'{args.model_name}, training time: {train_time} min')
        # plt.xlabel('Episode')
        # plt.ylabel('Return')
        # plt.savefig(f'image/tmp/train_{args.model_name}_{system_type}.pdf')
        # plt.close()
