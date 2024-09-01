import os
import sys
import cProfile
import gymnasium as gym
from tqdm import trange
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from utils.common_utils import train_SAC_agent, read_ckp, ReplayBuffer, PrioritizedReplayBuffer

import numpy as np
import random
import argparse
import warnings
warnings.filterwarnings('ignore')

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.low = low
        self.high = high
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def forward(self, x, device=None):
        x = F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        # std = torch.clip(std, self.min_std, self.max_std)
        dist = Normal(mu, std)
        # rsample()是重参数化采样, 输出一个采样值(动作), 直接sample会阻碍计算图梯度更新
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)  # 输出的是该动作的对数概率密度
        action = torch.tanh(normal_sample)
        # action经过tanh的变换，策略不再是高斯分布，重新计算对数概率
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        action = self.low + (action + 1.0) * (self.high - self.low) / 2  # 放缩到指定空间
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    
class SAC:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha, tau, gamma, action_type, device):
        # 策略网络
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        # 第一个Q网络
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        # 第二个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        # self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float)
        # self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        # self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = alpha  # self.log_alpha <-> self.alpha
        # self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.action_type = action_type
        self.device = device

    def take_action(self, state, device=None):
        the_device = device if device else self.device
        state = torch.tensor(state, dtype=torch.float).to(the_device)
        action, _ = self.actor(state)  # 第二个输出是 log_prob
        return action.detach().cpu()

    def calc_target_continuous(self, rewards, next_states, dones, truncated):
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob  # 熵 = -对数动作概率
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.alpha * entropy  # self.alpha.exp() <-> self.alpha
        td_target = rewards + self.gamma * next_value * (1 - dones | truncated)
        return td_target
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).to(self.device)  # 动作不再是float类型
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.int).view(-1, 1).to(self.device)
        truncated = torch.tensor(transition_dict['truncated'], dtype=torch.int).view(-1, 1).to(self.device)
        weights = torch.FloatTensor(transition_dict["weights"].reshape(-1, 1)).to(self.device)  # 无PER时是1
        
        td_target = self.calc_target_continuous(rewards, next_states, dones, truncated)
        critic_1_losses = F.mse_loss(self.critic_1(states, actions), td_target.detach(), reduction='none')
        critic_1_loss = torch.mean(critic_1_losses * weights)
        critic_2_losses = F.mse_loss(self.critic_2(states, actions), td_target.detach(), reduction='none')
        critic_2_loss = torch.mean(critic_2_losses * weights)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.alpha * entropy - torch.min(q1_value, q2_value))  # self.log_alpha.exp() <-> self.alpha
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新alpha值
        # alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.alpha.exp())
        # self.log_alpha_optimizer.zero_grad()
        # alpha_loss.backward()
        # self.log_alpha_optimizer.step()
        
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        
        # PER 更新用
        combined_loss = torch.min(critic_1_losses, critic_2_losses).detach().cpu()
        return combined_loss
            
        
    
# * --------------------- 参数 -------------------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='SAC_cons 任务')
    parser.add_argument('-m', '--model_name', default="SAC", type=str, help='基本算法名称')
    parser.add_argument('-t', '--task', default="pendulum", type=str, help='任务名称')
    parser.add_argument('-w', '--writer', default=0, type=int, help='存档等级, 0: 不存，1: 本地')
    parser.add_argument('--sta', action="store_true", help='是否利用sta辅助')
    parser.add_argument('--per', action="store_true", help='是否采用PER')
    parser.add_argument('-d', '--distance_ratio', default=0.05, type=float, help='虚拟经验比例')
    parser.add_argument('-n', '--num_new_samples', default=3, type=int, help='反事实动作采样个数')
    parser.add_argument('-e', '--episodes', default=200, type=int, help='运行回合数')
    parser.add_argument('-s', '--seed', nargs='+', default=[1, 7], type=int, help='起始种子')

    args = parser.parse_args()
    
    # 环境相关
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 连续动作空间环境基准测试
    if args.task == 'pendulum':
        env = gym.make('Pendulum-v1')
    elif args.task == 'lunar':
        env = gym.make("LunarLander-v2", continuous=True)
    elif args.task == 'walker':
        env = gym.make("BipedalWalker-v3")
    # SAC
    action_type = 'discrete' if isinstance(env.action_space, gym.spaces.Discrete) else 'continuous'
    actor_lr = 2e-4
    critic_lr = 2e-4
    alpha = 0.2
    hidden_dim = 128
    gamma = 0.99
    tau = 0.005  # 软更新参数
    buffer_size = int(5e4)
    minimal_size = buffer_size // 6
    # target_entropy = 0.98 * (-np.log(1 / env.action_space.n)) if action_type == 'discrete' else -env.action_space.shape[0]
    # model_alpha = 0.01  # 模型损失函数中的加权权重
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    if (abs(env.action_space.high) != abs(env.action_space.low)).any():
        print('WARNING: 动作空间不对称')
    # 动作空间范围
    low = torch.tensor(env.action_space.low, device=device)
    high = torch.tensor(env.action_space.high, device=device)
    
    # 任务相关
    total_epochs = 1
    batch_size = 256
    system_type = sys.platform  # 操作系统

    # CEA 参数
    if args.sta:
        path = f'model/sta/{args.task}/regular.pt'
        sta = torch.load(path, map_location=device)
        if args.per:
            args.model_name = f'CEA_{args.model_name}'
        else:
            args.model_name = f'CEA_{args.model_name}-no_PER'
    else:
        if args.per:
            args.model_name = f'{args.model_name}_PER'
        sta = False
    
    print(f'[ 开始训练, 任务: {args.task}, 模型: {args.model_name}, 设备: {device} ]')
    # * ----------------------- 训练 ----------------------------
    for seed in trange(args.seed[0], args.seed[-1] + 1, mininterval=40, ncols=100):
        CKP_PATH = f'ckpt/{args.task}/{args.model_name}/{seed}_{system_type}.pt'
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.per:
            replay_buffer = PrioritizedReplayBuffer(state_dim, action_dim, buffer_size, batch_size)
        else:
            replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size, batch_size)
        agent = SAC(state_dim, hidden_dim, action_dim, actor_lr,
                    critic_lr, alpha, tau, gamma, action_type, device)
        s_epoch, s_episode, return_list, time_list, seed_list = read_ckp(CKP_PATH, agent, args.model_name)
        
        # cProfile.run('''return_list, train_time = train_SAC_agent(env, agent, args.writer, s_epoch, total_epochs,
        #                                           s_episode, args.episodes, replay_buffer, minimal_size, 
        #                                           return_list, time_list, seed_list, seed, CKP_PATH, 
        #                                           sta, args.per, args.num_new_samples, args.distance_ratio
        #                                           )''', './SAC_analysis')
                
        return_list, train_time = train_SAC_agent(env, agent, args.writer, s_epoch, total_epochs,
                                                  s_episode, args.episodes, replay_buffer, minimal_size, 
                                                  return_list, time_list, seed_list, seed, CKP_PATH, 
                                                  sta, args.per, args.num_new_samples, args.distance_ratio
                                                  )