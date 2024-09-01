import os
import sys
import cProfile
import gymnasium as gym
from tqdm import trange
import torch
import torch.nn.functional as F
from utils.common_utils import train_DDPG_agent, read_ckp, ReplayBuffer, PrioritizedReplayBuffer
import numpy as np
import random
import argparse
import warnings
warnings.filterwarnings('ignore')

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.low = low
        self.high = high
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 这里直接缩放并输出, 而非像PPO中输出均值方差再采样
        action = torch.tanh(self.fc2(x))
        action = self.low + (action + 1.0) * (self.high - self.low) / 2  # 放缩到指定空间
        return action


class QValueNet(torch.nn.Module):
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

class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim,sigma, actor_lr, 
                 critic_lr, tau, gamma, device, training=True):
        self.training = training
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差, 均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device

    def take_action(self, state, the_device=None):
        state = torch.tensor(state) if not isinstance(state, torch.Tensor) else state
        state = state.to(the_device) if the_device else state.to(self.device)
        action = self.actor(state).detach().cpu()
        if self.training:
            # 给动作添加噪声，增加探索, 但是验证和测试时不需要
            action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        '''将target_net往net方向软更新, 每次更新幅度都很小

        参数说明
        ----------
        net : torch.nn.module
        target_net : torch.nn.module
        '''
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.int).view(-1, 1).to(self.device)
        truncated = torch.tensor(transition_dict['truncated'], dtype=torch.int).view(-1, 1).to(self.device)
        weights = torch.FloatTensor(transition_dict["weights"].reshape(-1, 1)).to(self.device)

        # 评论员还是时序差分更新, 评论员现在叫Q网络, 但是和之前价值网络一样
        # 不同点是需要输入状态和动作, 动作由演员选择, DQN里面的Q网络不需要输入动作
        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones | truncated)
        critic_losses = F.mse_loss(self.critic(states, actions), q_targets.detach(), reduction='none')
        critic_loss = torch.mean(critic_losses * weights)
        # 评论员梯度下降
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络
        
        return critic_losses.detach().cpu()
    
# * --------------------- 参数 -------------------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='DDPG 任务')
    parser.add_argument('--model_name', default="DDPG", type=str, help='基本算法名称')
    parser.add_argument('-t', '--task', default="lunar", type=str, help='任务名称')
    parser.add_argument('-w', '--writer', default=0, type=int, help='存档等级, 0: 不存，1: 本地')
    parser.add_argument('--sta', action="store_true", help='是否利用sta辅助')
    parser.add_argument('--per', action="store_true", help='是否采用PER')
    parser.add_argument('-d', '--distance_ratio', default=0.2, type=float, help='虚拟经验比例')
    parser.add_argument('-n', '--num_new_samples', default=3, type=int, help='反事实动作采样个数')
    parser.add_argument('-e', '--episodes', default=300, type=int, help='运行回合数')
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
    # DDPG
    action_type = 'discrete' if isinstance(env.action_space, gym.spaces.Discrete) else 'continuous'
    actor_lr = 2e-4
    critic_lr = 2e-4
    hidden_dim = 128
    buffer_size = int(5e4)
    minimal_size = buffer_size // 10
    gamma = 0.98
    sigma = 0.01  # 高斯噪声标准差
    tau = 0.005  # 软更新参数, tau越小更新幅度越小
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    if (abs(env.action_space.high) != abs(env.action_space.low)).any():
        print('WARNING: 动作空间不对称')
    # 动作空间范围
    low = torch.tensor(env.action_space.low, device=device)
    high = torch.tensor(env.action_space.high, device=device)
    
    # 任务相关
    total_epochs = 1
    batch_size = 128
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
        agent = DDPG(state_dim, hidden_dim, action_dim, sigma, actor_lr,
                    critic_lr, tau, gamma, device)
        s_epoch, s_episode, return_list, time_list, seed_list = read_ckp(CKP_PATH, agent, args.model_name)
        
        # cProfile.run('''return_list, train_time = train_DDPG_agent(env, agent, args.writer, s_epoch, total_epochs,
        #                                           s_episode, args.episodes, replay_buffer, minimal_size,  
        #                                           return_list, time_list, seed_list, seed, CKP_PATH,
        #                                           sta, args.per, args.num_new_samples, args.distance_ratio
        #                                           )''', './DDPG_analysis')
        
        return_list, train_time = train_DDPG_agent(env, agent, args.writer, s_epoch, total_epochs,
                                                  s_episode, args.episodes, replay_buffer, minimal_size,  
                                                  return_list, time_list, seed_list, seed, CKP_PATH,
                                                  sta, args.per, args.num_new_samples, args.distance_ratio
                                                  )