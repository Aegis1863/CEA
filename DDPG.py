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
        action = torch.tanh(self.fc2(x))
        action = self.low + (action + 1.0) * (self.high - self.low) / 2  # Zoom to the specified space
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
    ''' DDPG '''
    def __init__(self, state_dim, hidden_dim, action_dim,sigma, actor_lr, 
                 critic_lr, tau, gamma, device, training=True):
        self.training = training
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma
        self.tau = tau
        self.action_dim = action_dim
        self.device = device

    def take_action(self, state, the_device=None):
        state = torch.tensor(state) if not isinstance(state, torch.Tensor) else state
        state = state.to(the_device) if the_device else state.to(self.device)
        action = self.actor(state).detach().cpu()
        if self.training:
            action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        '''Soft update target_net in the direction of net, each update is very small

        Params
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

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones | truncated)
        critic_losses = F.mse_loss(self.critic(states, actions), q_targets.detach(), reduction='none')
        critic_loss = torch.mean(critic_losses * weights)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
        
        return critic_losses.detach().cpu()
    
# * --------------------- 参数 -------------------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='DDPG task')
    parser.add_argument('--model_name', default="DDPG", type=str, help='Alg name')
    parser.add_argument('-t', '--task', default="lunar", type=str, help='Task name')
    parser.add_argument('-w', '--writer', default=0, type=int, help='Saving type, 0: No，1: local')
    parser.add_argument('--sta', action="store_true", help='Whether to use STA')
    parser.add_argument('--per', action="store_true", help='Whether to use PER')
    parser.add_argument('-d', '--distance_ratio', default=0.2, type=float, help='Generate experience ratio')
    parser.add_argument('-n', '--num_new_samples', default=3, type=int, help='The number of counterfactual actions sampled')
    parser.add_argument('-e', '--episodes', default=300, type=int, help='episodes')
    parser.add_argument('-s', '--seed', nargs='+', default=[1, 7], type=int, help='Start and end seeds')

    args = parser.parse_args()
    
    # env
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Continuous Action Space Environment Benchmark
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
    sigma = 0.01
    tau = 0.005
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    if (abs(env.action_space.high) != abs(env.action_space.low)).any():
        print('WARNING: Action space asymmetry')
    # action bound
    low = torch.tensor(env.action_space.low, device=device)
    high = torch.tensor(env.action_space.high, device=device)
    
    # task
    total_epochs = 1
    batch_size = 128
    system_type = sys.platform

    # CEA params
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
    
    print(f'[ Start training, task: {args.task}, alg: {args.model_name}, device: {device} ]')
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