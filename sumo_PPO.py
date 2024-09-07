import os
os.environ['LIBSUMO_AS_TRACI'] = '1'  # terminal accelerate
import sys
import random
import gymnasium as gym
import time
import sumo_rl
import torch
import torch.nn.functional as F
import numpy as np
from utils.sumo_utils import train_PPO_agent, compute_advantage, read_ckp
from utils.STA import CVAE, cvae_train
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='sumo_PPO task')
parser.add_argument('--model_name', default="PPO", type=str, help='alg name')
parser.add_argument('--mission', default="highway", type=str, help='task name')
parser.add_argument('-n', '--net', default="env/big-intersection/big-intersection.net.xml", type=str, help='a roadnet file of SUMO')
parser.add_argument('-f', '--flow', default="env/big-intersection/big-intersection.rou.xml", type=str, help='a car flow file of SUMO')
parser.add_argument('-w', '--writer', default=1, type=int, help='saving type, 0: No，1: local 2: localn + wandb offline, 3. local + wandb online')
parser.add_argument('-o', '--online', action="store_true", help='whether wandb online')
parser.add_argument('-e', '--episodes', default=100, type=int, help='episodes')
parser.add_argument('-r', '--reward', default='diff-waiting-time', type=str, help='reward function')
parser.add_argument('--begin_time', default=1000, type=int, help='start time of simulation')
parser.add_argument('--duration', default=2000, type=int, help='duration time of simulation')
parser.add_argument('--begin_seed', default=42, type=int, help='begin seed')
parser.add_argument('--end_seed', default=52, type=int, help='end seed')

args = parser.parse_args()

if args.writer == 2:
    if os.path.exists("api_key.txt"):
        with open("api_key.txt", "r") as f:  # The file should be written one line API of wandb
            api_key = f.read()
    os.environ["WANDB_API_KEY"] = api_key
    os.environ["WANDB_MODE"] = "offline"

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
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state) -> list:
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(self.device)
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
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clip(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
    def train_cvae(self, state, next_state, test_and_feedback, batch_size):
        vae_action = next_state[:, :4]
        diff_state = next_state[:, 5:] - state[:, 5:]
        loss = cvae_train(self.sta, self.device, diff_state, vae_action, self.sta_optimizer, test_and_feedback, batch_size)
        return loss
    
    def predict_next_state(self, state, next_state):
        action = state[:, :4]
        with torch.no_grad():
            sample = torch.randn(state.shape[0], 32).to(device)
            generated = self.sta.decode(sample, action)
        pre_next_state = torch.concat([next_state[:, :5], state[:, 5:] + generated], dim=-1)
        return pre_next_state
    
    
# * --------------------- parameters -------------------------
if __name__ == '__main__':
    # env
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
    mission = args.model_name.split('_')[0]
    model_name = args.model_name.split('_')[1]
    
    # PPO
    actor_lr = 1e-3
    critic_lr = 1e-2
    lmbda = 0.95
    gamma = 0.98  # TD gamma
    total_epochs = 1  # KEEP 1 is ok
    eps = 0.2  # truncating eps of PPO
    epochs = 10  # inner epochs of PPO

    # network
    hidden_dim = 128
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # task
    system_type = sys.platform  # operation system
    print('device:', device)

    # * ----------------------- 训练 ----------------------------
    for seed in range(args.begin_seed, args.end_seed + 1):
        CKP_PATH = f'ckpt/{args.mission}/{args.model_name}/{seed}_{system_type}.pt'
        env = gym.make('sumo-rl-v0',
                net_file=args.net,
                route_file=args.flow,
                use_gui=False,
                begin_time=args.begin_time,
                num_seconds=args.duration,
                reward_fn=args.reward,
                sumo_warnings=False,
                sumo_seed=seed,
                additional_sumo_cmd='--no-step-log')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, 
                    critic_lr, gamma, lmbda, epochs, eps, device)
        (s_epoch, s_episode, return_list,  waitt_list, 
        queue_list, speed_list, time_list, seed_list) = read_ckp(CKP_PATH, agent, 'PPO')

        if args.writer > 1:
            wandb.init(
                project="MBPO-SUMO",
                group=args.model_name,
                name=f"{seed}",
                config={
                "episodes": args.episodes,
                "seed": seed,
                "road net": args.net,
                "mission name": args.model_name
                }
            )
        return_list, train_time = train_PPO_agent(env, agent, args.writer, s_epoch, total_epochs, 
                                            s_episode, args.episodes, return_list, queue_list, 
                                            waitt_list, speed_list, time_list, seed_list, seed, CKP_PATH,
                                            )
        # * ----------------- 绘图 ---------------------
        sns.lineplot(return_list, label=f'{seed}')
        plt.title(f'{args.model_name}, training time: {train_time} min')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.savefig(f'image/tmp/{mission}_{args.model_name}_{system_type}.pdf')
