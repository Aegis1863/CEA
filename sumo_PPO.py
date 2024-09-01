import os
os.environ['LIBSUMO_AS_TRACI'] = '1'  # 终端运行加速
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

parser = argparse.ArgumentParser(description='sumo_PPO 任务')
parser.add_argument('--model_name', default="PPO", type=str, help='任务_基本算法名称')
parser.add_argument('--mission', default="highway", type=str, help='任务名称')
parser.add_argument('-n', '--net', default="env/big-intersection/big-intersection.net.xml", type=str, help='SUMO路网文件路径')
parser.add_argument('-f', '--flow', default="env/big-intersection/big-intersection.rou.xml", type=str, help='SUMO车流文件路径')
parser.add_argument('-w', '--writer', default=1, type=int, help='存档等级, 0: 不存，1: 本地 2: 本地 + wandb本地, 3. 本地 + wandb云存档')
parser.add_argument('-o', '--online', action="store_true", help='是否上传wandb云')
parser.add_argument('--sta', action="store_true", help='是否利用sta辅助')
parser.add_argument('--sta_kind', default=False, help='sta 预训练模型类型，"expert"或"regular"')
parser.add_argument('-e', '--episodes', default=100, type=int, help='运行回合数')
parser.add_argument('-r', '--reward', default='diff-waiting-time', type=str, help='奖励函数')
parser.add_argument('--begin_time', default=1000, type=int, help='回合开始时间')
parser.add_argument('--duration', default=2000, type=int, help='单回合运行时间')
parser.add_argument('--begin_seed', default=42, type=int, help='起始种子')
parser.add_argument('--end_seed', default=52, type=int, help='结束种子')

args = parser.parse_args()

if args.writer == 2:
    if os.path.exists("api_key.txt"):
        with open("api_key.txt", "r") as f:  # 该文件中写入一行wandb的API
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
        sta: object,
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
        if sta:
            self.sta = cvae.to(device)
            self.sta_optimizer = torch.optim.Adam(self.sta.parameters(), lr=1e-3)
        else:
            self.sta = None

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
        
        # * 技巧
        if not args.sta_kind and args.sta:  # ! 在线训练
            loss = self.train_cvae(states, next_states, False, states.shape[0]//400)  # 训练 vae, 如果数据比较少则batch_size一定要小
            quality = self.sta.generate_test(32, 4, save_path=f'image/{mission}/{args.model_name}/')  # 生成 cvae 图像观察效果
        if self.sta:
            pre_next_state = self.predict_next_state(states, next_states)
            target_q1 = self.critic(pre_next_state).detach()
            target_q2 = self.critic(next_states).detach()
            target_q = torch.min(target_q1, target_q2)
        else:
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
            
    def train_cvae(self, state, next_state, test_and_feedback, batch_size):
        vae_action = next_state[:, :4]
        diff_state = next_state[:, 5:] - state[:, 5:]
        loss = cvae_train(self.sta, self.device, diff_state, vae_action, self.sta_optimizer, test_and_feedback, batch_size)
        return loss
    
    def predict_next_state(self, state, next_state):
        '''sumo 此处构造与其他的不一致'''
        action = state[:, :4]
        with torch.no_grad():
            sample = torch.randn(state.shape[0], 32).to(device)  # 随机采样的
            generated = self.sta.decode(sample, action)
        pre_next_state = torch.concat([next_state[:, :5], state[:, 5:] + generated], dim=-1)
        return pre_next_state
    
    
# * --------------------- 参数 -------------------------
if __name__ == '__main__':
    # 环境相关
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
    
    # PPO相关
    actor_lr = 1e-3
    critic_lr = 1e-2
    lmbda = 0.95  # 似乎可以去掉，这一项仅用于调整计算优势advantage时，额外调整折算奖励的系数
    gamma = 0.98  # 时序差分学习率，也作为折算奖励的系数之一
    total_epochs = 1  # 迭代轮数
    eps = 0.2  # 截断范围参数, 1-eps ~ 1+eps
    epochs = 10  # PPO中一条序列训练多少轮，和迭代算法无关

    # 神经网络相关
    hidden_dim = 128
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # VAE
    if args.sta:
        args.model_name = args.model_name + '~' + 'cvae'
        if args.sta_kind and args.sta:  # 读取预训练模型
            print(f'==> 读取{args.sta_kind} cvae模型')
            args.model_name = args.model_name + '~' + args.sta_kind
            cvae = torch.load(f'model/sta/{mission}/{args.sta_kind}.pt', map_location=device)
        else:
            print(f'==> 在线训练 cvae模型')
            cvae = CVAE(state_dim - 5, action_dim, state_dim - 5)
    else:
        cvae = None
    
    # 任务相关
    system_type = sys.platform  # 操作系统
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
                sumo_seed=seed,  # 需要切换种子
                additional_sumo_cmd='--no-step-log')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        agent = PPO(state_dim, hidden_dim, action_dim, cvae, actor_lr, 
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
