import os
import sys
import torch
import torch.nn as nn
import time
import wandb
import numpy as np
import pandas as pd
import collections
import random
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import random
from collections import deque
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.nn.utils import clip_grad_norm_
# from tools.segment_tree import MinSegmentTree, SumSegmentTree


def read_ckp(ckp_path: str, agent: object, model_name: str, buffer_size: int = 0):
    """读取已有数据, 如果报错, 可以先删除存档"""
    path = "/".join(ckp_path.split('/')[:-1])  # ckpt/sumo/PPO/
    if not os.path.exists(path):  # 检查路径在不在
        os.makedirs(path)
    if os.path.exists(ckp_path):  # 检查文件在不在  # 'ckpt/sumo/PPO/42_win32.pt'
        print('\033[34m[ checkpoint ]\033[0m 读取已有模型权重和训练数据...')
        checkpoint = torch.load(ckp_path)
        s_epoch = checkpoint["epoch"]
        s_episode = checkpoint["episode"]

        # 区分算法
        if 'DQN' in model_name:
            agent.q_net.load_state_dict(checkpoint["best_weight"])
        elif 'PPO' in model_name:
            assert not buffer_size, 'PPO 没有经验池!'
            agent.actor.load_state_dict(checkpoint["actor_best_weight"])
            agent.critic.load_state_dict(checkpoint["critic_best_weight"])
        elif 'SAC' in model_name:
            agent.actor.load_state_dict(checkpoint['actor_best_weight'])
            agent.critic_1.load_state_dict(checkpoint['critic_1_best_weight'])
            agent.critic_2.load_state_dict(checkpoint['critic_2_best_weight'])

        return_list = checkpoint["return_list"]
        time_list = checkpoint["time_list"]
        seed_list = checkpoint['seed_list']
        wait_time_list = checkpoint["wait_time_list"]
        queue_list = checkpoint["queue_list"]
        speed_list = checkpoint["speed_list"]

        if buffer_size:
            replay_buffer = checkpoint["replay_buffer"]
            return s_epoch, s_episode, return_list, wait_time_list, \
                queue_list, speed_list, time_list, seed_list, replay_buffer
        return s_epoch, s_episode, return_list, wait_time_list, \
            queue_list, speed_list, time_list, seed_list
    else:
        print('\033[34m[ checkpoint ]\033[0m 全新训练...')
        if buffer_size:
            return 0, 0, [], [], [], [], [], [], ReplayBuffer(buffer_size)
        return 0, 0, [], [], [], [], [], []


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:  # 逆向折算
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    advantage_list = torch.tensor(np.array(advantage_list), dtype=torch.float)
    advantage_list = (advantage_list - advantage_list.mean()) / (advantage_list.std() + 1e-5)
    return advantage_list


def save_plot_data(return_list, queue_list, wait_time_list,
                   speed_list, time_list, seed_list, 
                   ckpt_path, seed, pool_size=None):
    system_type = sys.platform  # 操作系统标识
    # 'ckpt/sumo/PPO~cvae/42_win32.pt'
    mission_name = ckpt_path.split('/')[1]
    alg_name = ckpt_path.split('/')[2]
    if not os.path.exists(f"data/plot_data/{mission_name}/{alg_name}/"):  # 路径不存在时创建
        os.makedirs(f"data/plot_data/{mission_name}/{alg_name}/")  # data/plot_data/sumo/PPO/
    log_path = f"data/plot_data/{mission_name}/{alg_name}/{seed}_{system_type}.csv"
    return_save = pd.DataFrame()
    return_save["Algorithm"] = [alg_name] * len(return_list)  # 算法名称
    return_save["Seed"] = seed_list
    return_save["Return"] = return_list
    return_save["Waiting time"] = wait_time_list
    return_save["Queue length"] = queue_list
    return_save["Mean speed"] = speed_list
    if pool_size:
        return_save["Pool size"] = pool_size
    return_save["Log time"] = time_list
    return_save.to_csv(log_path, index=False, encoding='utf-8-sig')


def train_PPO_agent(
    env: object,
    agent: object,
    writer: int,
    s_epoch: int,
    total_epochs: int,
    s_episode: int,
    total_episodes: int,
    return_list: list,
    queue_list: list,
    wait_time_list: list,
    speed_list: list,
    time_list: list,
    seed_list: list,
    seed: int,
    ckpt_path: str,
    # dynamic_model: object = None,
):
    """
    同策略, 没有经验池, 仅限演员评论员框架
    """
    start_time = time.time()
    best_score = -1e10  # 初始分数
    if not return_list:
        return_list = []
    for epoch in range(s_epoch, total_epochs):
        for episode in range(s_episode, total_episodes):
            episode_begin_time = time.time()
            transition_dict = {
                "states": [],
                "actions": [],
                "next_states": [],
                "rewards": [],
                "dones": [],
                "truncated": [],
            }
            episode_return = 0
            state, done, truncated = env.reset(seed=seed)[0], False, False
            while not (done | truncated):
                action = agent.take_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                transition_dict["states"].append(state)
                transition_dict["actions"].append(action)
                transition_dict["next_states"].append(next_state)
                transition_dict["rewards"].append(reward)
                transition_dict["dones"].append(done)
                transition_dict["truncated"].append(truncated)
                state = next_state
                episode_return += reward
            env.close()
            # 记录
            return_list.append(episode_return)
            wait_time_list.append(info["system_total_waiting_time"])
            queue_list.append(info["system_total_stopped"])
            speed_list.append(info["system_mean_speed"])
            time_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))
            seed_list.append(seed)
            agent.update(transition_dict)  # 更新参数

            if episode_return > best_score:
                actor_best_weight = agent.actor.state_dict()
                critic_best_weight = agent.critic.state_dict()
                best_score = episode_return
                best_weight = [actor_best_weight, critic_best_weight]

            # if dynamic_model:
                # PPO_rollout(agent, dynamic_model, transition_dict, roll_size, roll_step)

            # 存档
            save_PPO_data(writer, return_list, queue_list, wait_time_list, speed_list,
                          time_list, seed_list, ckpt_path, epoch, episode, best_weight, seed)
            # 记录时间
            episode_time = (time.time() - episode_begin_time) / 60
            # 打印回合信息
            print('\033[32m[ Seed %d, episode <%d/%d>, time spent: %.2f min ]\033[0m: return: %d, total waitting: %d'
                  % (seed, episode+1, total_episodes, episode_time, episode_return, info['system_total_waiting_time']))

            s_episode = 0
    agent.actor.load_state_dict(actor_best_weight)
    agent.critic.load_state_dict(critic_best_weight)
    total_time = round((time.time() - start_time) / 60, 2)
    print(f"\033[32m[ 总耗时 ]\033[0m {total_time} 分钟")
    # 如果检查点保存了回报列表, 可以不返回return_list
    return return_list, total_time // 60


def save_PPO_data(writer, return_list, queue_list, wait_time_list, speed_list,
                  time_list, seed_list, ckpt_path, epoch, episode, weight, seed):
    # wandb 存档
    actor_best_weight, critic_best_weight = weight
    if writer > 1:  # wandb 存档
        wandb.log({"_return_list": return_list[-1],
                   "waiting_time": wait_time_list[-1],
                   "queue_length": queue_list[-1],
                   "mean_speed": speed_list[-1],
                   })
    if writer > 0:  
        # 训练权重存档
        torch.save(
            {
                "epoch": epoch,
                "episode": episode,
                "actor_best_weight": actor_best_weight,
                "critic_best_weight": critic_best_weight,
                "return_list": return_list,
                "wait_time_list": wait_time_list,
                "queue_list": queue_list,
                "speed_list": speed_list,
                "time_list": time_list,
                "seed_list": seed_list,
            },
            ckpt_path,
        )

        # 绘图数据存档
        save_plot_data(return_list, queue_list, wait_time_list,
                    speed_list, time_list, seed_list, ckpt_path, seed)


def PPO_rollout(agent, dynamic_model, transition_dict, roll_size=2, roll_step=5):
    action_map = [
        torch.tensor([1, 0, 0, 0]),
        torch.tensor([0, 1, 0, 0]),
        torch.tensor([0, 0, 1, 0]),
        torch.tensor([0, 0, 0, 1]),
    ]
    model_dict = {
        "states": [],
        "actions": [],
        "next_states": [],
        "rewards": [],
        "dones": [],
        "truncated": [],
    }
    index = torch.randint(low=0, high=len(transition_dict["states"]), size=(roll_size,))
    for i in index:
        state = transition_dict["states"][i]
        for _ in range(roll_step):
            action = agent.take_action(state)
            next_state, reward = dynamic_model.step(state, action_map[action])
            model_dict["states"].append(state)
            model_dict["actions"].append(action)
            model_dict["next_states"].append(next_state)
            model_dict["rewards"].append(reward)
            model_dict["dones"].append(0)
            model_dict["truncated"].append(0)
            state = next_state
        agent.update(model_dict)  # 更新参数


def train_SAC_agent(
    env: object,
    agent: object,
    writer: int,
    s_epoch: int,
    total_epochs: int,
    s_episode: int,
    total_episodes: int,
    replay_buffer: object,
    minimal_size: int,
    batch_size: int,
    return_list: list,
    queue_list: list,
    wait_time_list: list,
    speed_list: list,
    time_list: list,
    seed_list: list,
    seed: int,
    ckpt_path: str,
    dynamic_model: object=None,
    baseline: int= -1000,
    # rollout_batch_size: int=100,
    # roll_step: int=2,
):
    """
    异策略
    """
    start_time = time.time()
    best_score = -1e10  # 初始分数
    for epoch in range(s_epoch, total_epochs):
        for episode in range(s_episode, total_episodes):
            env_pool = ReplayBuffer(5000)  # 环境模型经验
            episode_begin_time = time.time()
            episode_return = 0
            step = 0
            state, done, truncated = env.reset()[0], False, False
            while not (done | truncated):
                # if step % 50 == 0 and step > 0 and dynamic_model:
                #     train_model(dynamic_model, env_pool)
                #     rollout_model(agent, dynamic_model, roll_step, 
                #                   rollout_batch_size, env_pool, replay_buffer)
                action = agent.take_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                # if dynamic_model and step < 100:
                #     env_pool.add(state, action, reward, next_state, done, truncated)
                
                if dynamic_model and best_score < baseline:
                    mpc_action, pre_reward = mpc(dynamic_model, state, 2, give_reward=True)
                    if mpc_action == action:
                        reward += abs(pre_reward)
                    else:
                        reward += pre_reward
                
                replay_buffer.add(state, action, reward, next_state, done, truncated)
                state = next_state
                episode_return += reward
                if replay_buffer.size() > minimal_size:  # 确保先收集到一定量的数据再采样
                    b_s, b_a, b_r, b_ns, b_d, b_t = replay_buffer.sample(batch_size)
                    transition_dict = {
                        "states": b_s, "actions": b_a, "next_states": b_ns,
                        "rewards": b_r, "dones": b_d, "truncated": b_t,
                    }
                    agent.update(transition_dict)
                step += 1
            env.close()
            return_list.append(episode_return)
            wait_time_list.append(info["system_total_waiting_time"])
            queue_list.append(info["system_total_stopped"])
            speed_list.append(info["system_mean_speed"])
            time_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))
            seed_list.append(seed)
            assert len(wait_time_list) == len(queue_list) == len(
                return_list) == len(speed_list), '列表长度不一致!'

            if episode_return > best_score:
                actor_best_weight = agent.actor.state_dict()
                critic_1_best_weight = agent.critic_1.state_dict()
                critic_2_best_weight = agent.critic_2.state_dict()
                best_score = episode_return
                best_weight = [actor_best_weight,
                               critic_1_best_weight,
                               critic_2_best_weight]
            if writer > 0:  # 存档
                save_SAC_data(writer, replay_buffer, return_list, queue_list, wait_time_list,
                            speed_list, time_list, seed_list, ckpt_path, epoch, episode, best_weight, seed)
            episode_time = (time.time() - episode_begin_time) // 60
            print('\033[32m[ Seed %d, episode <%d/%d>, time spent: %d min ]\033[0m: return: %d, total waitting: %d'
                  % (seed, episode+1, total_episodes, episode_time, episode_return, info['system_total_waiting_time']))
        s_episode = 0
    agent.actor.load_state_dict(actor_best_weight)
    agent.critic_1.load_state_dict(critic_1_best_weight)
    agent.critic_2.load_state_dict(critic_2_best_weight)
    total_time = (time.time() - start_time) // 60
    print("\033[32m[ 总耗时 ]\033[0m %d分钟" % total_time)
    return return_list, total_time

def train_model(dynamic_mode, env_pool):
    ''' 训练动力模型适应当前环境， env_pool 是动力模型经验池 '''
    obs, action, reward, next_obs, done, truncated = env_pool.return_all_samples()
    one_hot_action = torch.nn.functional.one_hot(torch.tensor(action), 4)
    inputs = torch.cat([torch.tensor(obs), one_hot_action], dim=-1)
    reward = torch.tensor(reward).unsqueeze(-1)
    labels = torch.cat([torch.tensor(next_obs), reward], dim=-1)
    dynamic_mode.train(inputs, labels)

def rollout_model(agent, dynamic_model, rollout_step, 
                  rollout_batch_size, env_pool, agent_pool):
    ''' 增广agent经验 '''
    states, _, _, _, _, _ = env_pool.sample(rollout_batch_size)
    for state in states:
        for i in range(rollout_step):
            action = agent.take_action(state)
            one_hot_action = torch.nn.functional.one_hot(torch.tensor(action), 4)
            next_state, reward = dynamic_model.step(state, one_hot_action)
            agent_pool.add(state, action, reward, next_state, False, False)
            state = next_state

def save_SAC_data(writer, replay_buffer, return_list, queue_list, wait_time_list,
                  speed_list, time_list, seed_list, ckpt_path, epoch, episode, weight, seed):
    actor_best_weight, critic_1_best_weight, critic_2_best_weight = weight
    if writer > 1:  # wandb 存档
        wandb.log({"_return_list": return_list[-1],
                   "waiting_time": wait_time_list[-1],
                   "queue_length": queue_list[-1],
                   "mean_speed": speed_list[-1],
                   "pool_size": replay_buffer.size(),
                   })
    # 训练权重存档
    torch.save(
        {
            "epoch": epoch,
            "episode": episode,
            "actor_best_weight": actor_best_weight,
            "critic_1_best_weight": critic_1_best_weight,
            "critic_2_best_weight": critic_2_best_weight,
            "return_list": return_list,
            "wait_time_list": wait_time_list,
            "queue_list": queue_list,
            "speed_list": speed_list,
            "time_list": time_list,
            "seed_list": seed_list,
            "replay_buffer": replay_buffer,
        },
        ckpt_path,
    )

    # 绘图数据存档
    save_plot_data(return_list, queue_list, wait_time_list,
                   speed_list, time_list, seed_list, 
                   ckpt_path, seed, replay_buffer.size())


def train_DQN(
        env: object,
        agent: object,
        writer: bool,
        s_epoch: int,
        total_epoch: int,
        s_episode: int,
        total_episodes: int,
        replay_buffer: object,
        minimal_size: int,
        batch_size: int,
        return_list: list,
        queue_list: list,
        wait_time_list: list,
        speed_list: list,
        time_list: list,
        seed_list: list,
        seed: int,
        ckpt_path: str):
    start_time = time.time()
    best_score = -100  # 初始化最佳分数
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for epoch in range(s_epoch, total_epoch):
        for episode in range(s_episode, total_episodes):
            episode_begin_time = time.time()
            episode_return = 0
            state, done, truncated = env.reset()[0], False, False
            while not done | truncated:
                action = agent.take_action(state)
                max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995  # 平滑处理, 主要保留前一状态
                max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                next_state, reward, done, truncated, info = env.step(action)

                replay_buffer.add(state, action, reward, next_state, done, truncated)
                state = next_state
                episode_return += reward
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d, b_t = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s, 'actions': b_a, 'next_states': b_ns,
                        'rewards': b_r, 'dones': b_d, 'truncated': b_t,
                    }
                    agent.update(transition_dict)
                if episode_return > best_score:
                    best_weight = agent.q_net.state_dict()
                    best_score = episode_return
            env.close()
            return_list.append(episode_return)
            wait_time_list.append(info["system_total_waiting_time"])
            queue_list.append(info["system_total_stopped"])
            speed_list.append(info["system_mean_speed"])
            time_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))
            seed_list.append(seed)
            # 调整epsilon
            agent.epsilon = max(1 - epoch / (total_epoch / 3), 0.01)
            # 保存检查点
            save_DQN_data(writer, replay_buffer, return_list, queue_list, wait_time_list,
                          speed_list, time_list, seed_list, ckpt_path, epoch, episode, agent.epsilon,
                          best_weight, seed)
            episode_time = (time.time() - episode_begin_time) // 60
            print('\033[32m[ Seed %d, episode <%d/%d>, time spent: %d min ]\033[0m: return: %d, total waitting: %d'
                  % (seed, episode+1, total_episodes, episode_time, episode_return, info['system_total_waiting_time']))
            s_episode = 0
    
    agent.q_net.load_state_dict(best_weight)  # 应用最佳权重
    total_time = (time.time() - start_time) // 60
    print("\033[32m[ 总耗时 ]\033[0m %d分钟" % total_time)
    return return_list, total_time


def save_DQN_data(writer, replay_buffer, return_list, queue_list, wait_time_list,
                  speed_list, time_list, seed_list, ckpt_path, epoch, episode, epsilon,
                  best_weight, seed):
    # wandb 存档
    if writer:
        wandb.log({"_return_list": return_list[-1],
                   "waiting_time": wait_time_list[-1],
                   "queue_length": queue_list[-1],
                   "mean_speed": speed_list[-1],
                   "pool_size": replay_buffer.size(),
                   })
    # 训练权重存档
    torch.save({
        'epoch': epoch,
        'episode': episode,
        'best_weight': best_weight,
        'epsilon': epsilon,
        "return_list": return_list,
        "wait_time_list": wait_time_list,
        "queue_list": queue_list,
        "speed_list": speed_list,
        "time_list": time_list,
        "seed_list": seed_list,
        "replay_buffer": replay_buffer,
    }, ckpt_path)

    # 绘图数据存档
    save_plot_data(return_list, queue_list, wait_time_list,
                   speed_list, time_list, seed_list, ckpt_path, seed)


class ReplayBuffer:
    """异策略的经验缓存"""

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state: dict, action: dict, reward: float, next_state: dict, done: dict, truncated: dict):
        self.buffer.append((state, action, reward, next_state, done, truncated))

    def sample(self, batch_size):
        if batch_size > self.size():
            return self.return_all_samples()
        else:
            transitions = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done, truncated = zip(*transitions)
            return np.array(state), action, reward, np.array(next_state), done, truncated

    def size(self):
        return len(self.buffer)
    
    def return_all_samples(self):
        all_transitions = list(self.buffer)
        state, action, reward, next_state, done, truncated = zip(*all_transitions)
        return np.array(state), action, reward, np.array(next_state), done, truncated

def get_action(time: int, flag: int, max_action: int, action_index: int):
    '''fixed-time 专用，每 flag 秒切换一次相位'''
    if time % flag == 0 and time > 0:
        if (action_index + 1) <= max_action - 1:
            action_index += 1
        else:
            action_index = 0
    return action_index

class DynamicEnv:
    def __init__(self, state_model, reward_model, device) -> None:
        self.state_model = state_model
        self.reward_model = reward_model
        self.state_optimizer = torch.optim.Adam(state_model.parameters(), 
                                                lr=1e-3, weight_decay=1e-4)
        self.reward_optimizer = torch.optim.Adam(reward_model.parameters(), 
                                                lr=1e-3, weight_decay=1e-4)
        self.criterion = torch.nn.MSELoss()
        self.device = device
        
    def step(self, state, action):
        ''' 输入独热向量化的 action '''
        if isinstance(action, int):
            action = torch.nn.functional.one_hot(torch.tensor(action), 4)
        inputs = torch.cat([torch.tensor(state), torch.tensor(action)], dim=-1).to(self.device)
        self.state_model.eval()
        self.reward_model.eval()
        next_state = self.state_model(inputs)[..., :-1]
        reward = self.reward_model(inputs)[..., -1]
        return next_state.detach().cpu().numpy()[0], reward.item()
    
    def _forward(self, inputs):
        next_state = self.state_model(inputs)[..., :-1]
        reward = self.reward_model(inputs)[..., -1]
        return next_state, reward
    
    def train(self, inputs, labels, num_epochs=5):
        train_dataset = TensorDataset(inputs, labels)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        self.state_model.train()
        self.reward_model.train()
        for _ in range(num_epochs):
            for inputs, label in train_loader:
                self.state_optimizer.zero_grad()
                self.reward_optimizer.zero_grad()
                next_state, reward = self._forward(inputs.to(self.device))
                state_loss = self.criterion(next_state, label[..., :-1].to(self.device))
                state_loss.backward()
                self.state_optimizer.step()
                reward_loss = self.criterion(reward, label[..., -1].to(self.device))
                reward_loss.backward()
                self.reward_optimizer.step()
                

def mpc(dynamic_model, state, horizon=3, actions = [0, 1, 2, 3], give_reward=False):
    """
    MPC函数
    :param state: 当前状态
    :param horizon: 模拟的步数
    :return: 最优的第一步决策
    """
    
    def simulate(state, depth):
        if depth == 0:
            return 0
        
        best_reward = -torch.inf
        for action in actions:
            one_hot_action =  torch.nn.functional.one_hot(torch.tensor(action), 4)
            next_state, reward = dynamic_model.step(state, one_hot_action)
            reward = reward + simulate(next_state, depth - 1)
            if reward > best_reward:
                best_reward = reward
        
        return best_reward
    
    best_action = 0
    best_reward = -torch.inf
    for action in actions:
        one_hot_action = torch.nn.functional.one_hot(torch.tensor(action), 4)
        next_state, now_reward = dynamic_model.step(state, one_hot_action)
        reward = now_reward + simulate(next_state, horizon - 1)
        if reward > best_reward:
            best_reward = now_reward
            best_action = action
    if give_reward:
        return best_action, now_reward
    return best_action

'''
class RDQN_ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        n_step: int = 1, 
        gamma: float = 0.99
    ):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self, 
        obs: np.ndarray, 
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]
        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            # for N-step Learning
            indices=idxs,
        )
    
    def sample_batch_from_idxs(self, idxs: np.ndarray) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )
        
    def length(self):
        return self.obs_buf.shape[0]
        
class PrioritizedReplayBuffer(RDQN_ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6,
        n_step: int = 1, 
        gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0
        
        super().__init__()
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(
        self, 
        obs: np.ndarray, 
        act: int, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done)
        
        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        
        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert self.length() >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < self.length()

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, self.length() - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.length()) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.length()) ** (-beta)
        weight = weight / max_weight
        
        return weight
'''