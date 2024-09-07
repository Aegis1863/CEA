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
    path = "/".join(ckp_path.split('/')[:-1])  # ckpt/sumo/PPO/
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.exists(ckp_path):  # 'ckpt/sumo/PPO/42_win32.pt'
        print('\033[34m[ checkpoint ]\033[0m 读取已有模型权重和训练数据...')
        checkpoint = torch.load(ckp_path)
        s_epoch = checkpoint["epoch"]
        s_episode = checkpoint["episode"]

        # 区分算法
        if 'DQN' in model_name:
            agent.q_net.load_state_dict(checkpoint["best_weight"])
        elif 'PPO' in model_name:
            assert not buffer_size, 'PPO do not have exp pool!'
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
        print('\033[34m[ checkpoint ]\033[0m brand new train...')
        if buffer_size:
            return 0, 0, [], [], [], [], [], [], ReplayBuffer(buffer_size)
        return 0, 0, [], [], [], [], [], []


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    advantage_list = torch.tensor(np.array(advantage_list), dtype=torch.float)
    advantage_list = (advantage_list - advantage_list.mean()) / (advantage_list.std() + 1e-5)
    return advantage_list


def save_plot_data(return_list, queue_list, wait_time_list,
                   speed_list, time_list, seed_list, 
                   ckpt_path, seed, pool_size=None):
    system_type = sys.platform
    # 'ckpt/sumo/PPO~cvae/42_win32.pt'
    mission_name = ckpt_path.split('/')[1]
    alg_name = ckpt_path.split('/')[2]
    if not os.path.exists(f"data/plot_data/{mission_name}/{alg_name}/"):
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
    For PPO
    """
    start_time = time.time()
    best_score = -1e10
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
            # log
            return_list.append(episode_return)
            wait_time_list.append(info["system_total_waiting_time"])
            queue_list.append(info["system_total_stopped"])
            speed_list.append(info["system_mean_speed"])
            time_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))
            seed_list.append(seed)
            agent.update(transition_dict)  # update agent

            if episode_return > best_score:
                actor_best_weight = agent.actor.state_dict()
                critic_best_weight = agent.critic.state_dict()
                best_score = episode_return
                best_weight = [actor_best_weight, critic_best_weight]
                
            # checkpoint
            save_PPO_data(writer, return_list, queue_list, wait_time_list, speed_list,
                          time_list, seed_list, ckpt_path, epoch, episode, best_weight, seed)

            episode_time = (time.time() - episode_begin_time) / 60
            print('\033[32m[ Seed %d, episode <%d/%d>, time spent: %.2f min ]\033[0m: return: %d, total waitting: %d'
                  % (seed, episode+1, total_episodes, episode_time, episode_return, info['system_total_waiting_time']))

            s_episode = 0
    agent.actor.load_state_dict(actor_best_weight)
    agent.critic.load_state_dict(critic_best_weight)
    total_time = round((time.time() - start_time) / 60, 2)
    print(f"\033[32m[ Total time ]\033[0m {total_time}  min")
    return return_list, total_time // 60


def save_PPO_data(writer, return_list, queue_list, wait_time_list, speed_list,
                  time_list, seed_list, ckpt_path, epoch, episode, weight, seed):
    actor_best_weight, critic_best_weight = weight
    if writer > 1:
        wandb.log({"_return_list": return_list[-1],
                   "waiting_time": wait_time_list[-1],
                   "queue_length": queue_list[-1],
                   "mean_speed": speed_list[-1],
                   })
    if writer > 0:  
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
        agent.update(model_dict)


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
):
    """
    For MBPO (SAC)
    """
    start_time = time.time()
    best_score = -1e10
    for epoch in range(s_epoch, total_epochs):
        for episode in range(s_episode, total_episodes):
            env_pool = ReplayBuffer(5000)
            episode_begin_time = time.time()
            episode_return = 0
            step = 0
            state, done, truncated = env.reset()[0], False, False
            while not (done | truncated):
                action = agent.take_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                
                if dynamic_model and best_score < baseline:
                    mpc_action, pre_reward = mpc(dynamic_model, state, 2, give_reward=True)
                    if mpc_action == action:
                        reward += abs(pre_reward)
                    else:
                        reward += pre_reward
                
                replay_buffer.add(state, action, reward, next_state, done, truncated)
                state = next_state
                episode_return += reward
                if replay_buffer.size() > minimal_size:
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
                return_list) == len(speed_list), 'Inconsistent list lengths!'

            if episode_return > best_score:
                actor_best_weight = agent.actor.state_dict()
                critic_1_best_weight = agent.critic_1.state_dict()
                critic_2_best_weight = agent.critic_2.state_dict()
                best_score = episode_return
                best_weight = [actor_best_weight,
                               critic_1_best_weight,
                               critic_2_best_weight]
            if writer > 0:
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
    print("\033[32m[ Total time ]\033[0m %d分钟" % total_time)
    return return_list, total_time

def train_model(dynamic_mode, env_pool):
    ''' Train the dynamic model to adapt to the current environment. env_pool is the dynamic model experience pool. '''
    obs, action, reward, next_obs, done, truncated = env_pool.return_all_samples()
    one_hot_action = torch.nn.functional.one_hot(torch.tensor(action), 4)
    inputs = torch.cat([torch.tensor(obs), one_hot_action], dim=-1)
    reward = torch.tensor(reward).unsqueeze(-1)
    labels = torch.cat([torch.tensor(next_obs), reward], dim=-1)
    dynamic_mode.train(inputs, labels)

def rollout_model(agent, dynamic_model, rollout_step, 
                  rollout_batch_size, env_pool, agent_pool):
    ''' Augmenting agent experience '''
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
    if writer > 1:
        wandb.log({"_return_list": return_list[-1],
                   "waiting_time": wait_time_list[-1],
                   "queue_length": queue_list[-1],
                   "mean_speed": speed_list[-1],
                   "pool_size": replay_buffer.size(),
                   })
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
    best_score = -100
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
                max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995  # Smoothing, mainly retaining the previous state
                max_q_value_list.append(max_q_value)
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
            agent.epsilon = max(1 - epoch / (total_epoch / 3), 0.01)
            save_DQN_data(writer, replay_buffer, return_list, queue_list, wait_time_list,
                          speed_list, time_list, seed_list, ckpt_path, epoch, episode, agent.epsilon,
                          best_weight, seed)
            episode_time = (time.time() - episode_begin_time) // 60
            print('\033[32m[ Seed %d, episode <%d/%d>, time spent: %d min ]\033[0m: return: %d, total waitting: %d'
                  % (seed, episode+1, total_episodes, episode_time, episode_return, info['system_total_waiting_time']))
            s_episode = 0
    
    agent.q_net.load_state_dict(best_weight)
    total_time = (time.time() - start_time) // 60
    print("\033[32m[ Total time ]\033[0m %d min" % total_time)
    return return_list, total_time


def save_DQN_data(writer, replay_buffer, return_list, queue_list, wait_time_list,
                  speed_list, time_list, seed_list, ckpt_path, epoch, episode, epsilon,
                  best_weight, seed):
    if writer:
        wandb.log({"_return_list": return_list[-1],
                   "waiting_time": wait_time_list[-1],
                   "queue_length": queue_list[-1],
                   "mean_speed": speed_list[-1],
                   "pool_size": replay_buffer.size(),
                   })
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

    save_plot_data(return_list, queue_list, wait_time_list,
                   speed_list, time_list, seed_list, ckpt_path, seed)


class ReplayBuffer:

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
    '''Fixed-time dedicated, switching phase every flag seconds'''
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
        ''' input one-hot vectorized action '''
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
    MPC
    ---
    :param state: current state
    :param horizon: planing steps
    :return: strategy
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