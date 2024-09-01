from typing import Tuple, List, Dict
import os
import sys
import random
import collections
from utils.CEA import *
import torch
import numpy as np
import pandas as pd
import time
from utils.segment_tree import SumSegmentTree, MinSegmentTree

def read_ckp(ckp_path: str, agent: object, model_name: str, buffer_size: int = 0):
    """读取已有数据, 如果报错, 可以先删除存档"""
    path = "/".join(ckp_path.split('/')[:-1])
    if not os.path.exists(path):  # 检查路径在不在
        os.makedirs(path)
    if os.path.exists(ckp_path):  # 检查文件在不在
        print('\n\033[34m[ checkpoint ]\033[0m 读取已有模型权重和训练数据...')
        checkpoint = torch.load(ckp_path)
        s_epoch = checkpoint["epoch"]
        s_episode = checkpoint["episode"]

        # 区分算法
        if 'DQN' in model_name:
            agent.q_net.load_state_dict(checkpoint["best_weight"])
        elif 'PPO' in model_name or 'DDPG' in model_name:
            assert not buffer_size, 'PPO 没有经验池!'
            agent.actor.load_state_dict(checkpoint["actor_best_weight"])
            agent.critic.load_state_dict(checkpoint["critic_best_weight"])
        elif 'SAC' in model_name or 'TD3' in model_name or 'CEA' in model_name:
            agent.actor.load_state_dict(checkpoint['actor_best_weight'])
            agent.critic_1.load_state_dict(checkpoint['critic_1_best_weight'])
            agent.critic_2.load_state_dict(checkpoint['critic_2_best_weight'])

        return_list = checkpoint["return_list"]
        time_list = checkpoint["time_list"]
        seed_list = checkpoint['seed_list']

        if buffer_size:
            replay_buffer = checkpoint["replay_buffer"]
            return s_epoch, s_episode, return_list, time_list, seed_list, replay_buffer
        return s_epoch, s_episode, return_list, time_list, seed_list
    else:
        print('\n\033[34m[ checkpoint ]\033[0m 无存档')
        if buffer_size:
            return 0, 0, [], [], [], ReplayBuffer(buffer_size)
        return 0, 0, [], [], []


def save_plot_data(return_list, time_list, seed_list, ckpt_path, seed, pool_list=None):
    system_type = sys.platform  # 操作系统标识
    # ckpt/SAC/big-intersection_42_win32.pt
    mission_name = ckpt_path.split('/')[1]
    alg_name = ckpt_path.split('/')[2]  # 在本项目路径命名中，第二个是算法名
    # data/plot_data/highway/SAC/
    file_path = f"data/plot_data/{mission_name}/{alg_name}"
    if not os.path.exists(file_path):  # 路径不存在时创建
        os.makedirs(file_path)
    log_path = f"{file_path}/{seed}_{system_type}.csv"
    return_save = pd.DataFrame()
    return_save["Algorithm"] = [alg_name] * len(return_list)  # 算法名称
    return_save["Seed"] = seed_list
    return_save["Return"] = return_list
    if pool_list:
        return_save["Pool size"] = pool_list
    return_save["Log time"] = time_list
    return_save.to_csv(log_path, index=False, encoding='utf-8-sig')


class ReplayBuffer:
    """CEA."""

    def __init__(
        self,
        states_dim: int,
        action_dim: int,
        size: int,
        batch_size: int = 32,
    ):
        self.states_buf = np.zeros([size, states_dim], dtype=np.float32)
        self.next_states_buf = np.zeros([size, states_dim], dtype=np.float32)
        self.actions_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.rewards_buf = np.zeros([size], dtype=np.float32)
        self.dones_buf = np.zeros(size, dtype=np.float32)
        self.truncated_buf = np.zeros(size, dtype=np.float32)
        self.exp_type_buf = np.zeros(size, dtype=np.float32)  # 虚拟经验是 1，真实经验是 0
        # 没被抽的是 0 , cf_spd: counterfactual_sampled
        self.cf_spd_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        self.capacity = size

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        truncated: bool,
        exp_type: bool = 0,
        cf_spd: bool = 0,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (state, action, reward, next_state, done, exp_type, cf_spd)
        self.states_buf[self.ptr] = state
        self.next_states_buf[self.ptr] = next_state
        self.actions_buf[self.ptr] = action
        self.rewards_buf[self.ptr] = reward
        self.dones_buf[self.ptr] = done
        self.truncated_buf[self.ptr] = truncated
        self.exp_type_buf[self.ptr] = exp_type
        self.cf_spd_buf[self.ptr] = cf_spd

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        return transition
        
    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        weights = np.ones(self.cf_spd_buf[idxs].shape)
        return dict(
            states=self.states_buf[idxs],
            next_states=self.next_states_buf[idxs],
            actions=self.actions_buf[idxs],
            rewards=self.rewards_buf[idxs],
            dones=self.dones_buf[idxs],
            truncated=self.truncated_buf[idxs],
            exp_type=self.exp_type_buf[idxs],
            cf_spd=self.cf_spd_buf[idxs],
            weights=weights,
            indices=idxs,
        )

    def retrieve_real_experiences(self) -> Dict[str, np.ndarray]:
        """采样真实经验，即exp_type值为0的"""
        assert len(self) > 0

        indices = np.where(self.exp_type_buf == 0)[0][:self.size-1]

        states = self.states_buf[indices]
        next_states = self.next_states_buf[indices]
        actions = self.actions_buf[indices]
        rewards = self.rewards_buf[indices]
        dones = self.dones_buf[indices]
        truncated = self.truncated_buf[indices],

        return dict(
            states=states,
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            truncated=truncated,
        )

    def sample_new_real_exp(self, batch_size):
        '''采样未经反事实推断的真实经验，即exp_type和spd_buf值同时为0的'''
        assert len(self) > 0
        indices = np.where((self.exp_type_buf == 0) & (self.cf_spd_buf == 0))[0][:self.size-1]
        if len(indices) >= batch_size:
            sampled_indices = np.random.choice(indices, size=batch_size, replace=False)
        else:
            sampled_indices = np.random.choice(indices, size=len(indices), replace=False)
        states = self.states_buf[sampled_indices]
        next_states = self.next_states_buf[sampled_indices]
        actions = self.actions_buf[sampled_indices]
        rewards = self.rewards_buf[sampled_indices]
        dones = self.dones_buf[sampled_indices]
        truncated = self.truncated_buf[sampled_indices]
        self.cf_spd_buf[sampled_indices] = 1  # 标记为已经采样过
        return dict(
            states=states,
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            truncated=truncated,
        )
    
    def __len__(self) -> int:
        return self.size

class PrioritizedReplayBuffer(ReplayBuffer):
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
        states_dim: int, 
        action_dim:int,
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6, 
    ):
        """Initialization."""
        assert alpha >= 0
        
        super().__init__(states_dim, action_dim, size, batch_size)
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
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool,
        truncated: bool,
        exp_type: bool,
        cf_sped: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store experience and priority."""
        transition = super().store(state, action, reward, next_state, done, truncated, exp_type, cf_sped)
        
        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        
        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta >= 0
        
        indices = self._sample_proportional()
        
        states = self.states_buf[indices]
        next_states = self.next_states_buf[indices]
        actions = self.actions_buf[indices]
        rewards = self.rewards_buf[indices]
        dones = self.dones_buf[indices]
        truncated = self.truncated_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            states=states,
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            truncated=truncated,
            weights=weights,
            indices=indices,
        )
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
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
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight

# -------------- SAC, TD3, CEA ----------------

def train_SAC_agent(
    env: object,
    agent: object,
    writer: int,
    s_epoch: int,
    total_epochs: int,
    s_episode: int,
    total_episodes: int,
    replay_buffer: ReplayBuffer,
    minimal_size: int,
    return_list: list,
    time_list: list,
    seed_list: list,
    seed: int,
    ckpt_path: str,
    sta: bool = False,
    per: bool = False,
    num_new_samples: int=0,
    threshold_ratio: int=0.4,
):
    '''支持SAC, CEA, TD3'''
    pool_list = []
    start_time = time.time()
    best_score = -1e10  # 初始分数
    total_step = 0
    return_list = [] if not return_list else return_list
    flag = True
    for epoch in range(s_epoch, total_epochs):  # 实际上没用，一般只设定 1 个epoch
        for episode in range(s_episode, total_episodes):
            episode_return = 0
            ep_start_time = time.time()
            state, done, truncated = env.reset()[0], False, False  # 这里不给env seed避免对单个种子过拟合
            # 执行单轮仿真
            while not (done | truncated):
                action = agent.take_action(state) if replay_buffer.size > minimal_size else env.action_space.sample()
                action = np.array(action).reshape(-1)  # 检查action类型
                next_state, reward, done, truncated, info = env.step(action)
                total_step += 1
                next_state = next_state.reshape(-1)
                transition = [state, action, reward, next_state, done, truncated, 0, 0]
                replay_buffer.store(*transition)
                state = next_state
                episode_return += float(reward)
                if replay_buffer.size > minimal_size:  # 确保先收集到一定量的数据再采样
                    if flag:
                        print('[ 开始抽样训练 ]')
                        flag = False
                    transition_dict = replay_buffer.sample_batch()
                    combined_loss = agent.update(transition_dict)
                    # * 执行 CEA
                    if sta and total_step % (4 * replay_buffer.batch_size) == 0 and replay_buffer.size < replay_buffer.capacity:
                        replay_buffer = counterfactual_exp_expand(replay_buffer, sta,
                                                                replay_buffer.batch_size, env,
                                                                num_new_samples, threshold_ratio)
                    if per:
                        loss_for_prior = combined_loss.numpy()
                        new_priorities = loss_for_prior + 1e-6
                        replay_buffer.update_priorities(transition_dict['indices'], new_priorities)
            # 记录数据
            return_list.append(episode_return)
            time_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))
            seed_list.append(seed)
            pool_list.append(replay_buffer.size)

            if episode_return > best_score:  # 保存最佳分数
                actor_best_weight = agent.actor.state_dict()
                critic_1_best_weight = agent.critic_1.state_dict()
                critic_2_best_weight = agent.critic_2.state_dict()
                best_score = episode_return
                best_weight = [actor_best_weight, critic_1_best_weight, critic_2_best_weight]
            if writer > 0:  # 存档
                save_SAC_data(writer, replay_buffer, return_list, pool_list, time_list, seed_list, 
                              ckpt_path, epoch, episode, best_weight, agent, seed)
            
            if episode % 10 == 0:
                episode_time = (time.time() - ep_start_time) / 6
                print('\033[32m[ %d, %d/%d, %.2f min ]\033[0m: return: %d, pool: %d'
                      % (seed, episode+1, total_episodes, episode_time, np.mean(return_list[-10:]), replay_buffer.size))
        s_episode = 0
    env.close()
    
    # agrnt存档
    if not flag:
        agent.actor.load_state_dict(actor_best_weight)
        agent.critic_1.load_state_dict(critic_1_best_weight)
        agent.critic_2.load_state_dict(critic_2_best_weight)
    
    # 打印总时间
    total_time = (time.time() - start_time) // 60
    print("\033[32m[ 总耗时 ]\033[0m %d分钟" % total_time)
    
    return return_list, total_time


def save_SAC_data(writer, replay_buffer, return_list, pool_list, time_list,
                  seed_list, ckpt_path, epoch, episode, weight, agent, seed):
    actor_best_weight, critic_1_best_weight, critic_2_best_weight = weight
    # 训练权重存档
    torch.save(
        {
            "epoch": epoch,
            "episode": episode,
            "actor_best_weight": actor_best_weight,
            "critic_1_best_weight": critic_1_best_weight,
            "critic_2_best_weight": critic_2_best_weight,
            "return_list": return_list,
            "time_list": time_list,
            "seed_list": seed_list,
            "pool_list": pool_list,
            "replay_buffer": replay_buffer,
            "agent": agent,
        },
        ckpt_path,
    )
    # 绘图数据存档
    save_plot_data(return_list, time_list, seed_list,
                    ckpt_path, seed, pool_list)

# -------------- DDPG ---------------

def train_DDPG_agent(
    env: object,
    agent: object,
    writer: int,
    s_epoch: int,
    total_epochs: int,
    s_episode: int,
    total_episodes: int,
    replay_buffer: ReplayBuffer,
    minimal_size: int,
    return_list: list,
    time_list: list,
    seed_list: list,
    seed: int,
    ckpt_path: str,
    sta: bool = False,
    per: bool = False,
    num_new_samples: int=0,
    threshold_ratio: int=0.4,
):
    '''支持DDPG'''
    pool_list = []
    start_time = time.time()
    best_score = -1e10  # 初始分数
    total_step = 0
    return_list = [] if not return_list else return_list
    flag = True
    for epoch in range(s_epoch, total_epochs):  # 实际上没用，一般只设定 1 个 epoch
        for episode in range(s_episode, total_episodes):
            episode_return = 0
            ep_start_time = time.time()
            state, done, truncated = env.reset()[0], False, False  # 这里不给env seed避免对单个种子过拟合
            # 执行单轮仿真
            while not (done | truncated):
                action = agent.take_action(state) if replay_buffer.size > minimal_size else env.action_space.sample()
                action = np.array(action).reshape(-1)
                next_state, reward, done, truncated, info = env.step(action)
                total_step += 1
                next_state = next_state.reshape(-1)
                transition = [state, action, reward, next_state, done, truncated, 0, 0]
                replay_buffer.store(*transition)
                state = next_state
                episode_return += reward
                if replay_buffer.size > minimal_size:  # 确保先收集到一定量的数据再采样
                    if flag:
                        print('[ 开始抽样训练 ]')
                        flag = False
                    transition_dict = replay_buffer.sample_batch()
                    combined_loss = agent.update(transition_dict)
                    # 执行PER
                    if per:
                        loss_for_prior = combined_loss.numpy()
                        new_priorities = loss_for_prior + 1e-6
                        replay_buffer.update_priorities(transition_dict['indices'], new_priorities)
            # * 执行 CEA
            if sta and episode % 20 == 0 and replay_buffer.size < replay_buffer.capacity: # replay_buffer.size < minimal_size:
                replay_buffer = counterfactual_exp_expand(replay_buffer, sta,
                                                        replay_buffer.batch_size, env,
                                                        num_new_samples, threshold_ratio)
            # 记录数据
            return_list.append(episode_return)
            time_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))
            seed_list.append(seed)
            pool_list.append(replay_buffer.size)

            if episode_return > best_score:  # 保存最佳分数
                actor_best_weight = agent.actor.state_dict()
                critic_best_weight = agent.critic.state_dict()
                best_score = episode_return
                best_weight = [actor_best_weight, critic_best_weight]
            if writer > 0:  # 存档
                save_DDPG_data(writer, replay_buffer, return_list, pool_list, time_list, seed_list, 
                              ckpt_path, epoch, episode, best_weight, agent, seed)
            
            if episode % 10 == 0:
                episode_time = (time.time() - ep_start_time) / 6
                print('\033[32m[ %d, %d/%d, %.2f min ]\033[0m: return: %d, pool: %d'
                      % (seed, episode+1, total_episodes, episode_time, np.mean(return_list[-10:]), replay_buffer.size))
        s_episode = 0
    env.close()
    # agrnt存档
    if not flag:
        agent.actor.load_state_dict(actor_best_weight)
        agent.critic.load_state_dict(critic_best_weight)
    # 打印总时间
    total_time = (time.time() - start_time) // 60
    print("\033[32m[ 总耗时 ]\033[0m %d分钟" % total_time)
    return return_list, total_time

def save_DDPG_data(writer, replay_buffer, return_list, pool_list, time_list,
                  seed_list, ckpt_path, epoch, episode, weight, agent, seed):
    actor_best_weight, critic_best_weight = weight
    # 训练权重存档
    torch.save(
        {
            "epoch": epoch,
            "episode": episode,
            "actor_best_weight": actor_best_weight,
            "critic_best_weight": critic_best_weight,
            "return_list": return_list,
            "time_list": time_list,
            "seed_list": seed_list,
            "pool_list": pool_list,
            "replay_buffer": replay_buffer,
            "agent": agent,
        },
        ckpt_path,
    )
    # 绘图数据存档
    save_plot_data(return_list, time_list, seed_list,
                    ckpt_path, seed, pool_list)

# -------------- PPO ----------------

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

def train_PPO_agent(
    env: object,
    agent: object,
    writer: int,
    s_epoch: int,
    total_epochs: int,
    s_episode: int,
    total_episodes: int,
    return_list: list,
    time_list: list,
    seed_list: list,
    seed: int,
    ckpt_path: str,
):
    """
    同策略, 没有经验池, 仅限演员评论员框架
    """
    start_time = time.time()
    best_score = -1e10  # 初始分数
    return_list = [] if not return_list else return_list
    flag = True
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
            ep_start_time = time.time()
            state, done, truncated = env.reset()[0], False, False
            while not (done | truncated):
                action = agent.take_action(state)
                action = np.array(action.cpu()) if not isinstance(action, np.ndarray) else action  # 检查action类型
                next_state, reward, done, truncated, info = env.step(action)
                transition_dict["states"].append(state)
                transition_dict["actions"].append(action)
                transition_dict["next_states"].append(next_state)
                transition_dict["rewards"].append(reward)
                transition_dict["dones"].append(done)
                transition_dict["truncated"].append(truncated)
                state = next_state
                episode_return += reward
            # 记录
            return_list.append(episode_return)
            time_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))
            seed_list.append(seed)
            agent.update(transition_dict)  # 更新参数

            if episode_return > best_score:
                actor_best_weight = agent.actor.state_dict()
                critic_best_weight = agent.critic.state_dict()
                best_score = episode_return
                best_weight = [actor_best_weight, critic_best_weight]

            # 存档
            if writer > 0:  # 存档
                save_PPO_data(writer, return_list, time_list, seed_list, 
                            ckpt_path, epoch, episode, best_weight, agent, seed)
            # 记录时间
            if episode % 10 == 0:
                duration_time = (time.time() - ep_start_time) / 6
                print('\033[32m[ %d, <%d/%d>, %.2f min ]\033[0m: return: %d'
                  % (seed, episode+1, total_episodes, duration_time, np.mean(return_list[-10:])))

            s_episode = 0
    env.close()
    if not flag:
        agent.actor.load_state_dict(actor_best_weight)
        agent.critic.load_state_dict(critic_best_weight)
    total_time = time.time() - start_time
    print(f"\033[32m[ 总耗时 ]\033[0m {(total_time / 60):.2f}分钟")
    # 如果检查点保存了回报列表, 可以不返回return_list
    return return_list, total_time // 60


def save_PPO_data(writer, return_list, time_list, seed_list, 
                  ckpt_path, epoch, episode, weight, agent, seed):
    # wandb 存档
    actor_best_weight, critic_best_weight = weight
    # 训练权重存档
    torch.save(
        {
            "epoch": epoch,
            "episode": episode,
            "actor_best_weight": actor_best_weight,
            "critic_best_weight": critic_best_weight,
            "return_list": return_list,
            "time_list": time_list,
            "seed_list": seed_list,
            "agent": agent,
        },
        ckpt_path,
    )
    # 绘图数据存档
    save_plot_data(return_list, time_list, seed_list, ckpt_path, seed)