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
import numpy as np
import torch
import seaborn as sns
from utils.segment_tree import SumSegmentTree, MinSegmentTree
from typing import List, Dict

def read_ckp(ckp_path: str, agent: object, model_name: str, buffer_size: int = 0):
    """读取已有数据, 如果报错, 可以先删除存档"""
    path = "/".join(ckp_path.split('/')[:-1])
    if not os.path.exists(path):  # 检查路径在不在
        os.makedirs(path)
    if os.path.exists(ckp_path):  # 检查文件在不在
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

        if buffer_size:
            replay_buffer = checkpoint["replay_buffer"]
            return s_epoch, s_episode, return_list, time_list, seed_list, replay_buffer
        return s_epoch, s_episode, return_list, time_list, seed_list
    else:
        print('\n\033[34m[ checkpoint ]\033[0m 全新训练...')
        if buffer_size:
            return 0, 0, [], [], [], ReplayBuffer(buffer_size)
        return 0, 0, [], [], []


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


def save_plot_data(return_list, time_list, seed_list, ckpt_path, seed, pool_list=None):
    system_type = sys.platform  # 操作系统标识
    # ckpt/SAC/big-intersection_42_win32.pt
    mission_name = ckpt_path.split('/')[1]
    alg_name = ckpt_path.split('/')[2]  # 在本项目路径命名中，第二个是算法名
    file_path = f"data/plot_data/{mission_name}/{alg_name}"  # data/plot_data/highway/SAC/
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
            state = state.reshape(-1)
            while not (done | truncated):
                action = agent.take_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                next_state = next_state.reshape(-1)
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
            if writer > 0:
                save_PPO_data(writer, return_list, time_list, seed_list, 
                            ckpt_path, epoch, episode, best_weight, seed)
            # 记录时间
            if episode % 10 == 0:
                # 打印回合信息
                duration_time = (time.time() - episode_begin_time) / 6
                print('\033[32m[ %d, <%d/%d>, %.2f min ]\033[0m: return: %d'
                  % (seed, episode+1, total_episodes, duration_time, np.mean(return_list[-10:])))

            s_episode = 0
    env.close()
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
    time_list: list,
    seed_list: list,
    seed: int,
    ckpt_path: str,
):
    """
    异策略
    """
    pool_list = []
    start_time = time.time()
    best_score = -1e10  # 初始分数
    for epoch in range(s_epoch, total_epochs):
        for episode in range(s_episode, total_episodes):
            episode_begin_time = time.time()
            episode_return = 0
            step = 0
            state, done, truncated = env.reset(seed=seed)[0], False, False
            state = state.reshape(-1)
            while not (done | truncated):
                action = agent.take_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                next_state = next_state.reshape(-1)
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
            return_list.append(episode_return)
            time_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))
            seed_list.append(seed)
            pool_list.append(replay_buffer.size())

            if episode_return > best_score:
                actor_best_weight = agent.actor.state_dict()
                critic_1_best_weight = agent.critic_1.state_dict()
                critic_2_best_weight = agent.critic_2.state_dict()
                best_score = episode_return
                best_weight = [actor_best_weight,
                               critic_1_best_weight,
                               critic_2_best_weight]
            if writer > 0:  # 存档
                save_SAC_data(writer, replay_buffer, return_list, pool_list,
                              time_list, seed_list, ckpt_path, epoch, episode, best_weight, seed)
            episode_time = (time.time() - episode_begin_time) // 60
            if episode % 10 == 0:
                print('\033[32m[ %d, <%d/%d>, %.2f min ]\033[0m: return: %d'
                  % (seed, episode+1, total_episodes, episode_time, np.mean(return_list[-10:])))
        s_episode = 0
    env.close()
    agent.actor.load_state_dict(actor_best_weight)
    agent.critic_1.load_state_dict(critic_1_best_weight)
    agent.critic_2.load_state_dict(critic_2_best_weight)
    total_time = (time.time() - start_time) // 60
    print("\033[32m[ 总耗时 ]\033[0m %d分钟" % total_time)
    return return_list, total_time

def save_SAC_data(writer, replay_buffer, return_list, pool_list, time_list, 
                  seed_list, ckpt_path, epoch, episode, weight, seed):
    actor_best_weight, critic_1_best_weight, critic_2_best_weight = weight
    if writer > 1:  # wandb 存档
        wandb.log({
            "_return_list": return_list[-1],
            "pool_size": pool_list[-1],
            })
    # 训练权重存档
    if writer > 0:
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
            },
            ckpt_path,
        )

        # 绘图数据存档
        save_plot_data(return_list, time_list, seed_list, 
                   ckpt_path, seed, pool_list)


def train_DDPG_agent(
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
    time_list: list,
    seed_list: list,
    seed: int,
    ckpt_path: str,
):
    """
    异策略
    """
    def save_data():
        system_type = sys.platform
        ckpt = f'ckpt/{ckpt_path}'
        csv_path = f'data/plot_data/{ckpt_path}'
        os.makedirs(ckpt) if not os.path.exists(ckpt) else None
        os.makedirs(csv_path) if not os.path.exists(csv_path) else None
        alg_name = ckpt_path.split('/')[1]
        torch.save(
            {
                "epoch": epoch,
                "episode": episode,
                "actor_best_weight": actor_best_weight,
                "critic_best_weight": critic_best_weight,
                "return_list": return_list,
                "seed_list": seed_list,
                "time_list": time_list,
            },
            f'{ckpt}/{seed}_{system_type}.pt',
        )
        return_save = pd.DataFrame({
            'Algorithm': [alg_name] * len(return_list),
            'Seed': [seed] * len(return_list),
            "Return": return_list,
            "Pool size": pool_list,
            })
        return_save.to_csv(f'{csv_path}/{seed}_{system_type}.csv', index=False, encoding='utf-8-sig')
    
    return_list = []
    time_list = []
    seed_list = []
    pool_list = []
    start_time = time.time()
    best_score = -1e10  # 初始分数
    for epoch in range(s_epoch, total_epochs):
        for episode in range(s_episode, total_episodes):
            episode_begin_time = time.time()
            episode_return = 0
            step = 0
            state, done, truncated = env.reset(seed=seed)[0], False, False
            state = state.reshape(-1)
            while not (done | truncated):
                action = agent.take_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                next_state = next_state.reshape(-1)
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
            return_list.append(episode_return)
            time_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))
            seed_list.append(seed)
            pool_list.append(replay_buffer.size())
            if episode_return > best_score:
                actor_best_weight = agent.actor.state_dict()
                critic_best_weight = agent.critic.state_dict()
                best_score = episode_return
            if writer > 0:  # 存档
                save_data()
                
            episode_time = (time.time() - episode_begin_time) // 60
            if episode % 10 == 0:
                print('\033[32m[ %d, <%d/%d>, %.2f min ]\033[0m: return: %d'
                  % (seed, episode+1, total_episodes, episode_time, np.mean(return_list[-10:])))
        s_episode = 0
    env.close()
    agent.actor.load_state_dict(actor_best_weight)
    agent.critic.load_state_dict(critic_best_weight)
    total_time = (time.time() - start_time) // 60
    print("\033[32m[ 总耗时 ]\033[0m %d分钟" % total_time)
    return return_list, total_time


def train_DQN(
        env: object,
        agent: object,
        writer: bool,
        s_epoch: int,
        total_epoch: int,
        s_episode: int,
        total_episodes: int,
        total_control_point: int,
        replay_buffer: object,
        minimal_size: int,
        batch_size: int,
        return_list: list,
        time_list: list,
        seed_list: list,
        seed: int,
        ckpt_path: str,):
    start_time = time.time()
    episode_time = time.time()
    # fake_pool = ReplayBuffer(replay_buffer.capacity)
    best_score = -1e10  # 初始化最佳分数
    return_list = [] if not return_list else return_list
    for epoch in range(s_epoch, total_epoch):
        for episode in range(s_episode, total_episodes):
            episode_return = 0
            state, done, truncated = env.reset(seed=seed)[0], False, False
            state = state.reshape(-1)
            while not done | truncated:
                action = agent.take_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                next_state = next_state.reshape(-1)
                replay_buffer.add(state, action, reward, next_state, done, truncated, )
                state = next_state
                episode_return += reward
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d, b_t = sample_exp(agent, replay_buffer, batch_size)
                    transition_dict = {
                        'states': b_s, 'actions': b_a, 'next_states': b_ns,
                        'rewards': b_r, 'dones': b_d, 'truncated': b_t,
                    }
                    agent.update(transition_dict)
                if episode_return > best_score:
                    best_weight = agent.q_net.state_dict()
                    best_score = episode_return

            return_list.append(episode_return)
            time_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))
            seed_list.append(seed)
            # 调整epsilon
            # ! total_control_point=0.3 表示在整个训练的 0.3 处模型取得完全控制
            agent.epsilon = max(-(total_episodes*total_control_point)**(-2)*episode**2+1, 0.01)
            # 保存检查点
            save_DQN_data(writer, replay_buffer, return_list, time_list, 
                          seed_list, ckpt_path, epoch, episode, agent.epsilon,
                          best_weight, seed)
            if episode % 40 == 0:
                episode_time = (time.time() - episode_time) / 60
                print('\033[32m[ %d, <%d/%d>, %.2f min ]\033[0m: return: %.2f, epsilon: %.2f, pool_size: %d'
                    % (seed, episode+1, total_episodes, episode_time, np.mean(return_list[-40:]), agent.epsilon, replay_buffer.size()))
                episode_time = time.time()
            s_episode = 0
    env.close()
    agent.q_net.load_state_dict(best_weight)  # 应用最佳权重
    total_time = (time.time() - start_time) // 60
    print("\033[32m[ 总耗时 ]\033[0m %d分钟" % total_time)
    return return_list, total_time

def sample_exp(agent, replay_buffer, batch_size):
    if agent.sta:
        vae_sample = replay_buffer.return_all_samples()
        s = torch.tensor(vae_sample[0])
        a = torch.tensor(vae_sample[1])
        ns = torch.tensor(vae_sample[3])
        if agent.sta.quality < 0.7:  # 在线训练
            vae_batch = max(s.shape[0] // 600, 1)
            agent.train_cvae(s, a, ns, False, vae_batch)  # 训练 vae
            quality = agent.sta.generate_test(32, len(a.unique()))  # 当前模型生成图像的分类质量
        if agent.sta.quality > 0.3 and replay_buffer.size() > 2000:
            return counterfactual_exp_expand(replay_buffer, agent.sta, batch_size, len(a.unique()), agent.distance_threshold)
        else:
            return replay_buffer.sample(batch_size)
    return replay_buffer.sample(batch_size)


def save_DQN_data(writer, replay_buffer, return_list, time_list, 
                  seed_list, ckpt_path, epoch, episode, epsilon,
                  best_weight, seed):
    # wandb 存档
    if writer > 1:
        wandb.log({
            "_return_list": return_list[-1],
            "pool_size": replay_buffer.size(),
            })
    # 训练权重存档
    torch.save({
        'epoch': epoch,
        'episode': episode,
        'best_weight': best_weight,
        'epsilon': epsilon,
        "return_list": return_list,
        "time_list": time_list,
        "seed_list": seed_list,
        "replay_buffer": replay_buffer,
    }, ckpt_path)

    # 绘图数据存档
    save_plot_data(return_list, time_list, seed_list, ckpt_path, seed)


class ReplayBuffer:
    """异策略的经验缓存"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, truncated):
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
    
    def __len__(self):
        return len(self.buffer)
    
    def return_all_samples(self):
        all_transitions = list(self.buffer)
        state, action, reward, next_state, done, truncated = zip(*all_transitions)
        return np.array(state), action, reward, np.array(next_state), done, truncated
    
    
def counterfactual_exp_expand(replay_buffer, sta, batch_size, action_space_size, distance_threshold):
    '''
    replay_buffer: 经验池
    sta: cvae
    batch_size: 抽多少经验
    action_space_size: 动作空间大小
    distance_threshold: 经验差距阈值，差距太大的匹配经验被放弃
    '''
    # 抽样batch_size组真实经验
    b_s, b_a, b_r, b_ns, b_d, b_t = [torch.tensor(i) for i in replay_buffer.sample(batch_size)]
    
    # 总动作空间大小
    action_space_size = 5

    # 生成反事实动作和其独热向量表示
    counterfactual_actions = []
    for a in b_a:
        counterfactual_actions.append([i for i in range(action_space_size) if i != a])
    counterfactual_actions = torch.tensor(counterfactual_actions).flatten()

    one_hot_cf_actions = torch.nn.functional.one_hot(counterfactual_actions, num_classes=action_space_size)

    # 生成反事实状态转移向量
    diff_state = sta.inference(one_hot_cf_actions)

    # 扩展状态以匹配反事实状态转移
    expand_b_s = b_s.repeat_interleave(action_space_size - 1, dim=0)
    b_ns_prime = expand_b_s + diff_state

    # 读取所有真实经验
    all_s, all_a, all_r, all_ns, all_d, all_t = [torch.tensor(i) for i in replay_buffer.return_all_samples()]

    # 将真实经验和虚拟经验拼接成向量
    # real_exp = torch.cat((all_s, torch.nn.functional.one_hot(all_a, num_classes=action_space_size), all_ns), dim=1)
    # fake_exp = torch.cat((expand_b_s, one_hot_actions, b_ns_prime), dim=1)
    
    # 计算虚拟经验与真实经验的距离并找到最匹配的真实经验
    # distances = torch.cdist(fake_exp, real_exp)
    distances = torch.cdist(b_ns_prime, all_ns)
    min_indices = torch.argmin(distances, dim=1)
    min_distances = distances[torch.arange(distances.size(0)), min_indices]

    # 筛选出距离小于阈值的虚拟经验
    close_matches = min_distances < distance_threshold
    valid_min_indices = min_indices[close_matches]
    
    valid_fake_s = expand_b_s[close_matches]
    valid_fake_r = all_r[valid_min_indices]
    valid_fake_a = one_hot_cf_actions[close_matches].argmax(dim=1)
    valid_fake_ns = b_ns_prime[close_matches]
    
    # 虚拟经验的其他标记
    b_d_prime = torch.zeros_like(valid_fake_r, dtype=torch.bool)
    b_t_prime = torch.zeros_like(valid_fake_r, dtype=torch.bool)

    # 组合虚拟经验与真实经验
    augmented_s = torch.cat((b_s, valid_fake_s), dim=0)
    augmented_a = torch.cat((b_a, valid_fake_a), dim=0)
    augmented_r = torch.cat((b_r, valid_fake_r), dim=0)
    augmented_ns = torch.cat((b_ns, valid_fake_ns), dim=0)
    augmented_d = torch.cat((b_d, b_d_prime), dim=0)
    augmented_t = torch.cat((b_t, b_t_prime), dim=0)

    return augmented_s, augmented_a, augmented_r, augmented_ns, augmented_d, augmented_t


class RDQNReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.trunc_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
        truncated: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.trunc_buf[self.ptr] = truncated
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    truncated=self.trunc_buf[idxs])

    def __len__(self) -> int:
        return self.size

class PrioritizedReplayBuffer(RDQNReplayBuffer):
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
        alpha: float = 0.6
    ):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size)
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
        truncated: bool,
    ):
        """Store experience and priority."""
        super().store(obs, act, rew, next_obs, done, truncated)
        
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        truncated = self.truncated_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
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