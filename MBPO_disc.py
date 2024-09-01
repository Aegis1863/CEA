import os
os.environ['LIBSUMO_AS_TRACI'] = '1'  # 终端运行加速
import sys
import time
import gymnasium as gym
from collections import namedtuple
import itertools
from itertools import count
import sumo_rl
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import pandas as pd
import collections
import random
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='MBPO 任务')
parser.add_argument('--model_name', default="MBPO", type=str, help='基本算法名称')
parser.add_argument('-t', '--task', default="highway", type=str, help='任务名称')
parser.add_argument('-n', '--net', default="env/big-intersection/big-intersection.net.xml", type=str, help='SUMO路网文件路径')
parser.add_argument('-f', '--flow', default="env/big-intersection/big-intersection.rou.xml", type=str, help='SUMO车流文件路径')
parser.add_argument('-w', '--writer', default=0, type=int, help='存档等级, 0: 不存，1: 本地 2: 本地 + wandb本地, 3. 本地 + wandb云存档')
parser.add_argument('-o', '--online', action="store_true", help='是否上传wandb云')
parser.add_argument('--sta', action="store_true", help='是否利用sta辅助')
parser.add_argument('--sta_kind', default=False, help='sta 预训练模型类型，"expert"或"regular"')
parser.add_argument('-e', '--episodes', default=500, type=int, help='运行回合数')
parser.add_argument('-r', '--reward', default='diff-waiting-time', type=str, help='奖励函数')
parser.add_argument('--begin_time', default=1000, type=int, help='回合开始时间')
parser.add_argument('--duration', default=2000, type=int, help='单回合运行时间')
parser.add_argument('--begin_seed', default=1, type=int, help='起始种子')
parser.add_argument('--end_seed', default=7, type=int, help='结束种子')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.h_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.h_1(F.relu(self.fc1(x))))
        return F.softmax(self.fc2(x), dim=1)  # 直接输出softmax


class QValueNet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.h_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.h_1(F.relu(self.fc1(x))))
        return self.fc2(x)
    
class SAC:
    ''' 处理离散动作的SAC算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 第一个Q网络
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 第二个Q网络
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)  # 动作不再是float类型
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.int).view(-1, 1).to(self.device)
        truncated = torch.tensor(transition_dict['truncated'], dtype=torch.int).view(-1, 1).to(self.device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones | truncated)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新alpha值
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        

class Swish(nn.Module):
    ''' Swish激活函数 '''
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def init_weights(m):
    ''' 初始化模型权重 '''
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = (t < mean - 2 * std) | (t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(
                cond,
                torch.nn.init.normal_(torch.ones(t.shape, device=device),
                                      mean=mean,
                                      std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, FCLayer):
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(m._input_dim)))
        m.bias.data.fill_(0.0)


class FCLayer(nn.Module):
    ''' 集成之后的全连接层 '''
    def __init__(self, input_dim, output_dim, ensemble_size, activation):
        super(FCLayer, self).__init__()
        self._input_dim, self._output_dim = input_dim, output_dim
        self.weight = nn.Parameter(
            torch.Tensor(ensemble_size, input_dim, output_dim).to(device))
        self._activation = activation
        self.bias = nn.Parameter(
            torch.Tensor(ensemble_size, output_dim).to(device))

    def forward(self, x):
        return self._activation(torch.add(torch.bmm(x, self.weight), self.bias[:, None, :]))
    
    
class EnsembleModel(nn.Module):
    ''' 环境模型集成 '''
    def __init__(self,
                 state_dim,
                 action_dim,
                 model_alpha,
                 ensemble_size=5,
                 learning_rate=1e-3):
        super(EnsembleModel, self).__init__()
        # 输出包括均值和方差, 因此是'状态与奖励维度'之和的两倍
        self._output_dim = (state_dim + 1) * 2
        self._model_alpha = model_alpha  # 模型损失函数中优化可训练方差区间的权重
        self._max_logvar = nn.Parameter((torch.ones((1, self._output_dim // 2)).float() / 2).to(device), requires_grad=False)
        self._min_logvar = nn.Parameter((-torch.ones((1, self._output_dim // 2)).float() * 10).to(device), requires_grad=False)

        self.layer1 = FCLayer(state_dim + action_dim, 200, ensemble_size, Swish())
        self.layer2 = FCLayer(200, 200, ensemble_size, Swish())
        self.layer3 = FCLayer(200, 200, ensemble_size, Swish())
        self.layer4 = FCLayer(200, 200, ensemble_size, Swish())
        self.layer5 = FCLayer(200, self._output_dim, ensemble_size, nn.Identity())  # nn.Identity() 是恒等映射激活, 就是直接输出
        self.apply(init_weights)  # 初始化环境模型中的参数
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, return_log_var=False):
        ret = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        mean = ret[:, :, :self._output_dim // 2]  # 前面一半作为均值, 后面一半作为方差
        # 在PETS算法中, 将方差控制在最小值和最大值之间
        logvar = self._max_logvar - F.softplus(self._max_logvar - ret[:, :, self._output_dim // 2:])
        logvar = self._min_logvar + F.softplus(logvar - self._min_logvar)
        return mean, logvar if return_log_var else torch.exp(logvar)

    def loss(self, mean, logvar, labels, use_var_loss=True):
        inverse_var = torch.exp(-logvar)
        if use_var_loss:
            mse_loss = torch.mean(
                torch.mean(torch.pow(mean - labels, 2) * inverse_var,
                           dim=-1),
                dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)  # 带着方差损失一起优化
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()
        # loss 同时优化方差, 缩小方差区间
        loss += self._model_alpha * torch.sum(self._max_logvar) - self._model_alpha * torch.sum(self._min_logvar)
        loss.backward()
        self.optimizer.step()


class EnsembleDynamicsModel:
    ''' 环境模型集成,加入精细化的训练 '''
    def __init__(self, state_dim, action_dim, model_alpha=0.01, num_network=5):
        '''
        - state_dim : 状态维度
        - action_dim : 动作维度
        - model_alpha : float, 可选, 优化可训练方差区间的权重, 这是为了缩小方差, 默认 0.01
        - num_network : int, 可选, 与环境集成数一致, 默认 5
        '''
        self._num_network = num_network
        self._state_dim, self._action_dim = state_dim, action_dim
        self.model = EnsembleModel(state_dim,
                                   action_dim,
                                   model_alpha,
                                   ensemble_size=num_network)
        self._epoch_since_last_update = 0

    def train(self,
              inputs,
              labels,
              batch_size=64,
              holdout_ratio=0.1,  # 验证比例
              max_iter=20):
        # 设置训练集与验证集
        permutation = np.random.permutation(inputs.shape[0])  # 给出随机打乱的序号
        inputs, labels = inputs[permutation], labels[permutation]
        num_holdout = int(inputs.shape[0] * holdout_ratio)
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]  # 训练集
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]  # 验证集
        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self._num_network, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self._num_network, 1, 1])

        # 保留最好的结果
        # 若干个环境模型的推演快照
        self._snapshots = {i: (None, 1e10) for i in range(self._num_network)}

        for epoch in itertools.count():  # 无终止序列, 用于需要循环的次数无法确定时, 可以用break跳出
            # 定义每一个网络的训练数据
            train_index = np.vstack([
                np.random.permutation(train_inputs.shape[0])
                for _ in range(self._num_network)
            ])
            # 所有真实数据都用来训练
            for batch_start_pos in range(0, train_inputs.shape[0], batch_size):
                batch_index = train_index[:, batch_start_pos:batch_start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[batch_index]).float().to(device)
                train_label = torch.from_numpy(train_labels[batch_index]).float().to(device)

                mean, logvar = self.model(train_input, return_log_var=True)
                loss, _ = self.model.loss(mean, logvar, train_label)  # 这里返回的loss是同时带均值方差的
                self.model.train(loss)

            # 验证模型
            with torch.no_grad():
                mean, logvar = self.model(holdout_inputs, return_log_var=True)
                _, holdout_losses = self.model.loss(mean,
                                                    logvar,
                                                    holdout_labels,
                                                    use_var_loss=False)
                holdout_losses = holdout_losses.cpu()
                break_condition = self._save_best(epoch, holdout_losses)
                # 如果五个动力模型都优化超过10%, 或到迭代限制时则结束训练
                if break_condition or epoch > max_iter:
                    break

    def _save_best(self, epoch, losses, threshold=0.1):
        updated = False
        for i in range(len(losses)):
            current = losses[i]
            _, best = self._snapshots[i]  # best 一开始是个很大的值
            improvement = (best - current) / best
            if improvement > threshold:  # 对于i模型来说, 提升是否大于10% (0.1)
                self._snapshots[i] = (epoch, current)
                updated = True
        self._epoch_since_last_update = 0 if updated else self._epoch_since_last_update + 1
        return self._epoch_since_last_update > 5  # 如果五个动力模型都更新了最佳状态返回True

    def predict(self, inputs, batch_size=64):
        inputs = np.tile(inputs, (self._num_network, 1, 1))
        inputs = torch.tensor(inputs, dtype=torch.float).to(device)
        mean, var = self.model(inputs, return_log_var=False)
        return mean.detach().cpu().numpy(), var.detach().cpu().numpy()


class FakeEnv:
    def __init__(self, model: EnsembleDynamicsModel):
        self.model = model

    def step(self, obs, act):
        inputs = np.concatenate((obs, np.array(act)[np.newaxis]), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        ensemble_model_means[:, :, 1:] += obs  # * 这一步还原了next_obs的预测, 因为之前 label = next_obs - obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)
        # 重参数化
        ensemble_samples = ensemble_model_means + np.random.normal(
            size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        models_to_use = np.random.choice([i for i in range(self.model._num_network)], size=batch_size)  # 五个模型里面抽一个
        batch_inds = np.arange(0, batch_size)  # 抽批量大小1, 也只存了一步, 因为传进来的obs和act就是一步, 理论上可以多步
        samples = ensemble_samples[models_to_use, batch_inds]
        rewards, next_obs = samples[:, :1][0][0], samples[:, 1:][0]
        return rewards, next_obs
    

class MBPO:
    def __init__(self, env, agent, fake_env, env_pool, model_pool,
                 rollout_length, rollout_batch_size, real_ratio, num_episode):

        self.env = env
        self.agent = agent
        self.fake_env = fake_env
        self.env_pool = env_pool  # 真环境模型经验池
        self.model_pool = model_pool  # 假环境模型经验池
        self.rollout_length = rollout_length
        self.rollout_batch_size = rollout_batch_size
        self.real_ratio = real_ratio
        self.num_episode = num_episode

    def rollout_model(self):
        observations, _, _, _, _, _ = self.env_pool.sample(self.rollout_batch_size)
        for obs in observations:
            for i in range(self.rollout_length):
                action = self.agent.take_action(obs)
                reward, next_obs = self.fake_env.step(obs, action)
                self.model_pool.add(obs, action, reward, next_obs, False, False)
                obs = next_obs

    def update_agent(self, policy_train_batch_size=64):
        env_batch_size = int(policy_train_batch_size * self.real_ratio)  # real_ratio = 0.5
        model_batch_size = policy_train_batch_size - env_batch_size
        for _ in range(10):
            env_obs, env_action, env_reward, env_next_obs, env_done, env_truncated = self.env_pool.sample(env_batch_size)
            if self.model_pool.size() > 0:
                model_obs, model_action, model_reward, model_next_obs, model_done, model_truncated = self.model_pool.sample(model_batch_size)
                obs = np.concatenate((env_obs, model_obs), axis=0)
                action = np.concatenate((env_action, model_action), axis=0)
                next_obs = np.concatenate((env_next_obs, model_next_obs), axis=0)
                reward = np.concatenate((env_reward, model_reward), axis=0)
                done = np.concatenate((env_done, model_done), axis=0)
                truncated = np.concatenate((env_truncated, model_truncated), axis=0)
            else:
                obs, action, next_obs, reward, done, truncated = env_obs, env_action, env_next_obs, env_reward, env_done, env_truncated
            transition_dict = {
                'states': obs,
                'actions': action,
                'next_states': next_obs,
                'rewards': reward,
                'dones': done,
                'truncated': truncated,
            }
            self.agent.update(transition_dict)

    def train_model(self):
        '''输入 obs 和 action ，label是cat(reward, next_obs - obs)'''
        
        obs, action, reward, next_obs, done, truncated = self.env_pool.return_all_samples()
        inputs = np.concatenate((obs, np.array(action)[:, np.newaxis]), axis=-1)
        reward = np.array(reward)
        # reward:[200] -> [200, 1], (next_obs - obs):[200, 3], labels -> [200, 4]
        labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), next_obs - obs),axis=-1,)
        self.fake_env.model.train(inputs, labels)

    def explore(self):
        obs, done, truncated, episode_return = self.env.reset()[0], False, False, 0
        obs = obs.reshape(-1)
        while not done | truncated:
            action = self.agent.take_action(obs)
            next_obs, reward, done, truncated, info = self.env.step(action)
            next_obs = next_obs.reshape(-1)
            self.env_pool.add(obs, action, reward, next_obs, done, truncated)
            obs = next_obs
            episode_return += reward
        return episode_return

    def train(self, seed, writer, ckpt_path):
        def save_data():
            system_type = sys.platform
            ckpt = f'ckpt/{ckpt_path}'
            csv_path = f'data/plot_data/{ckpt_path}'
            os.makedirs(ckpt) if not os.path.exists(ckpt) else None
            os.makedirs(csv_path) if not os.path.exists(csv_path) else None
            alg_name = ckpt_path.split('/')[1]
            torch.save(
                {
                    "episode": i_episode,
                    "agent": self.agent,
                    "return_list": return_list,
                    "time_list": time_list,
                    "seed_list": seed_list,
                    "pool_list": pool_list,
                    "replay_buffer": self.model_pool,
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
        time_list = [time.time()]
        seed_list = [seed]
        pool_list = [0]
        explore_return = self.explore()  # 模型未经训练时相当于随机探索，采集数据至动力模型经验池
        print('\n\033[32m[ Explore episode ]\033[0m: 1, return: %d' % explore_return)
        return_list.append(explore_return)
        with tqdm(total=self.num_episode - 1, mininterval=40, ncols=100) as pbar:
            for i_episode in range(self.num_episode - 1):
                obs, done, truncated, episode_return = self.env.reset(seed=seed)[0], False, False, 0
                obs = obs.reshape(-1)
                step = 0
                while not done | truncated:
                    if step % 50 == 0:  # 每50步训练一次动力环境、推演并收集经验
                        self.train_model()  # 训练动力环境
                        self.rollout_model()  # 在动力环境的经验池采样状态并推演，将经验增加到策略经验池
                    
                    action = self.agent.take_action(obs)
                    next_obs, reward, done, truncated, info = self.env.step(action)
                    next_obs = next_obs.reshape(-1)
                    self.env_pool.add(obs, action, reward, next_obs, done, truncated)
                    obs = next_obs
                    episode_return += reward
                    self.update_agent()
                    step += 1
                return_list.append(episode_return)
                time_list.append(time.time())
                seed_list.append(seed)
                pool_list.append(self.env_pool.size())
                if writer > 0:
                    save_data()
                pbar.set_postfix({
                    'return': round(np.mean(return_list[-20:]), 2),
                    'Pool size': pool_list[-1],
                })
                pbar.update(1)
                
                # print('\n\033[32m[ Episode ]\033[0m %d, return: %d' % (i_episode + 2, episode_return))
        env.close()
        return return_list


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, truncated):
        self.buffer.append((state, action, reward, next_state, done, truncated))

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            return self.return_all_samples()
        else:
            transitions = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done, truncated = zip(*transitions)
            return np.array(state), action, reward, np.array(next_state), done, truncated

    def return_all_samples(self):
        all_transitions = list(self.buffer)
        state, action, reward, next_state, done, truncated = zip(*all_transitions)
        return np.array(state), action, reward, np.array(next_state), done, truncated
    
if __name__ == '__main__':
    def seed_torch(seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    
    # 环境相关
    if args.task == 'sumo':
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
    elif args.task == 'highway':
        env = gym.make('highway-fast-v0')
        env.configure({
            "lanes_count": 4,
            "vehicles_density": 2,
            "duration": 100,
        })
    elif args.task == 'cliff':
        gym.make('CliffWalking-v0')
    

    real_ratio = 0.5

    actor_lr = 5e-4
    critic_lr = 5e-3
    alpha_lr = 1e-3
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 20000
    target_entropy = 0.98 * (-np.log(1 / env.action_space.n))
    model_alpha = 0.01  # 模型损失函数中的加权权重
    state_dim = env.observation_space.shape[0] if args.task == 'sumo' else torch.multiply(*env.observation_space.shape)
    num_actions = env.action_space.n
    action_dim = 1

    rollout_batch_size = 1000
    rollout_length = 1  # 推演长度k, 推荐更多尝试
    model_pool_size = rollout_batch_size * rollout_length

    for seed in range(args.begin_seed, args.end_seed + 1):
        seed_torch(seed)
        agent = SAC(state_dim, hidden_dim, num_actions, actor_lr,
                    critic_lr, alpha_lr, target_entropy, tau, gamma, device)
        model = EnsembleDynamicsModel(state_dim, action_dim, model_alpha)
        fake_env = FakeEnv(model)
        env_pool = ReplayBuffer(buffer_size)
        model_pool = ReplayBuffer(model_pool_size)
        mbpo = MBPO(env, agent, fake_env, env_pool, model_pool, rollout_length,
                    rollout_batch_size, real_ratio, args.episodes)
        ckpt_path = f'{args.task}/{args.model_name}'
        return_list = mbpo.train(seed, args.writer, ckpt_path)