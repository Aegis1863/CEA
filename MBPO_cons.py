import os
import sys
import time
import gymnasium as gym
import itertools
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import pandas as pd
import collections
import random
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='MBPO 任务')
parser.add_argument('-m', '--model_name', default="MBPO", type=str, help='基本算法名称')
parser.add_argument('-t', '--task', default="lunar", type=str, help='任务名称')
parser.add_argument('-w', '--writer', default=0, type=int, help='存档等级, 0: 不存，1: 本地')
parser.add_argument('-e', '--episodes', default=200, type=int, help='运行回合数')
parser.add_argument('-s', '--seed', nargs='+', default=[1, 7], type=int, help='起始种子')

args = parser.parse_args()

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
        # weights = torch.FloatTensor(transition_dict["weights"].reshape(-1, 1)).to(self.device)  # 无PER时是1
        
        td_target = self.calc_target_continuous(rewards, next_states, dones, truncated)
        critic_1_losses = F.mse_loss(self.critic_1(states, actions), td_target.detach(), reduction='none')
        critic_1_loss = torch.mean(critic_1_losses)
        critic_2_losses = F.mse_loss(self.critic_2(states, actions), td_target.detach(), reduction='none')
        critic_2_loss = torch.mean(critic_2_losses)
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
        
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        
        # PER 更新用
        combined_loss = torch.min(critic_1_losses, critic_2_losses).detach().cpu()
        return combined_loss
            
        

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
        inputs = np.concatenate((obs, np.array(act)), axis=-1)
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
                action = np.array(action).reshape(-1)
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
        inputs = np.concatenate((obs, np.array(action)), axis=-1)
        reward = np.array(reward)
        # reward:[200] -> [200, 1], (next_obs - obs):[200, 3], labels -> [200, 4]
        labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), next_obs - obs),axis=-1,)
        self.fake_env.model.train(inputs, labels)

    def explore(self):
        obs, done, truncated, episode_return = self.env.reset()[0], False, False, 0
        obs = obs.reshape(-1)
        while not done | truncated:
            action = self.agent.take_action(obs)
            action = np.array(action).reshape(-1)
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
                    action = np.array(action).reshape(-1)
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 连续动作空间环境基准测试
    if args.task == 'pendulum':
        env = gym.make('Pendulum-v1')
    elif args.task == 'lunar':
        env = gym.make("LunarLander-v2", continuous=True)
    elif args.task == 'walker':
        env = gym.make("BipedalWalker-v3")
    
    real_ratio = 0.5

    actor_lr = 2e-4
    critic_lr = 2e-4
    alpha_lr = 1e-3
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 20000
    # target_entropy = 0.98 * (-np.log(1 / env.action_space.n))
    model_alpha = 0.01  # 模型损失函数中的加权权重
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    low = torch.tensor(env.action_space.low, device=device)
    high = torch.tensor(env.action_space.high, device=device)
    action_type = 'discrete' if isinstance(env.action_space, gym.spaces.Discrete) else 'continuous'

    rollout_batch_size = 1000
    rollout_length = 1  # 推演长度k, 推荐更多尝试
    model_pool_size = rollout_batch_size * rollout_length
    
    print(f'[ 开始训练, 任务: {args.task}, 模型: {args.model_name}, 设备: {device} ]')
    for seed in range(args.seed[0], args.seed[-1] + 1):
        seed_torch(seed)
        agent = SAC(state_dim, hidden_dim, action_dim, actor_lr,
                    critic_lr, alpha_lr, tau, gamma, action_type, device)
        model = EnsembleDynamicsModel(state_dim, action_dim, model_alpha)
        fake_env = FakeEnv(model)
        env_pool = ReplayBuffer(buffer_size)
        model_pool = ReplayBuffer(model_pool_size)
        mbpo = MBPO(env, agent, fake_env, env_pool, model_pool, rollout_length,
                    rollout_batch_size, real_ratio, args.episodes)
        ckpt_path = f'{args.task}/{args.model_name}'
        return_list = mbpo.train(seed, args.writer, ckpt_path)