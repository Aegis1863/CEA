import torch
import numpy as np
from torch.distributions import MultivariateNormal

def counterfactual_exp_expand(replay_buffer, sta, batch_size, env, num_new_samples=3, distance_ratio=0.5):
    '''
    适用于连续动作空间
    ---
    replay_buffer: 经验池 \\
    sta: cvae \\
    batch_size: 抽多少经验 \\
    action_space_size: 动作空间大小 \\
    distance_ratio: 控制虚拟经验数量，比如只取前面0.5 \\
    '''
    # 抽样 batch_size 组真实经验
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    samples = replay_buffer.sample_new_real_exp(batch_size)
    b_s, b_ns, b_a, b_r, b_d = samples['states'], samples['next_states'], samples['actions'], samples['rewards'], samples['dones']
    b_s, b_ns, b_a, b_r, b_d = [torch.tensor(i).to(device) for i in [b_s, b_ns, b_a, b_r, b_d]]
    
    low = torch.tensor(env.action_space.low)
    high = torch.tensor(env.action_space.high)
    action_interval = torch.stack((low, high)).T
    b_a = b_a.unsqueeze(1)
    counterfactual_actions = optimize_new_samples(init_action=b_a, num_new_samples=num_new_samples, interval=action_interval, the_device=device)
    _, n, _ = counterfactual_actions.shape  # n 是增加的倍数
    cf_actions = counterfactual_actions.view(-1, 2)
    
    diff_state = sta.inference(cf_actions, device)
    # 扩展状态以匹配反事实状态转移
    expand_b_s = b_s.repeat_interleave(n, dim=0)
    b_ns_prime = expand_b_s + diff_state
    # 读取所有真实经验
    all_samples = replay_buffer.retrieve_real_experiences()
    all_ns, all_r, all_d = torch.tensor(all_samples['next_states']), torch.tensor(all_samples['rewards']), torch.tensor(all_samples['dones'])
    # 计算虚拟经验与真实经验的距离并找到最匹配的真实经验
    # distances = torch.cdist(fake_exp, real_exp)
    distances = torch.cdist(b_ns_prime, all_ns.to(b_ns_prime.device))
    min_indices = torch.argmin(distances, dim=1)
    min_distances = distances[torch.arange(distances.size(0)), min_indices]
    k = int(len(min_distances) * distance_ratio)

    # 筛选出距离小于阈值的虚拟经验
    _, sorted_indices = torch.sort(min_distances)
    close_matches = sorted_indices[:k].cpu()
    valid_min_indices = min_indices[close_matches].cpu()
    valid_fake_s = expand_b_s[close_matches].cpu().numpy()
    valid_fake_r = all_r[valid_min_indices].numpy()
    if cf_actions.device != close_matches.device:
        cf_actions = cf_actions.to(close_matches.device)
    valid_fake_a = cf_actions[close_matches].argmax(dim=1).numpy()
    valid_fake_ns = b_ns_prime[close_matches].cpu().numpy()
    # 虚拟经验的其他标记
    valid_fake_d = all_d[valid_min_indices].numpy()
    # 更新经验池
    for s, a, r, ns, d in zip(valid_fake_s, valid_fake_a, valid_fake_r, valid_fake_ns, valid_fake_d):
        replay_buffer.store(s, a, r, ns, d, d, 1, 0)  # 最后两个数字含义：虚拟经验，未被反事实抽样
    return replay_buffer

def gaussian_kde(action, action_scope, bandwidth=0.1, grid_size=100, device='cpu'):
    ''' 
    action: tensor, shape=(batch, m, n) 
    action_scope: tensor, shape=(n, 2) 
    bandwidth: int | tensor, shape=(n,)
    '''
    b, m, n = action.shape  # b是批量, m是每个样本有多少小样本, n是小样本的维度
    entropy_values = torch.zeros(b, device=device)
    kde_values = []
    
    grid_n = torch.meshgrid(*[torch.linspace(*i_scope, grid_size, device=device) for i_scope in action_scope])
    grid = torch.stack(grid_n, dim=-1).reshape(-1, n).to(device)
    
    for i in range(b):
        sample_points = action[i]
        kde = torch.zeros(grid.shape[0], device=device)
        for j in range(m):
            mean = sample_points[j]
            cov_matrix = (torch.eye(n, device=device) * bandwidth)
            mvn = MultivariateNormal(mean, cov_matrix)
            kde += mvn.log_prob(grid).exp()
        kde /= m  # 取样本点数的平均
        kde = kde.view(*[grid_size for _ in range(n)])
        kde_values.append(kde)
        
        delta_v = (2 / grid_size) ** n  # 网格单元体积
        kde_sum = torch.sum(kde) * delta_v
        kde_norm = kde / kde_sum
        entropy = -torch.sum(kde_norm * torch.log(kde_norm + 1e-12)) * delta_v
        entropy_values[i] = entropy
    
    return kde_values, entropy_values, grid_n

def optimize_new_samples(init_action, bandwidth=0.2, num_new_samples=3, 
                         interval=torch.tensor([[-1, 1], ]), learning_rate=8e-2, 
                         num_iterations=20, the_device=None):
    assert len(init_action.shape) == 3, 'action should have 3 dimensions: (batch, m, n)'
    device = the_device if the_device else ('cuda' if torch.cuda.is_available() else 'cpu')
    init_action = init_action.to(device)  # (batch, dim) -> (batch, num=1, dim)
    
    if interval.shape[0] == 1:
        new_samples_init = torch.linspace(*interval[0], num_new_samples, device=device).repeat(init_action.shape[0], 1)
        new_samples = new_samples_init.view(new_samples_init.size(0), -1).unsqueeze(-1)
    else:
        interval = interval.to(device)
        linspaces = [torch.linspace(*i_space, num_new_samples, device=device) for i_space in interval]
        grids = torch.meshgrid(*linspaces)
        new_samples = torch.stack(grids, dim=-1).view(-1, len(interval)).unsqueeze(0).repeat(init_action.shape[0], 1, 1)
    
    new_samples = new_samples.requires_grad_()
    optimizer = torch.optim.Adam([new_samples], lr=learning_rate)
    
    for _ in range(num_iterations):
        optimizer.zero_grad()
        if init_action.shape[-1] > 1:
            action_with_new = torch.cat([init_action, new_samples], dim=1)
        else:
            action_with_new = torch.cat([init_action, new_samples], dim=1)
        _, entropy_with_new, _ = gaussian_kde(action_with_new, interval, bandwidth, device=device)
        loss = -entropy_with_new.mean()
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        min_vals = interval[:, 0].unsqueeze(1).to(device)
        max_vals = interval[:, 1].unsqueeze(1).to(device)
        if init_action.shape[-1] == 1:
            # new_samples = new_samples.transpose(0, 2).squeeze(-1)
            new_samples = torch.clamp(new_samples, min=min_vals, max=max_vals).T.transpose(0, 2)
        else:
            for samples_indim, i_interval in zip(range(new_samples.shape[-1]), interval):
                new_samples[:,:,samples_indim] = torch.clamp(new_samples[:,:,samples_indim], *i_interval)
            
    return new_samples.detach()