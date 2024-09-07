import os
from re import S
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim, 128)
        self.fc21 = nn.Linear(128, latent_dim)  # Mean
        self.fc22 = nn.Linear(128, latent_dim)  # Log variance
        self.fc3 = nn.Linear(latent_dim + condition_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim  # action dim
        self.quality = 0  # Silhouette Coefficient
        self.pic_num = 0

    def encode(self, x, c):
        h1 = torch.relu(self.fc1(torch.cat([x, c], dim=1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        h3 = torch.relu(self.fc3(torch.cat([z, c], dim=-1)))
        return self.fc4(h3)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
    
    def inference(self, c, the_device=None):
        '''Generate results directly based on conditions
        ---
        - c: Condition, it should be one-hot in discrete action space and single value in continuous action space
        '''
        device = the_device if the_device else ('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            c = c.to(device)
            sample = torch.randn(c.shape[0], self.latent_dim).to(device)
            generated = self.decode(sample, c)
        return generated
    
    def generate_test(self, batch, action_space, save_path=None, action_type='discrete'):
        '''Generate some conditions for generation and return the silhouette coefficient of the generated data
        ---
        - batch: batch size
        - action_space: Action space, if it is a discrete action, write the number of optional actions, if it is a continuous action, pass in the action limit: [lower bound, upper bound]
        - save_path: Image path, the default is None, which means that the generated image will not be saved
        - action_type: Action space type: discrete or continous
        '''
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if action_type == 'discrete':
            action_space = int(action_space)
            # (batch // action_space) Indicates how many each action generates
            conditions = torch.tensor([[i] * (batch // action_space) for i in range(action_space)]).view(-1)
            conditions = torch.eye(action_space)[conditions].to(device)
        elif isinstance(action_space, tuple):
            conditions = torch.round(torch.linspace(*action_space, batch//4), decimals=3).repeat(4).sort()[0].unsqueeze(-1).to(device)
        else:
            random_samples = [torch.round(torch.linspace(*i_space, batch//6), decimals=3) for i_space in action_space]
            conditions = torch.cartesian_prod(*random_samples).repeat_interleave(3, dim=0).to(device)
        with torch.no_grad():
            sample = torch.randn(conditions.shape[0], self.latent_dim).to(device)
            generated = self.decode(sample, conditions).cpu()
            
        quality = silhouette_score(generated, [f"({x.item()}, {y.item()})" for x, y in conditions])
        self.quality = quality
        if save_path and self.quality < 0.8:
            os.makedirs(save_path) if not os.path.exists(save_path) else None
            plt.figure(figsize=(12, 12))
            ax = sns.heatmap(generated, cbar=False)
            ax.set_yticks(np.arange(len(conditions)) + 0.5)
            label = [f"({x.item()}, {y.item()})" for x, y in conditions]
            ax.set_yticklabels(label, rotation=0)
            plt.xlabel('State components')
            plt.ylabel('Action')
            plt.title(f'Silhouette score: {quality:.3f}')
            plt.savefig(f'{save_path}/{self.pic_num}.png')
            plt.close()
            self.pic_num += 1
        return quality
    

def cvae_train(model, device, diff_state:torch.tensor, action:torch.tensor, 
               optimizer: object, test_and_feedback=False, batch_size=32, 
               action_type='discrete', action_scope=None):
    '''
    model: cvae
    diff_state: diff_state = state[1:, 5:] - state[:-1, 5:]
    action: action, one-hot or a value
    optimizer: torch.optim.Adam
    test_and_feedback: feedback, False
    batch_size: suggest 32
    '''
    if not isinstance(diff_state, torch.Tensor):
        diff_state = torch.tensor(diff_state)
        action = torch.tensor(action)
    
    def cvae_loss():
        MSE = nn.functional.mse_loss(recon_batch, state) 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD
    
    def prepare_data():
        dataset = TensorDataset(diff_state, action)
        train_size = int(0.8 * len(dataset)) if test_and_feedback else len(dataset)
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader,test_loader
    
    train_loader, test_loader = prepare_data()
    model.train()
    train_loss = 0
    # train
    for state, action in train_loader:
        state, action = state.to(device), action.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(state, action)
        if action_type == 'discrete':
            model.quality = model.generate_test(32, model.condition_dim, None,action_type)  # 轮廓系数
        else:
            model.quality = model.generate_test(32, action_scope, None, action_type)
        loss = cvae_loss()
        loss.backward()
        train_loss = loss.item()
        optimizer.step()
    # test
    if test_and_feedback:
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for state, action in test_loader:
                state, action = state.to(device), action.to(device)
                recon_batch, mu, logvar = model(state, action)
                test_loss += cvae_loss().item()

        test_loss /= len(test_loader.dataset)
        print(f'Train/test loss: {train_loss/state.shape[0]:.4f} / {test_loss:.4f}')
        return train_loss, test_loss
    return train_loss

if __name__ == '__main__':
    
    # ------- Notice -------
    #   It is best not to train here. For training, refer to train_sta.py in the main directory.
    # ----------------------
    
    mission = 'sumo'
    state, kind = torch.load(f'data/buffer/{mission}/Buffer_of_expert.pt'), 'expert'
    # state, kind = torch.load(f'data/buffer/{mission}/Buffer_of_regular.pt'), 'regular'

    action = state[1:, :4]
    diff_state = state[1:, 5:] - state[:-1, 5:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = diff_state.shape[-1]
    condition_dim = action.shape[-1]
    latent_dim = input_dim
    batch_size = 16

    fig_path = f'image/VAE/{kind}/{batch_size}/'
    
    # 训练
    model = CVAE(input_dim, condition_dim, latent_dim).to(device)
    num_epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    quality = []
    for epoch in trange(num_epochs, ncols=70):
        cvae_train(model, device, diff_state, action, optimizer, False, batch_size)
        quality.append(model.generate_test(32, 4, epoch, fig_path))
    print(f'\n==> Generate silhouette score: {[round(i, 3) for i in quality]}')
    plt.figure()
    sns.lineplot(quality)
    plt.xlabel('Epoch')
    plt.ylabel('Silhouette score')
    plt.grid()
    plt.savefig(f'{fig_path}/Silhouette score.png')
    plt.close()
    torch.save(model, f'model/sta/{mission}/{kind}.pt')
    print(model.quality)