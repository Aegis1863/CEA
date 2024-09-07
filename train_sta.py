from utils.STA import *
import os
import gymnasium as gym
# from SAC_cons import *
from DDPG import *
import argparse

parser = argparse.ArgumentParser(description='Train_STA task')
parser.add_argument('-t', '--task', default="lunar", type=str, help='task name')
parser.add_argument('-l', '--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('-e', '--epochs', default=150, type=int, help='training epochs')
parser.add_argument('-b', '--batch_size', default=128, type=int, help='batch size')
args = parser.parse_args()


if args.task == 'pendulum':
    env = gym.make('Pendulum-v1')
elif args.task == 'cartpole':
    env = gym.make('CartPole-v1')
elif args.task == 'lunar':
    env = gym.make("LunarLander-v2", continuous=True)
elif args.task == 'walker':
    env = gym.make("BipedalWalker-v3")

action_type = 'discrete' if isinstance(env.action_space, gym.spaces.discrete.Discrete) else "continuous"
if action_type == 'continuous':
    if env.action_space.shape[0] > 1:
        action_scope = torch.tensor((env.action_space.low, env.action_space.high)).T
    else:
        action_scope = (float(env.action_space.low), float(env.action_space.high))
else:
    action_scope = None
    
# ==========
# init
diff_state = []
actions = []

# sequential read and concatenate
for i in os.listdir(f'ckpt/{args.task}/DDPG/'):
    file_name = f'ckpt/{args.task}/DDPG/{i}'
    buffer = torch.load(file_name)['replay_buffer']
    diff_s = buffer.next_states_buf - buffer.states_buf
    diff_state.append(diff_s)
    actions.append(buffer.actions_buf)

diff_states = np.concatenate(diff_state, axis=0)
actions = np.concatenate(actions, axis=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = diff_states.shape[-1]
condition_dim = actions.shape[-1]
latent_dim = input_dim

fig_path = f'image/VAE/{args.task}/bs_{args.batch_size}/'

# training
model = CVAE(input_dim, condition_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
quality = []
loss_list = []
print(f'[ Start STA training, Task: {args.task}, Epochs: {args.epochs}, Batch_size: {args.batch_size} ]')
for epoch in trange(args.epochs, ncols=70):
    loss = cvae_train(model, device, diff_states, actions, optimizer, False, 
                      args.batch_size, action_type, action_scope)
    loss_list.append(loss)
    quality.append(model.generate_test(32, action_scope, fig_path, action_type))
    
print(f'\n==> Generate silhouette score: {[round(i, 3) for i in quality]}')
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.lineplot(quality)
plt.xlabel('Epoch')
plt.ylabel('Silhouette score')
plt.grid()
plt.subplot(1, 2, 2)
sns.lineplot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.tight_layout()
plt.savefig(f'{fig_path}/Silhouette score.png')
plt.close()
os.mkdir(f'model/sta/{args.task}') if not os.path.exists(f'model/sta/{args.task}') else None
torch.save(model, f'model/sta/{args.task}/regular.pt')
print(model.quality)