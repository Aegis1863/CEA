# Counterfactual experience augmented off-policy reinforcement learning

Code of ***Counterfactual Experience Augmented Off-policy Reinforcement Learning.***

The code files have not been fully organized and are only for temporary reference. A clearer structure and instructions will be updated later.

**Counterfactual experience augmentation** method refers to `utils/CEA.py`.

The **maximum entropy sampling** method can be referenced in a separate repository: https://github.com/Aegis1863/HdGkde

# Requirements

python 3.8, torch, numpy, pandas, seaborn, tqdm, gymnasium, scikit-learn

# To run our method

Continuous control:

`python .\DDPG.py -w 1 --sta --per -t pendulum`

`python .\DDPG.py -w 1 --sta --per -t lunar`

Discrete Control:

`python .\RDQN.py -w 1 --sta --sta_kind regular -t sumo`

`python .\RDQN.py -w 1 --sta --sta_kind regular -t highway`

* terminal parameters:
  * -w: 1 for save data, 0 for test and do not save data;
  * -t task: pendulum, lunar; sumo highway;

Then data will be in `data\plot_data\{task}\{model_name}\{...}.csv`.

# Parameter setting

## Rainbow DQN algorithm parameter settings

| **Parameter** | **Value** | **Description**                           |
| ------------------- | --------------- | ----------------------------------------------- |
| gamma               | 0.99            | Discount factor for future                      |
| alpha               | 0.2             | Determines how much prioritization is used      |
| beta                | 0.6             | Determines how much importance sampling is used |
| prior_eps           | 1e-6            | Guarantees every transition can be sampled      |
| v_min               | 0               | Min value of support                            |
| v_max               | 200             | Max value of support                            |
| atom_size           | 51              | The unit number of support                      |
| memory_size         | 20000           | Size of the replay buffer                       |
| batch_size          | 128             | Batch size for updates                          |
| target_update       | 100             | Period for target model's hard update           |

## SAC-discrete algorithm parameter settings

| **Parameter** | **Value** | **Description**                             |
| ------------------- | --------------- | ------------------------------------------------- |
| actor_lr            | 5e-4            | Learning rate for the actor network               |
| critic_lr           | 5e-3            | Learning rate for the critic network              |
| alpha_lr            | 1e-3            | Learning rate for the temperature parameter       |
| hidden_dim          | 128             | Dimension of hidden layers                        |
| gamma               | 0.98            | Discount factor for future rewards                |
| tau                 | 0.005           | Soft update parameter                             |
| buffer_size         | 20000           | Size of the replay buffer                         |
| target_entropy      | 1.36            | Target entropy for the policy                     |
| model_alpha         | 0.01            | Weighting factor in the model loss function       |
| total_epochs        | 1               | Total number of training epochs                   |
| minimal_size        | 500             | Minimum size of the replay buffer before updating |
| batch_size          | 64              | Batch size for updates                            |

## CEA algorithm parameter settings

| **Parameter** | **Value** | **Description**                 |
| ------------------- | --------------- | ------------------------------------- |
| memory_size         | 20000           | Size of the replay buffer             |
| batch_size          | 128             | Batch size for updates                |
| target_update       | 100             | Period for target model's hard update |
| threshold_ratio     | 0.1             | Threshold ratio for choosing CTP      |

## PPO algorithm parameter settings

| **Parameter** | **Value** | **Description**                                               |
| ------------------- | --------------- | ------------------------------------------------------------------- |
| actor_lr            | 3e-4            | Learning rate for the actor network                                 |
| critic_lr           | 3e-4            | Learning rate for the critic network                                |
| gamma               | 0.99            | Discount factor for future rewards                                  |
| total_epochs        | 1               | Number of training iterations                                       |
| total_episodes      | 100             | Number of simulation played per-training iteration                  |
| eps                 | 0.2             | Clipping range parameter for the PPO objective (1 - eps to 1 + eps) |
| epochs              | 10              | Number of epochs per training sequence in PPO                       |

## MBPO (SAC-discrete) algorithm parameter settings

| **Parameter** | **Value** | **Description**                       |
| ------------------- | --------------- | ------------------------------------------- |
| real_ratio          | 0.5             | Ratio of real and model-generated data      |
| actor_lr            | 5e-4            | Learning rate for the actor network         |
| critic_lr           | 5e-3            | Learning rate for the critic network        |
| alpha_lr            | 1e-3            | Learning rate for the temperature parameter |
| hidden_dim          | 128             | Dimension of hidden layers                  |
| gamma               | 0.98            | Discount factor for future rewards          |
| tau                 | 0.005           | Soft update parameter                       |
| buffer_size         | 20000           | Size of the replay buffer                   |
| target_entropy      | 1.36            | Target entropy for the policy               |
| model_alpha         | 0.01            | Weighting factor in the model loss function |
| rollout_batch_size  | 1000            | Batch size for rollouts                     |
| rollout_length      | 1               | Length of the model rollouts                |
