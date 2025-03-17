# Counterfactual experience augmented off-policy reinforcement learning

Code of ***[Counterfactual Experience Augmented Off-policy Reinforcement Learning](https://doi.org/10.1016/j.neucom.2025.130017).*** The paper has been accepted by Neurocomputing.

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

# Cite

```
@article{LEE2025130017,
    title = {Counterfactual experience augmented off-policy reinforcement learning},
    journal = {Neurocomputing},
    pages = {130017},
    year = {2025},
    issn = {0925-2312},
    doi = {https://doi.org/10.1016/j.neucom.2025.130017},
    url = {https://www.sciencedirect.com/science/article/pii/S0925231225006897},
    author = {Sunbowen Lee and Yicheng Gong and Chao Deng},
    keywords = {Reinforcement learning, Variational autoencoder, Counterfactual inference, Bisimulation},
}
```
