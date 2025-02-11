# Resource Usage Evaluation of Discrete Model-Free Deep Reinforcement Learning Algorithms
Code to run various Deep Reinforcement Learning (DRL) algorithms in the [bsuite](https://github.com/google-deepmind/bsuite) suite of environments.
The goal is to evaluate the practicality of discrete, model-free deep RL algorithms by characterizing their performance, runtime, and memory usage in a variety of different types of environments.
Code in this repository was used for the paper "Resource Usage Evaluation of Discrete Model-Free Deep Reinforcement Learning Algorithms", presented at the Reinforcement Learning Conference (RLC) in 2024.
For more information, please refer to our paper, published in [Reinforcement Learning Journal (RLJ), Volumes 1-5, 2024](https://doi.org/10.5281/zenodo.13899776)

![radar plot](_readme_figs/scores.png)
![radar plot](_readme_figs/resources.png)

## Installation
Code was run on Python 3.8.16. To install dependencies in a virtual environment, please download the [requirements.txt](requirements.txt) file and run

```bash
conda create --no-default-packages -n myenv python=3.8.16
conda activate myenv
pip install -r requirements.txt
```
## Technical Overview and Repository Organization
For this study, sixteen DRL algorithms were trained in 23 different base environments (468 seeds), resulting in a total of 7,488 trained agents.
Experiments were conducted on NVIDIA GeForce 2080Ti nodes, each with a cyclic allocation of 16GB CPU and 11GB GPU for processing. 
In total, it took 256 GB and 830 days CPU time (i.e., sixty-nine days on a twelve-node parallel system) to run all experiments and 1.8 GB to store all models.
This repository contains the python scripts from this study. 

This repository is organized into three main folders:
1. [demos/](demos/) contains short snippets of code to ensure the python environment is set up correctly. It also contains the code used to evaluate the RL agents
2. [bsuite_runners/](bsuite_runners/) contains the programs used to train the bsuite baseline RL algorithms for the paper. For more information, please refer to the [bsuite](https://github.com/google-deepmind/bsuite) repository on GitHub
3. [tianshou_runners/](tianshou_runners/) contains the programs used to train the tianshou RL algorithms for the paper. For more information, please refer to the [tianshou](https://github.com/thu-ml/tianshou) repository on GitHub

## Citing
If you use this in your work, please cite the accompanying paper:

```bibtex
@article{dizon-paradis2024resource,
    title={Resource Usage Evaluation of Discrete Model-Free Deep Reinforcement Learning Algorithms},
    author={Dizon-Paradis, Olivia P. and Wormald, Stephen E. and Capecci, Daniel E. and Bhandarkar, Avanti and Woodard, Damon L.},
    journal={Reinforcement Learning Journal},
    volume={5},
    pages={2162--2177},
    year={2024}
}
```
