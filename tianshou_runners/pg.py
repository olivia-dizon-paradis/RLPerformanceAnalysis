""" Wrapper to use REINFORCE Policy in bsuite experiment.
    Code adapted from: https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/pg.py
    IImplementation of REINFORCE algorithm. https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf
"""

from bsuite.baselines import base
import tianshou as ts
import torch
import dm_env
from gym import spaces
from tianshou.data import Batch, ReplayBuffer
import warnings
warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np

class default_agent(base.Agent):
  def __init__(self, obs_spec, act_spec, buffer_size=128):
    self.obs_shape = (1, obs_spec.shape[0]*obs_spec.shape[1])
    self.act_n = act_spec.num_values
    self.net = ts.utils.net.common.Net(self.obs_shape, self.act_n, hidden_sizes=[64,64], device=device)
    self.optim = torch.optim.Adam(self.net.parameters(), lr=0.0003)
    self.dist = lambda logits: torch.distributions.Categorical(probs=None, logits=logits, validate_args=False)
    self.agent = ts.policy.PGPolicy(self.net, self.optim, self.dist, action_space=spaces.Discrete(self.act_n)).to(device)
    self.buffer_size = buffer_size
    self.buffer = ReplayBuffer(self.buffer_size)

  def select_action(self, obs:dm_env.TimeStep) -> base.Action:
    act = self.agent(Batch(obs=obs.observation.reshape(self.obs_shape), info=str(obs))).act[0]
    return np.int64(act.cpu())

  def update(self, obs:dm_env.TimeStep, action:base.Action, new_obs:dm_env.TimeStep,) -> None:
    self.buffer.add(Batch(obs=obs.observation.reshape(self.obs_shape),
                          act=action,
                          rew=new_obs.reward,
                          truncated=new_obs.last(),
                          info=str(new_obs),
                          terminated=new_obs.last(),
                          obs_next=new_obs.observation.reshape(self.obs_shape)))
    self.agent.update(0, self.buffer, batch_size=self.buffer_size, repeat=1)