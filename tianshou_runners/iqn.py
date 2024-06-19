""" Wrapper to use Implicit Quantile Network (IQN) Policy in bsuite experiment.
    Code adapted from: https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/iqn.py
    Implementation of Implicit Quantile Network. arXiv:1806.06923.
"""

from bsuite.baselines import base
import tianshou as ts
import torch
import dm_env
from tianshou.data import Batch, ReplayBuffer
import warnings
warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class default_agent(base.Agent):
  def __init__(self, obs_spec, act_spec, buffer_size=128):
    self.obs_shape = (1, obs_spec.shape[0]*obs_spec.shape[1])
    self.act_n = act_spec.num_values
    self.net = ts.utils.net.common.Net(self.obs_shape, self.act_n, hidden_sizes=[64, 64], device=device)
    self.iqn = ts.utils.net.discrete.ImplicitQuantileNetwork(self.net, self.act_n, hidden_sizes=[32], device=device).to(device)
    self.optim = torch.optim.Adam(self.iqn.parameters(), lr=0.0003)
    self.agent = ts.policy.IQNPolicy(self.iqn, self.optim).to(device)
    self.buffer_size = buffer_size
    self.buffer = ReplayBuffer(self.buffer_size)

  def select_action(self, obs:dm_env.TimeStep) -> base.Action:
    act = self.agent(Batch(obs=obs.observation.reshape(self.obs_shape), info=str(obs))).act[0]
    return act

  def update(self, obs:dm_env.TimeStep, action:base.Action, new_obs:dm_env.TimeStep,) -> None:
    self.buffer.add(Batch(obs=obs.observation.reshape(self.obs_shape),
                          act=action,
                          rew=new_obs.reward,
                          truncated=new_obs.last(),
                          info=str(new_obs),
                          terminated=new_obs.last(),
                          obs_next=new_obs.observation.reshape(self.obs_shape)))
    self.agent.update(0, self.buffer)