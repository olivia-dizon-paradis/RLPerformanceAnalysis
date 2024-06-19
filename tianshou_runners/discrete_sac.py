""" Wrapper to use Discrete Soft Actor Critic (SAC) Policy in bsuite experiment.
    Code adapted from: https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/discrete_sac.py
    Implementation of SAC for Discrete Action Settings. arXiv:1910.07207.
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
    self.net = ts.utils.net.common.Net(self.obs_shape, self.act_n, hidden_sizes=[64, 64], device=device)
    self.actor = ts.utils.net.discrete.Actor(self.net, self.act_n, hidden_sizes=[32], softmax_output=False, device=device).to(device)
    self.critic1 = ts.utils.net.discrete.Critic(self.net, hidden_sizes=[32], last_size=self.act_n, device=device).to(device)
    self.critic2 = ts.utils.net.discrete.Critic(self.net, hidden_sizes=[32], last_size=self.act_n, device=device).to(device)
    self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.0003)
    self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=0.0003)
    self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=0.0003)
    self.agent = ts.policy.DiscreteSACPolicy(self.actor,
                                             self.actor_optim,
                                             self.critic1,
                                             self.critic1_optim,
                                             self.critic2,
                                             self.critic2_optim,
                                             action_space=spaces.Discrete(self.act_n)).to(device).to(device)
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
    self.agent.update(0, self.buffer)