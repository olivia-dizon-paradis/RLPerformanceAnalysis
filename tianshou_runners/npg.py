""" Wrapper to use Natural Policy Gradient (NPG) Policy in bsuite experiment.
    Code adapted from: https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/npg.py
    Implementation of Natural Policy Gradient. https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf
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
  def __init__(self, obs_spec, act_spec, buffer_size=5):
    self.obs_shape = (1, obs_spec.shape[0]*obs_spec.shape[1])
    self.act_n = act_spec.num_values
    self.net = ts.utils.net.common.Net(self.obs_shape, self.act_n, hidden_sizes=[64, 64], softmax=True, device=device)
    self.actor = ts.utils.net.discrete.Actor(self.net, self.act_n, hidden_sizes=[32], device=device).to(device)
    self.critic = ts.utils.net.discrete.Critic(self.net, hidden_sizes=[32], device=device).to(device)
    self.actor_critic = ts.utils.net.common.ActorCritic(self.actor, self.critic)
    self.optim = torch.optim.Adam(self.actor_critic.parameters(), lr=0.0003)
    self.dist = lambda logits: torch.distributions.Categorical(probs=None, logits=logits, validate_args=False)
    self.agent = ts.policy.NPGPolicy(self.actor,
                                     self.critic,
                                     self.optim,
                                     self.dist,
                                     advantage_normalization=False,
                                     action_space=spaces.Discrete(self.act_n)).to(device)
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