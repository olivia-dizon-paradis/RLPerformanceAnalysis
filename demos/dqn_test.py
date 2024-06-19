""" Demo to load and run trained DQN RL agents on bsuite environments
    Note: please run "dqn_train.py" before attempting to run this code
    Code adapted from: https://github.com/deepmind/bsuite/blob/main/bsuite/baselines/tf/actor_critic/run.py
"""

from bsuite.baselines import experiment
from bsuite.bsuite import sweep
from bsuite import load_and_record
from bsuite.baselines.tf import dqn
import os
cwd = os.getcwd()
import copy
import pickle
import tensorflow as tf

# Editable Parameters
model_load_path = cwd+"/dqn/models/"
save_path = cwd+'/dqn_expert' # where to save .csv results
num_episodes = 100 # number of episodes to train, set to 0 to accept defaults, use GPU or array if using high value
debug = True # use only a few environments

# Make folder paths for saving results
if not os.path.exists(save_path): os.mkdir(save_path)

# Tset DQN algorithm on bsuite environments
if debug: bsuite_ids = ['bandit/0', 'cartpole/0']
else: bsuite_ids = sweep.SWEEP

for bsuite_id in bsuite_ids:
  # Load environment by id name, a list of id names can be found by printing sweep.SWEEP
  env = load_and_record(bsuite_id=bsuite_id,save_path=save_path,logging_mode='csv',overwrite=True)
  # Load trained DQN agent's network from "dqn_demo.py"
  fpath = model_load_path+bsuite_id.replace('/', '_')+".pkl"
  with open(fpath, 'rb') as handle: network = pickle.load(handle)
  agent = dqn.default_agent(obs_spec=env.observation_spec(), action_spec=env.action_spec())
  agent._online_network = network
  agent._target_network = copy.deepcopy(network)
  agent._forward = tf.function(network)

  # Run on each environment for specified number of episodes
  if not num_episodes: num_episodes = getattr(env, 'bsuite_num_episodes')
  experiment.run(agent=agent,environment=env,num_episodes=num_episodes,verbose=False)