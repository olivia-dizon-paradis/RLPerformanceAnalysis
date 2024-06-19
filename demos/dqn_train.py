""" Demo to train DQN RL algorithm on bsuite environments
    Code adapted from: https://github.com/deepmind/bsuite/blob/main/bsuite/baselines/experiment.py
"""

from bsuite.baselines import experiment
from bsuite.bsuite import sweep
from bsuite import load_and_record
from bsuite.baselines.tf import dqn
import os
cwd = os.getcwd()
import pickle

# Editable Parameters
save_path = cwd+'/dqn' # where to save .csv results and .pkl models
num_episodes = 100 # number of episodes to train, set to 0 to accept defaults, use GPU or array if using high value
debug = True # use only a few environments

# Make folder paths for saving results and models
model_save_path = f"{save_path}/models/"
if not os.path.exists(save_path): os.mkdir(save_path)
if not os.path.exists(model_save_path): os.mkdir(model_save_path)

# Train DQN algorithm on bsuite environments
if debug: bsuite_ids = ['bandit/0', 'cartpole/0']
else: bsuite_ids = sweep.SWEEP
  
for bsuite_id in bsuite_ids:
  # Load environment by id name, a list of id names can be found by printing sweep.SWEEP
  env = load_and_record(bsuite_id=bsuite_id,save_path=save_path,logging_mode='csv',overwrite=True)
  # Use DQN algorithm with default parameters for the demo
  agent = dqn.default_agent(obs_spec=env.observation_spec(), action_spec=env.action_spec())
  # Train on each environment for specified number of episodes
  if not num_episodes: num_episodes = getattr(env, 'bsuite_num_episodes')
  experiment.run(agent=agent,environment=env,num_episodes=num_episodes)
  # Save model for later use
  fname = model_save_path + f"{bsuite_id.replace('/', '_')}.pkl"
  with open(fname, 'wb+') as handle: pickle.dump(agent._online_network, handle)