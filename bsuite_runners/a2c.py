# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Code adapted from: https://github.com/deepmind/bsuite/blob/main/bsuite/baselines/tf/actor_critic/run.py
#                    https://github.com/google-deepmind/bsuite/blob/main/bsuite/baselines/tf/actor_critic/agent.py
# Changes: using actor_critic.default_agent
#          added timing and model-saving functionality
#          allows editing of num_processes
#          garbage collection
# ============================================================================
"""Run an actor-critic agent instance on a bsuite experiment."""

from absl import app
from absl import flags

import bsuite
from bsuite import sweep

from bsuite.baselines import experiment
from bsuite.baselines.tf import actor_critic
from bsuite.baselines.utils import pool

import os
import pickle
import gc
import time

# Internal imports.

# Experiment flags.
flags.DEFINE_string(
    'bsuite_id', 'catch/0', 'BSuite identifier. '
    'This global flag can be used to control which environment is loaded.')
flags.DEFINE_string('save_path', '/tmp/bsuite', 'where to save bsuite results')
flags.DEFINE_enum('logging_mode', 'csv', ['csv', 'sqlite', 'terminal'],
                  'which form of logging to use for bsuite results')
flags.DEFINE_boolean('overwrite', False, 'overwrite csv logging if found')
flags.DEFINE_integer('num_episodes', None, 'Overrides number of training eps.')
flags.DEFINE_integer('num_processes', None, 'number of mpi workers')

# algorithm
flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')

FLAGS = flags.FLAGS

def run(bsuite_id: str) -> str:
  """Runs A2C agent on a single bsuite environment, logging to CSV."""
  fpath = f"{FLAGS.save_path}/models/{bsuite_id.replace('/', '_')}.pkl"
  if os.path.exists(fpath): return bsuite_id

  env = bsuite.load_and_record(
      bsuite_id=bsuite_id,
      save_path=FLAGS.save_path,
      logging_mode=FLAGS.logging_mode,
      overwrite=FLAGS.overwrite,
  )

  agent = actor_critic.default_agent(
      obs_spec=env.observation_spec(),
      action_spec=env.action_spec(),
  )

  num_episodes = FLAGS.num_episodes or getattr(env, 'bsuite_num_episodes')

  start = time.time()
  experiment.run(
      agent=agent,
      environment=env,
      num_episodes=num_episodes,
      verbose=FLAGS.verbose)
  end = time.time()
  print(f"Time elapsed for {bsuite_id}: {int(end - start)}", flush=True)

  # save
  fpath = f"{FLAGS.save_path}/models/"
  if not os.path.exists(fpath): os.makedirs(fpath, exist_ok=True)
  fname = fpath + f"{bsuite_id.replace('/', '_')}.pkl"
  with open(fname, 'wb+') as handle:
      pickle.dump(agent._network, handle)

  # garbage collection
  agent = env = None
  gc.collect()

  return bsuite_id


def main(_):
  # Parses whether to run a single bsuite_id, or multiprocess sweep.
  bsuite_id = FLAGS.bsuite_id

  if bsuite_id in sweep.SWEEP:
    print(f'Running single experiment: bsuite_id={bsuite_id}.')
    run(bsuite_id)

  elif bsuite_id.isnumeric():
    bsuite_sweep = [x for x in sweep.SWEEP if x.endswith(bsuite_id)]
    print(f'Running sweep over bsuite_id in sweep.{bsuite_sweep}')
    FLAGS.verbose = False
    pool.map_mpi(run, bsuite_sweep, num_processes=FLAGS.num_processes)

  elif hasattr(sweep, bsuite_id):
    bsuite_sweep = getattr(sweep, bsuite_id)
    print(f'Running sweep over bsuite_id in sweep.{bsuite_sweep}')
    FLAGS.verbose = False
    pool.map_mpi(run, bsuite_sweep, num_processes=FLAGS.num_processes)

  else:
    raise ValueError(f'Invalid flag: bsuite_id={bsuite_id}.')


if __name__ == '__main__':
  app.run(main)