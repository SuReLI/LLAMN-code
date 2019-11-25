# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module defining classes and helper methods for general agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.agents.llamn_network import expert_rainbow_agent, llamn_agent
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import llamn_atari_lib
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import logger

from dopamine.discrete_domains.run_experiment import TrainRunner

import tensorflow as tf

import gin.tf


def get_next_dir_index(base_dir, name):
  network_dirs = glob.glob(os.path.join(base_dir, name + "_*"))
  network_num = [0] + [int(exp_dir.rsplit('_', 1)[1]) for exp_dir in network_dirs]
  return max(network_num) + 1


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)


def create_expert(sess, environment, llamn_path, name,
                  summary_writer=None, debug_mode=False):
  if not debug_mode:
    summary_writer = None

  return expert_rainbow_agent.ExpertAgent(sess,
      num_actions=environment.action_space.n, llamn_path=llamn_path,
      name=name, summary_writer=summary_writer)

@gin.configurable
def create_runner(base_dir):
  assert base_dir is not None
  return MasterRunner(base_dir)


@gin.configurable
class MasterRunner:

  def __init__(self, base_dir):

    self.base_dir = base_dir

  def run_experiment(self):

    tf.logging.info('Beginning Master Runner')
    llamn_path = None
    experts_paths = []

    for i in range(10):

      expert = ExpertRunner(self.base_dir, create_expert, llamn_path)
      experts_paths.append(expert._base_dir)

      tf.logging.info('Running expert')
      expert.run_experiment()

      num_actions = expert._environment.action_space.n

      llamn = LLAMNRunner(self.base_dir, num_actions, experts_paths)
      llamn_path = llamn._base_dir

      tf.logging.info('Running llamn')
      llamn.run_experiment()


@gin.configurable
class ExpertRunner(TrainRunner):

  def __init__(self,
               base_dir,
               create_agent_fn,
               llamn_path=None):

    self._index = get_next_dir_index(base_dir, 'expert')
    name = f'expert_{self._index}'
    base_dir = os.path.join(base_dir, name)

    create_environment_fn = llamn_atari_lib.AtariEnvCreator()

    def create_expert_fn(*args, **kwargs):
      return create_agent_fn(*args, **kwargs, llamn_path=llamn_path, name=name)

    super().__init__(base_dir, create_expert_fn, create_environment_fn)

    self._agent._load_llamn()

@gin.configurable
class LLAMNRunner(TrainRunner):

  def __init__(self,
               base_dir,
               num_actions,
               expert_paths,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250000,
               evaluation_steps=125000,
               max_steps_per_episode=27000):

    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._index = get_next_dir_index(base_dir, 'llamn')
    self._name = f'llamn_{self._index}'
    self._base_dir = os.path.join(base_dir, self._name)
    self._create_directories()
    self._summary_writer = tf.summary.FileWriter(self._base_dir)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    self._sess = tf.Session('', config=config)

    self._agent = llamn_agent.AMNAgent(
        self._sess, num_actions=num_actions, 
        expert_paths=expert_paths, name=self._name,
        summary_writer=self._summary_writer)

    self._summary_writer.add_graph(graph=tf.get_default_graph())
    self._sess.run(tf.global_variables_initializer())

    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

    self._agent._load_experts()

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    tf.logging.info('Beginning training...')
    if self._num_iterations <= self._start_iteration:
      tf.logging.warning('num_iterations (%d) < start_iteration(%d)',
                         self._num_iterations, self._start_iteration)
      return

    for iteration in range(self._start_iteration, self._num_iterations):
      statistics = self._run_one_iteration(iteration)
      self._log_experiment(iteration, statistics)
      self._checkpoint_experiment(iteration)

