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

import sys
import os
import time
import glob

from dopamine.agents.llamn_network import expert_rainbow_agent, llamn_agent
from dopamine.discrete_domains import llamn_atari_lib
from dopamine.discrete_domains import iteration_statistics

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

  return expert_rainbow_agent.ExpertAgent(
      sess, num_actions=environment.action_space.n, llamn_path=llamn_path,
      name=name, summary_writer=summary_writer)


class MasterRunner:

  def __init__(self, base_dir):

    self.base_dir = base_dir

  def run_experiment(self):

    tf.logging.info('Beginning Master Runner')
    print('Beginning Master Runner')
    llamn_path = None

    for i in range(10):

      print(f"Running iteration {i}")
      print(f"\tCreating expert {i}")
      expert = ExpertRunner(self.base_dir, create_expert, llamn_path)
      last_experts_path = [expert._base_dir]

      print(f"\tRunning expert {i}")
      tf.logging.info('Running expert')
      expert.run_experiment()
      print()

      num_actions = expert._environment.action_space.n

      print(f"\tCreating llamn {i}")
      llamn = LLAMNRunner(self.base_dir, num_actions, last_experts_path)
      llamn_path = llamn._base_dir

      print(f"\tRunning llamn {i}")
      tf.logging.info('Running llamn')
      llamn.run_experiment()
      print()


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
               training_steps=250000):

    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._index = get_next_dir_index(base_dir, 'llamn')
    self._name = f'llamn_{self._index}'
    self._base_dir = os.path.join(base_dir, self._name)
    self._create_directories()
    self._summary_writer = tf.summary.FileWriter(self._base_dir)

    if self._index > 1:
      llamn_path = os.path.join(base_dir, f'llamn_{self._index - 1}')
    else:
      llamn_path = None

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    self._sess = tf.Session('', config=config)

    self._agent = llamn_agent.AMNAgent(
        self._sess, num_actions=num_actions,
        llamn_path=llamn_path, expert_paths=expert_paths,
        name=self._name, summary_writer=self._summary_writer)

    self._summary_writer.add_graph(graph=tf.get_default_graph())
    self._sess.run(tf.global_variables_initializer())

    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

    self._agent._load_networks()

  def _run_one_phase(self, min_steps):
    step_count = 0

    while step_count < min_steps:
      self._agent.step()

      if step_count % 1000 == 0:
        sys.stdout.write('\tSteps executed: {}\r'.format(step_count))
        sys.stdout.flush()

      step_count += 1

    return step_count

  def _run_train_phase(self, statistics):
    self._agent.eval_mode = False

    start_time = time.time()
    number_steps = self._run_one_phase(self._training_steps)
    time_delta = time.time() - start_time

    number_steps_per_sec = number_steps / time_delta
    tf.logging.info('Average trainning steps per second: %.2f',
                    number_steps_per_sec)

    return number_steps_per_sec

  def _run_one_iteration(self, iteration):
    statistics = iteration_statistics.IterationStatistics()
    tf.logging.info('Starting iteration %d', iteration)
    nb_steps_per_sec = self._run_train_phase(statistics)

    self._save_tensorboard_summaries(iteration, nb_steps_per_sec)
    return statistics.data_lists

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

  def _save_tensorboard_summaries(self, iteration, nb_steps_per_sec):
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Train/NumStepsPerSec', simple_value=nb_steps_per_sec)
    ])
    self._summary_writer.add_summary(summary, iteration)
