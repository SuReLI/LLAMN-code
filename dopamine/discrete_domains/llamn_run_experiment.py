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

from dopamine.agents.llamn_network import expert_rainbow_agent, llamn_agent
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains.llamn_atari_lib import Game

from dopamine.discrete_domains.run_experiment import TrainRunner

import tensorflow as tf

import gin.tf


def create_expert(sess, environment, llamn_path, name,
                  summary_writer=None, debug_mode=False):
  if not debug_mode:
    summary_writer = None

  return expert_rainbow_agent.ExpertAgent(
      sess, num_actions=environment.action_space.n, llamn_path=llamn_path,
      name=name, summary_writer=summary_writer)


@gin.configurable
class MasterRunner:

  def __init__(self, base_dir, resume, games_names=None, sticky_actions=True):
    assert games_names is not None

    if resume:
      self.base_dir = base_dir
    else:
      self.base_dir = base_dir
    self.ckpt = os.path.join(base_dir, 'progress')

    self.games = [[Game(game_name, sticky_actions) for game_name in list_names]
                  for list_names in games_names]

    self.max_num_actions = max([game.num_actions for game_list in self.games
                                for game in game_list])

    self._load_ckpt()

  def _load_ckpt(self):
    if os.path.exists(self.ckpt):
      with open(self.ckpt, 'r') as ckpt:
        progress = ckpt.read().split('_')
      self.curr_day = int(progress[1])
      self.curr_exp = int(progress[2])

    else:
      self.curr_day = 0
      self.curr_exp = 0

  def _write_ckpt(self):
    phase = 'Night' if self.curr_exp == len(self.games[self.curr_day]) else 'Day'
    data = f"{phase}_{self.curr_day}_{self.curr_exp}"

    with open(self.ckpt, 'w') as ckpt:
      ckpt.write(data)

  def run_experiment(self):

    tf.logging.info('Beginning Master Runner')
    print('Beginning Master Runner')

    while self.curr_day < len(self.games):

      print(f"Running day {self.curr_day}")

      llamn_path = None
      if self.curr_day > 0:
        llamn_path = os.path.join(self.base_dir, f"night_{self.curr_day-1}")

      while self.curr_exp < len(self.games[self.curr_day]):
        game = self.games[self.curr_day][self.curr_exp]

        print(f"\tCreating expert on game {game}")
        base_dir = os.path.join(self.base_dir, f"day_{self.curr_day}")
        expert = ExpertRunner(base_dir,
                              create_agent_fn=create_expert,
                              environment=game,
                              llamn_path=llamn_path)

        print(f"\tRunning expert on game {game}")
        tf.logging.info('Running expert')
        expert.run_experiment()
        print('\n\n')

        self._write_ckpt()
        self.curr_exp += 1

      self.curr_exp = 0

      last_experts_paths = []
      last_experts_envs = []
      for game in self.games[self.curr_day]:
        path = os.path.join(self.base_dir, f"day_{self.curr_day}/expert_{game.name}")
        last_experts_paths.append(path)
        last_experts_envs.append(game.create())

      print(f"Running night {self.curr_day}")
      print(f"\tCreating llamn {self.curr_day}")
      base_dir = os.path.join(self.base_dir, f"night_{self.curr_day}")
      llamn = LLAMNRunner(base_dir,
                          num_actions=self.max_num_actions,
                          expert_envs=last_experts_envs,
                          expert_paths=last_experts_paths)

      print(f"\tRunning llamn {self.curr_day}")
      tf.logging.info('Running llamn')
      llamn.run_experiment()
      print('\n\n')

      self._write_ckpt()
      self.curr_day += 1


class ExpertRunner(TrainRunner):

  def __init__(self,
               base_dir,
               create_agent_fn,
               environment,
               llamn_path=None):

    name = f'expert_{environment.name}'
    base_dir = os.path.join(base_dir, name)

    def create_expert_fn(*args, **kwargs):
      return create_agent_fn(*args, **kwargs, llamn_path=llamn_path, name=name)

    tf.reset_default_graph()
    super().__init__(base_dir, create_expert_fn, environment.create)

    self._agent._load_llamn()


@gin.configurable
class LLAMNRunner(TrainRunner):

  def __init__(self,
               base_dir,
               num_actions,
               expert_envs,
               expert_paths,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250000,
               max_steps_per_episode=27000):

    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._base_dir = base_dir
    self._create_directories()
    self._summary_writer = tf.summary.FileWriter(self._base_dir)

    index = int(base_dir.rsplit('_', 1)[1])
    if index > 0:
      llamn_path = base_dir.replace(f'night_{index}', f'night_{index-1}')
    else:
      llamn_path = None

    self.ind_env = 0
    self._environments = expert_envs
    expert_num_actions = [env.action_space.n for env in expert_envs]
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    self._sess = tf.Session('', config=config)

    self._agent = llamn_agent.AMNAgent(
        self._sess,
        max_num_actions=num_actions, expert_num_actions=expert_num_actions,
        llamn_path=llamn_path, expert_paths=expert_paths,
        summary_writer=self._summary_writer)

    self._summary_writer.add_graph(graph=tf.get_default_graph())
    self._sess.run(tf.global_variables_initializer())

    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

    self._agent._load_networks()

  @property
  def _environment(self):
    return self._environments[self.ind_env]

  def _run_one_iteration(self, iteration):
    statistics = iteration_statistics.IterationStatistics()
    tf.logging.info('Starting iteration %d', iteration)
    print(f'\n\tLLAMN Running iteration {iteration}')

    for i in range(len(self._environments)):

      self.ind_env = i
      self._agent.ind_expert = self.ind_env

      print(f'\t\tTraining LLAMN on {self._environment.environment.game}')
      num_episodes_train, average_reward_train = self._run_train_phase(
          statistics)

      self._save_tensorboard_summaries(iteration, num_episodes_train,
                                       average_reward_train)
    return statistics.data_lists
