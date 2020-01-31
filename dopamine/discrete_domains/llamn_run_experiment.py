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
import functools
from multiprocessing import Process, Lock

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

  def __init__(self, base_dir, parallel, games_names=None, sticky_actions=True):
    assert games_names is not None

    self.base_dir = base_dir
    self.parallel = parallel
    self.sentinel = os.path.join(base_dir, 'progress')

    self.games = [[Game(game_name, sticky_actions) for game_name in list_names]
                  for list_names in games_names]

    self.max_num_actions = max([game.num_actions for game_list in self.games
                                for game in game_list])

    self._save_gin_config()
    self._load_sentinel()

    self.lock = Lock()

  def _save_gin_config(self):
    if not os.path.exists(self.base_dir):
      os.makedirs(self.base_dir, exist_ok=True)

    config_file = os.path.join(self.base_dir, 'config.gin')
    with open(config_file, 'w') as config:
      config.write(gin.config_str())

  def _load_sentinel(self):
    if os.path.exists(self.sentinel):
      with open(self.sentinel, 'r') as sentinel:
        progress = sentinel.read().split()

    else:
      with open(self.sentinel, 'w') as sentinel:
        sentinel.write('Day_0\n')
      progress = ['Day_0']

    self.curr_day = int(progress[0].split('_')[1])
    if self.curr_day >= len(self.games):
      return

    if 'Day' in progress[0]:
      for game in self.games[self.curr_day]:
        game.finished = (game.name in progress[1:])

  def _write_sentinel(self, game=None):
    # End of the night
    if game is None:
      with open(self.sentinel, 'w') as sentinel:
        sentinel.write(f'Day_{self.curr_day}\n')

    # An expert just finished its day
    else:
      self.lock.acquire()
      try:
        with open(self.sentinel, 'a') as sentinel:
          sentinel.write(game + '\n')
      except FileNotFoundError:
        pass
      finally:
        self.lock.release()

  def run_expert(self, base_dir, llamn_path, game):
    expert = ExpertRunner(base_dir,
                          environment=game,
                          llamn_path=llamn_path)

    tf.logging.info('Running expert')

    print(f"\tRunning expert {game.name}")
    expert.run_experiment()
    self._write_sentinel(game.name)

  def run_llamn(self, base_dir, envs, paths):
    llamn = LLAMNRunner(base_dir,
                        num_actions=self.max_num_actions,
                        expert_envs=envs,
                        expert_paths=paths)

    print(f"\tRunning llamn {self.curr_day}")
    tf.logging.info('Running llamn')
    llamn.run_experiment()
    print('\n\n')

  def run_experiment(self):

    tf.logging.info('Beginning Master Runner')
    print('Beginning Master Runner')

    while self.curr_day < len(self.games):

      print(f"Running day {self.curr_day}")

      base_dir = os.path.join(self.base_dir, f"day_{self.curr_day}")
      llamn_path = None
      if self.curr_day > 0:
        llamn_path = os.path.join(self.base_dir, f"night_{self.curr_day-1}")

      if self.parallel:
        # Run experts in different processes
        processes = []
        for game in self.games[self.curr_day]:
          if not game.finished:
            processes.append(Process(target=self.run_expert,
                                     args=(base_dir, llamn_path, game,)))

        for proc in processes:
          proc.start()
        for proc in processes:
          proc.join()

      else:
        for game in self.games[self.curr_day]:
          if not game.finished:
            self.run_expert(base_dir, llamn_path, game)

      # Run LLAMN
      last_experts_paths = []
      last_experts_envs = []
      for game in self.games[self.curr_day]:
        path = os.path.join(self.base_dir, f"day_{self.curr_day}/expert_{game.name}")
        last_experts_paths.append(path)
        last_experts_envs.append(game.create())

      print(f"Running night {self.curr_day}")
      print(f"\tCreating llamn {self.curr_day}")
      base_dir = os.path.join(self.base_dir, f"night_{self.curr_day}")

      if self.parallel:
        llamn_proc = Process(target=self.run_llamn,
                             args=(base_dir, last_experts_envs, last_experts_paths))

        llamn_proc.start()
        llamn_proc.join()

      else:
        self.run_llamn(base_dir, last_experts_envs, last_experts_paths)

      self.curr_day += 1
      self._write_sentinel()


class ExpertRunner(TrainRunner):

  def __init__(self,
               base_dir,
               environment,
               create_agent_fn=create_expert,
               llamn_path=None):

    name = f'expert_{environment.name}'
    base_dir = os.path.join(base_dir, name)

    create_expert_fn = functools.partial(create_agent_fn,
                                         llamn_path=llamn_path,
                                         name=name)

    tf.reset_default_graph()
    super().__init__(base_dir, create_expert_fn, environment.create)

    self._agent._load_llamn()

  def _save_tensorboard_summaries(self, iteration, num_episodes,
                                  average_reward):
    """Save statistics as tensorboard summaries."""
    game_name = self._environment.environment.game.capitalize()
    summary = tf.Summary(value=[
        tf.Summary.Value(
            tag=f'Train/{game_name}/NumEpisodes', simple_value=num_episodes),
        tf.Summary.Value(
            tag=f'Train/{game_name}/AverageReturns', simple_value=average_reward),
    ])
    self._summary_writer.add_summary(summary, iteration)


@gin.configurable
class LLAMNRunner(TrainRunner):

  def __init__(self,
               base_dir,
               num_actions,
               expert_envs,
               expert_paths,
               create_agent=llamn_agent.AMNAgent,
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

    self._agent = create_agent(self._sess,
                               max_num_actions=num_actions,
                               expert_num_actions=expert_num_actions,
                               llamn_path=llamn_path,
                               expert_paths=expert_paths,
                               summary_writer=self._summary_writer)

    self._summary_writer.add_graph(graph=tf.get_default_graph())
    self._sess.run(tf.global_variables_initializer())

    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

    self._agent.load_networks()

  @property
  def _environment(self):
    return self._environments[self.ind_env]

  def _run_one_iteration(self, iteration):
    statistics = iteration_statistics.IterationStatistics()
    tf.logging.info('Starting iteration %d', iteration)
    print(f'\n\tLLAMN Running iteration {iteration}')

    for self.ind_env in range(len(self._environments)):

      self._agent.ind_expert = self.ind_env

      print(f'\t\tTraining LLAMN on {self._environment.environment.game}')
      num_episodes_train, average_reward_train = self._run_train_phase(
          statistics)

      self._save_tensorboard_summaries(iteration, num_episodes_train,
                                       average_reward_train)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration, num_episodes,
                                  average_reward):
    """Save statistics as tensorboard summaries."""
    game_name = self._environment.environment.game.capitalize()
    summary = tf.Summary(value=[
        tf.Summary.Value(
            tag=f'Train/{game_name}/NumEpisodes', simple_value=num_episodes),
        tf.Summary.Value(
            tag=f'Train/{game_name}/AverageReturns', simple_value=average_reward),
    ])
    self._summary_writer.add_summary(summary, iteration)
    self._summary_writer.flush()
