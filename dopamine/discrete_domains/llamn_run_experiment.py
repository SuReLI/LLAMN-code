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
import time
import sys
import functools
from multiprocessing import Process, Lock

from absl import logging

from dopamine.agents.llamn_network import expert_rainbow_agent, llamn_agent
from dopamine.discrete_domains.llamn_game_lib import create_games
from dopamine.discrete_domains.run_experiment import TrainRunner

import numpy as np
import tensorflow as tf

import gin.tf


def create_expert(sess, environment, llamn_path, name,
                  summary_writer=None, debug_mode=False):
  if not debug_mode:
    summary_writer = None

  return expert_rainbow_agent.ExpertAgent(
      sess, num_actions=environment.action_space.n,
      llamn_path=llamn_path, name=name, summary_writer=summary_writer)


@gin.configurable
class MasterRunner:

  def __init__(self, base_dir, parallel, games_names=None, sticky_actions=True):
    assert games_names is not None

    self.base_dir = base_dir
    self.parallel = parallel
    self.sentinel = os.path.join(base_dir, 'progress')

    self.games = [create_games(list_names) for list_names in games_names]

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

    logging.info('Running expert')

    print(f"\tRunning expert {game.name}")
    expert.run_experiment()
    self._write_sentinel(game.name)

  def run_llamn(self, base_dir, games, paths):
    llamn = LLAMNRunner(base_dir,
                        num_actions=self.max_num_actions,
                        expert_games=games,
                        expert_paths=paths)

    print(f"\tRunning llamn {self.curr_day}")
    logging.info('Running llamn')
    llamn.run_experiment()
    print('\n\n')

  def run_experiment(self):

    logging.info('Beginning Master Runner')
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
      last_experts_games = []
      for game in self.games[self.curr_day]:
        path = os.path.join(self.base_dir, f"day_{self.curr_day}/expert_{game.name}")
        last_experts_paths.append(path)
        last_experts_games.append(game)

      print(f"Running night {self.curr_day}")
      print(f"\tCreating llamn {self.curr_day}")
      base_dir = os.path.join(self.base_dir, f"night_{self.curr_day}")

      if self.parallel:
        llamn_proc = Process(target=self.run_llamn,
                             args=(base_dir, last_experts_games, last_experts_paths))

        llamn_proc.start()
        llamn_proc.join()

      else:
        self.run_llamn(base_dir, last_experts_games, last_experts_paths)

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

    tf.compat.v1.reset_default_graph()
    super().__init__(base_dir, create_expert_fn, environment.create)

    if self._start_iteration == 0:
      self._agent._load_llamn()

  def _save_tensorboard_summaries(self, iteration, num_episodes,
                                  average_reward, average_steps_per_second):
    """Save statistics as tensorboard summaries."""
    env_name = self._environment.name
    summary = tf.compat.v1.Summary(value=[
        tf.compat.v1.Summary.Value(
            tag=f'Train/{env_name}/NumEpisodes', simple_value=num_episodes),
        tf.compat.v1.Summary.Value(
            tag=f'Train/{env_name}/AverageReturns', simple_value=average_reward),
        tf.compat.v1.Summary.Value(
            tag=f'Train/{env_name}/AverageStepsPerSecond',
            simple_value=average_steps_per_second),
    ])
    self._summary_writer.add_summary(summary, iteration)

    summary = tf.compat.v1.Summary(value=[
        tf.compat.v1.Summary.Value(
            tag=f'TrainSteps/{env_name}/NumEpisodes', simple_value=num_episodes),
        tf.compat.v1.Summary.Value(
            tag=f'TrainSteps/{env_name}/AverageReturns', simple_value=average_reward),
        tf.compat.v1.Summary.Value(
            tag=f'TrainSteps/{env_name}/AverageStepsPerSecond',
            simple_value=average_steps_per_second),
    ])
    self._summary_writer.add_summary(summary, self._agent.training_steps)


@gin.configurable
class LLAMNRunner(TrainRunner):

  def __init__(self,
               base_dir,
               num_actions,
               expert_games,
               expert_paths,
               create_agent=llamn_agent.AMNAgent,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               nb_steps_per_steps=-1,
               training_steps=250000,
               max_steps_per_episode=27000,
               buffer_prefill=None,
               clip_rewards=True):
    assert base_dir is not None
    tf.compat.v1.disable_v2_behavior()

    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._nb_steps_per_steps = nb_steps_per_steps
    self._training_steps = training_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._base_dir = base_dir
    self._clip_rewards = clip_rewards
    self._create_directories()
    self._summary_writer = tf.compat.v1.summary.FileWriter(self._base_dir)
    self.buffer_prefill = buffer_prefill

    index = int(base_dir.rsplit('_', 1)[1])
    if index > 0:
      llamn_path = base_dir.replace(f'night_{index}', f'night_{index-1}')
    else:
      llamn_path = None

    self._names = [game.name for game in expert_games]
    self._environments = [game.create() for game in expert_games]
    self._nb_envs = len(self._environments)
    expert_num_actions = [env.action_space.n for env in self._environments]
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    tf.compat.v1.reset_default_graph()
    self._sess = tf.compat.v1.Session('', config=config)

    self._agent = create_agent(self._sess,
                               max_num_actions=num_actions,
                               expert_num_actions=expert_num_actions,
                               expert_paths=expert_paths,
                               llamn_path=llamn_path,
                               summary_writer=self._summary_writer)

    self._summary_writer.add_graph(graph=tf.compat.v1.get_default_graph())
    self._sess.run(tf.compat.v1.global_variables_initializer())

    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

    self._agent.load_networks()

    if self.buffer_prefill == "copy":
      self._agent.load_buffers()

  @property
  def _game_index(self):
    return self._agent.ind_expert

  @_game_index.setter
  def _game_index(self, value):
    self._agent.ind_expert = value

  @property
  def _environment(self):
    return self._environments[self._game_index]

  def _prefill_buffers(self):
    self._agent.eval_mode = False
    sys.stdout.write('\tPre-filling buffers...\n')
    while not self._agent._is_buffer_prefilled():
      self._run_one_episode()
      self._game_index = (self._game_index + 1) % self._nb_envs
    sys.stdout.write('\tDone !\n')

  def _run_one_phase_steps(self, min_steps, statistics, run_mode_str):
    step_count = [0] * self._nb_envs
    num_episodes = [0] * self._nb_envs
    sum_returns = [0.] * self._nb_envs

    actions = [self._initialize_episode() for self._game_index in range(self._nb_envs)]
    finished = [False] * self._nb_envs

    step_number = [0] * self._nb_envs
    total_reward = [0] * self._nb_envs

    while not all(finished):

      for self._game_index in range(self._nb_envs):
        if finished[self._game_index]:
          continue

        for _ in range(self._nb_steps_per_steps):
          # print("Running one step on ", self._environment.name)
          observation, reward, is_terminal = self._run_one_step(actions[self._game_index])

          total_reward[self._game_index] += reward
          step_number[self._game_index] += 1

          # Perform reward clipping.
          reward = np.clip(reward, -1, 1)

          if hasattr(self._environment, 'game_over'):
            game_over = self._environment.game_over
          else:
            game_over = is_terminal

          # End of episode
          if game_over or step_number[self._game_index] == self._max_steps_per_episode:
            num_episodes[self._game_index] += 1
            step_count[self._game_index] += step_number[self._game_index]
            sum_returns[self._game_index] += total_reward[self._game_index]

            name = self._names[self._game_index]
            statistics.append({
                '{}_{}_episode_lengths'.format(name, run_mode_str): step_number[self._game_index],
                '{}_{}_episode_returns'.format(name, run_mode_str): total_reward[self._game_index]
            })

            sys.stdout.write('\t\tSteps executed on {}: {} '.format(name, step_count[self._game_index]) +
                             'Episode length: {} '.format(step_number[self._game_index]) +
                             'Return: {}\r'.format(total_reward[self._game_index]))
            sys.stdout.flush()

            step_number[self._game_index] = 0
            total_reward[self._game_index] = 0
            self._end_episode(reward)

            if step_count[self._game_index] > min_steps:
              finished[self._game_index] = True
              break
            else:
              actions[self._game_index] = self._initialize_episode()

          # Loss of a life
          elif is_terminal:
            self._agent.end_episode(reward)
            actions[self._game_index] = self._agent.begin_episode(observation)

          else:
            actions[self._game_index] = self._agent.step(reward, observation)

    return step_count, sum_returns, num_episodes

  def _run_one_phase_episodes(self, min_steps, statistics, run_mode_str):
    step_count = [0] * self._nb_envs
    num_episodes = [0] * self._nb_envs
    sum_returns = [0.] * self._nb_envs

    # Continue to run every environment while at least one didn't finish its steps
    while min(step_count) < min_steps:
      episode_length, episode_return = self._run_one_episode()

      game_name = self._names[self._game_index]
      statistics.append({
          '{}_{}_episode_lengths'.format(game_name, run_mode_str): episode_length,
          '{}_{}_episode_returns'.format(game_name, run_mode_str): episode_return
      })
      step_count[self._game_index] += episode_length
      sum_returns[self._game_index] += episode_return
      num_episodes[self._game_index] += 1
      # We use sys.stdout.write instead of logging so as to flush frequently
      # without generating a line break.
      sys.stdout.write('\t\tSteps executed on {}: {} '.format(game_name, step_count[self._game_index]) +
                       'Episode length: {} '.format(episode_length) +
                       'Return: {}\r'.format(episode_return))
      sys.stdout.flush()

      self._game_index = (self._game_index + 1) % self._nb_envs

    return step_count, sum_returns, num_episodes

  def _run_one_phase(self, min_steps, statistics, run_mode_str):
    # Run by episodes
    if self._nb_steps_per_steps < 1:
      return self._run_one_phase_episodes(min_steps, statistics, run_mode_str)
    # Run by steps
    else:
      return self._run_one_phase_steps(min_steps, statistics, run_mode_str)

  def _run_train_phase(self, statistics):
    # Perform the training phase, during which the agent learns.
    self._agent.eval_mode = False
    start_time = time.time()
    number_steps, sum_returns, num_episodes = self._run_one_phase(
        self._training_steps, statistics, 'train')
    time_delta = time.time() - start_time

    average_returns = [0] * len(self._environments)
    average_steps_per_second = [0] * len(self._environments)
    for i in range(len(self._environments)):
      env_name = self._names[i]

      average_returns[i] = sum_returns[i] / num_episodes[i] if num_episodes[i] > 0 else 0.0
      statistics.append({f'{env_name}_train_average_return': average_returns[i]})

      average_steps_per_second[i] = number_steps[i] / time_delta
      statistics.append({f'{env_name}_train_average_steps_per_second': average_steps_per_second[i]})

    total_average = sum(average_returns) / len(average_returns)
    logging.info('Average undiscounted return per training episode: %.2f',
                 total_average)
    logging.info('Average training steps per second: %.2f',
                 sum(number_steps) / time_delta)
    return num_episodes, average_returns, average_steps_per_second

  def _run_one_iteration(self, iteration):
    logging.info('Starting iteration %d', iteration)
    print(f'\n\tLLAMN Running iteration {iteration}')
    super()._run_one_iteration(iteration)

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    logging.info('Beginning training...')
    if self._num_iterations <= self._start_iteration:
      logging.warning('num_iterations (%d) < start_iteration(%d)',
                      self._num_iterations, self._start_iteration)
      return

    if self.buffer_prefill == "exploration":
      self._prefill_buffers()

    for iteration in range(self._start_iteration, self._num_iterations):
      statistics = self._run_one_iteration(iteration)
      self._log_experiment(iteration, statistics)
      if iteration > 0 and iteration % 10 == 0 or iteration == self._num_iterations-1:
        self._checkpoint_experiment(iteration)

  def _save_tensorboard_summaries(self, iteration, num_episodes,
                                  average_reward, average_steps_per_second):
    """Save statistics as tensorboard summaries."""
    for i in range(len(num_episodes)):

      env_name = self._environments[i].name
      summary = tf.compat.v1.Summary(value=[
          tf.compat.v1.Summary.Value(
              tag=f'Train/{env_name}/NumEpisodes', simple_value=num_episodes[i]),
          tf.compat.v1.Summary.Value(
              tag=f'Train/{env_name}/AverageReturns', simple_value=average_reward[i]),
          tf.compat.v1.Summary.Value(
              tag=f'Train/{env_name}/AverageStepsPerSecond',
              simple_value=average_steps_per_second[i]),
      ])
      self._summary_writer.add_summary(summary, iteration)

      summary = tf.compat.v1.Summary(value=[
          tf.compat.v1.Summary.Value(
              tag=f'TrainSteps/{env_name}/NumEpisodes', simple_value=num_episodes[i]),
          tf.compat.v1.Summary.Value(
              tag=f'TrainSteps/{env_name}/AverageReturns', simple_value=average_reward[i]),
          tf.compat.v1.Summary.Value(
              tag=f'TrainSteps/{env_name}/AverageStepsPerSecond',
              simple_value=average_steps_per_second[i]),
      ])
      self._summary_writer.add_summary(summary, self._agent.training_steps_list[i])

    self._summary_writer.flush()
