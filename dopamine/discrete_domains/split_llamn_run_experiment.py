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
import fcntl

from dopamine.discrete_domains.llamn_game_lib import create_games
from dopamine.discrete_domains.llamn_run_experiment import ExpertRunner, LLAMNRunner

import gin.tf


@gin.configurable
class SplitMasterRunner:

  def __init__(self, base_dir, phase, index, games_names=None, sticky_actions=True):
    assert games_names is not None

    self.base_dir = base_dir
    self.sentinel = os.path.join(base_dir, 'progress')

    self.games = [create_games(list_names) for list_names in games_names]
    self.is_day = phase.startswith('day')
    self.curr_day = int(phase.split('_')[1])

    if self.is_day:
      self.day_game = self.games[self.curr_day][index]

    else:
      self.max_num_actions = max([game.num_actions for game_list in self.games
                                  for game in game_list])

    if phase.lower() == 'day_0' and index == 0:
      self._save_gin_config()
    self._load_sentinel()

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
      with open(self.sentinel, 'a') as sentinel:
        fcntl.flock(sentinel, fcntl.LOCK_EX)
        sentinel.write(game + '\n')
        fcntl.flock(sentinel, fcntl.LOCK_UN)

  def run_experiment(self):

    if self.is_day:
      if self.day_game.finished:
        return

      base_dir = os.path.join(self.base_dir, f"day_{self.curr_day}")
      llamn_path = None
      if self.curr_day > 0:
        llamn_path = os.path.join(self.base_dir, f"night_{self.curr_day-1}")

      expert = ExpertRunner(base_dir, environment=self.day_game, llamn_path=llamn_path)

      print(f"\tRunning expert {self.day_game.name}")
      expert.run_experiment()
      self._write_sentinel(self.day_game.name)

    # Run LLAMN
    else:
      last_expert_games = []
      last_expert_paths = []
      for game in self.games[self.curr_day]:
        path = os.path.join(self.base_dir, f"day_{self.curr_day}/expert_{game.name}")
        last_expert_games.append(game)
        last_expert_paths.append(path)

      base_dir = os.path.join(self.base_dir, f"night_{self.curr_day}")

      llamn = LLAMNRunner(base_dir, num_actions=self.max_num_actions,
                          expert_games=last_expert_games, expert_paths=last_expert_paths)

      print(f"\tRunning llamn {self.curr_day}")
      llamn.run_experiment()
      self._write_sentinel()
