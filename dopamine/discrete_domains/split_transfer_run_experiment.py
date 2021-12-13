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

import tensorflow as tf
from tensorflow.python.training import py_checkpoint_reader
import gin.tf

from dopamine.discrete_domains.llamn_game_lib import create_games
from dopamine.discrete_domains.run_experiment import TrainRunner, create_agent


@gin.configurable
def load_expert(self, ckpt_dir, nb_layers=4):

  ckpt = tf.compat.v1.train.get_checkpoint_state(os.path.join(ckpt_dir, "checkpoints"))
  ckpt_path = ckpt.model_checkpoint_path

  prev_layers = tf.train.list_variables(ckpt_path)
  prev_output_size = [layer[1] for layer in prev_layers if layer[0] == 'online/fully_connected_1/bias'][0][0]
  output_size = self._agent.online_convnet.variables[-1].shape.dims[0].value

  # Restore layers except the output one
  layer_names = ('Conv', 'Conv_1', 'Conv_2', 'fully_connected')
  filter_fn = lambda s: any((f'/{layer}/' in s for layer in layer_names[:nb_layers]))

  var_names = {'online/'+var.name.split('/', 1)[1][:-2]: var
               for var in self._agent.online_convnet.variables
               if filter_fn(var.name)}

  saver = tf.compat.v1.train.Saver(var_list=var_names)
  saver.restore(self._sess, ckpt_path)

  if nb_layers == 5 and prev_output_size != output_size:
    var_names = {'online/'+var.name.split('/', 1)[1][:-2]: var
                 for var in self._agent.online_convnet.variables
                 if '/fully_connected_1/' in var.name}
    weight_reader = py_checkpoint_reader.NewCheckpointReader(ckpt_path)
    prev_kernel = weight_reader.get_tensor('online/fully_connected_1/kernel')
    prev_bias = weight_reader.get_tensor('online/fully_connected_1/bias')

    agent_kernel, agent_bias = self._agent.online_convnet.variables[-2:]

    if prev_output_size < output_size:
      assign_kernel_op = agent_kernel[:, :prev_output_size].assign(prev_kernel)
      assign_bias_op = agent_bias[:prev_output_size].assign(prev_bias)
    else:
      assign_kernel_op = agent_kernel.assign(prev_kernel[:, :output_size])
      assign_bias_op = agent_bias.assign(prev_bias[:output_size])

    self._sess.run([assign_kernel_op, assign_bias_op])

  self._sess.run(self._agent._sync_qt_ops)


@gin.configurable
class SplitMasterRunner:

  def __init__(self, base_dir, phase, index, first_game_name=None,
               transferred_games_names=None, sticky_actions=True):

    self.base_dir = base_dir
    self.phase = phase
    self.index = index

    if transferred_games_names is None:
      transferred_games_names = []

    self.first_game = create_games([first_game_name])[0]
    self.transferred_games = create_games(transferred_games_names)

    self._save_gin_config()

  def _save_gin_config(self):
    if not os.path.exists(self.base_dir):
      os.makedirs(self.base_dir, exist_ok=True)

    config_file = os.path.join(self.base_dir, 'config.gin')
    with open(config_file, 'w') as config:
      config.write(gin.config_str())

  def run_expert(self, base_dir, game, ckpt_dir=None):
    tf.compat.v1.reset_default_graph()
    runner = TrainRunner(base_dir, create_agent, game.create)
    if ckpt_dir:
      runner.load_expert(ckpt_dir)
    runner.run_experiment()

  def run_experiment(self):
    print('Beginning Master Runner')
    tf.compat.v1.reset_default_graph()
    first_game_dir = os.path.join(self.base_dir, "day_0")

    # First expert
    if self.phase == "day_0":
      print("Running first expert")

      runner = TrainRunner(first_game_dir, create_agent, self.first_game.create)
      runner.run_experiment()

    # Next experts
    else:
      print("Running next expert", self.index)
      TrainRunner.load_expert = load_expert

      game = self.transferred_games[self.index]
      next_game_dir = os.path.join(self.base_dir, "day_1", game.name)

      runner = TrainRunner(next_game_dir, create_agent, game.create)
      runner.load_expert(first_game_dir)
      runner.run_experiment()
