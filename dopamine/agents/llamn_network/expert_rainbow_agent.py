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
"""Compact implementation of a simplified Rainbow agent.

Specifically, we implement the following components from Rainbow:

  * n-step updates;
  * prioritized replay; and
  * distributional RL.

These three components were found to significantly impact the performance of
the Atari game-playing agent.

Furthermore, our implementation does away with some minor hyperparameter
choices. Specifically, we

  * keep the beta exponent fixed at beta=0.5, rather than increase it linearly;
  * remove the alpha parameter, which was set to alpha=0.5 throughout the paper.

Details in "Rainbow: Combining Improvements in Deep Reinforcement Learning" by
Hessel et al. (2018).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import llamn_atari_lib
import tensorflow as tf

import gin.tf


@gin.configurable
class ExpertAgent(rainbow_agent.RainbowAgent):
  """A compact implementation of a simplified Rainbow agent."""

  def __init__(self,
               sess,
               num_actions,
               llamn_path,
               name,
               feature_size=512,
               observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
               stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
               network=llamn_atari_lib.ExpertNetwork,
               distributional_night=False,
               init_option=1,
               num_atoms=51,
               vmax=10.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               tf_device='/cpu:*',
               optimizer=tf.compat.v1.train.AdamOptimizer(
                   learning_rate=0.00025, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=500):

    self.llamn_path = llamn_path
    self.name = name
    self.init_option = init_option
    self.feature_size = feature_size
    self.distributional_night = False

    super().__init__(
        sess=sess,
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network,
        num_atoms=num_atoms,
        vmax=vmax,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        replay_scheme=replay_scheme,
        tf_device=tf_device,
        optimizer=optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

  def _load_llamn(self):
    if self.llamn_path:

      # Initialization by weights copy
      if self.init_option == 1:
        # We want to restore variables with names 'expert_Pong/online/conv_1:0'
        # from variables with names 'llamn/conv_1'
        var_names = {('llamn/'+var.name.split('/', 2)[2][:-2]): var
                     for var in self.online_convnet.variables
                     if 'dense_out' not in var.name}

      elif self.init_option == 2:
        raise NotImplementedError("This initialization option is not implemented yet")

      elif self.init_option == 3:
        # Restore llamn variables with names 'expert_pong/online/llamn/conv_1:0'
        # from variables with names 'llamn/conv_1'
        var_names = {var.name.split('/', 2)[2][:-2]: var
                     for var in self.online_convnet.variables
                     if 'llamn' in var.name}

      ckpt = tf.compat.v1.train.get_checkpoint_state(self.llamn_path + "/checkpoints")
      ckpt_path = ckpt.model_checkpoint_path

      saver = tf.compat.v1.train.Saver(var_list=var_names)
      saver.restore(self._sess, ckpt_path)

      self._sess.run(self._sync_qt_ops)

  def _create_network(self, name):
    """Builds a convolutional network that outputs Q-value distributions.

    Args:
      name: str, this name is passed to the tf.keras.Model and used to create
        variable scope under the hood by the tf.keras.Model.
    Returns:
      network: tf.keras.Model, the network instantiated by the Keras model.
    """
    scope_name = self.name + '/' + name

    network = self.network(self.num_actions,
                           self._num_atoms,
                           self._support,
                           self.feature_size,
                           create_llamn=self.llamn_path,
                           init_option=self.init_option,
                           distributional_night=self.distributional_night,
                           name=scope_name)
    return network

  def _build_sync_op(self):
    sync_qt_ops = []
    scope = tf.compat.v1.get_default_graph().get_name_scope()
    trainables_online = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, self.name, 'online'))
    trainables_target = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.join(scope, self.name, 'target'))

    assert trainables_online, "No variables found for online network"
    assert trainables_target, "No variables found for online target"

    for (w_online, w_target) in zip(trainables_online, trainables_target):
      # Assign weights from online to target network.
      sync_qt_ops.append(w_target.assign(w_online, use_locking=True))
    return sync_qt_ops

  def fill_sleeping_memory(self, sleeping_memory, nb_transitions):
    nb_tr = 0
    while nb_tr < nb_transitions:
      # ################################################################ #
      #                               TODO                               #
      # ---------------------------------------------------------------- #
      # + Check if the shapes of the tensors are correct                 #
      # + Check if there are really batch_size elements more each time   #
      # + Check if _net_outputs is really the output from the state_ph   #
      # ################################################################ #
      sleeping_memory.add(self.state_ph, self._net_outputs.features)
      nb_tr += self.batch_size
