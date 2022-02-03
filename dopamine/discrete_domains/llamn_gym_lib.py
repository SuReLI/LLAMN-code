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
"""Gym-specific (non-Atari) utilities.

Some network specifications specific to certain Gym environments are provided
here.

Includes a wrapper class around Gym environments. This class makes general Gym
environments conformant with the API Dopamine is expecting.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from dopamine.discrete_domains import llamn_atari_lib
import gin
import tensorflow as tf


@gin.configurable
class GymExpertNetwork(tf.keras.Model):

  def __init__(self, num_actions, num_atoms, support, feature_size,
               create_llamn, init_option, distributional_night, name):
    super().__init__(name=name)
    activation_fn = tf.keras.activations.relu
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.support = support

    # Defining layers
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                        name='dense_1')
    self.dense2 = tf.keras.layers.Dense(feature_size, activation=activation_fn,
                                        name='dense_2')
    self.last_layer = tf.keras.layers.Dense(num_actions * num_atoms,
                                            name='dense_3')

  def call(self, state):
    x = tf.cast(state, tf.float32)
    x = self.flatten(x)
    x = self.dense1(x)
    features = self.dense2(x)
    x = self.last_layer(features)

    logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)
    return llamn_atari_lib.ExpertNetworkType(features, q_values, logits, probabilities)


@gin.configurable
class GymAMNNetwork(tf.keras.Model):

  def __init__(self, num_actions, num_atoms, support, feature_size, name):
    super().__init__(name=name)
    activation_fn = tf.keras.activations.relu
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.support = support

    # Defining layers
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn, name='dense_1')
    self.dense2 = tf.keras.layers.Dense(feature_size, activation=activation_fn, name='dense_2')

    # Not distributional
    if not self.num_atoms:
      self.last_layer = tf.keras.layers.Dense(num_actions, name='dense_3')
    else:
      self.last_layer = tf.keras.layers.Dense(num_actions * num_atoms, name='dense_3')

  def call(self, state):
    x = tf.cast(state, tf.float32)
    x = self.flatten(x)
    x = self.dense1(x)
    features = self.dense2(x)
    x = self.last_layer(features)

    if not self.num_atoms:
      q_values = self.last_layer(features)
      return llamn_atari_lib.AMNNetworkType(q_values, features)

    else:
      output = self.last_layer(features)
      logits = tf.reshape(output, [-1, self.num_actions, self.num_atoms])
      probabilities = tf.keras.activations.softmax(logits)
      q_values = tf.reduce_sum(self.support * probabilities, axis=2)
      return llamn_atari_lib.AMNNetworkDistributionalType(q_values, probabilities, features)
