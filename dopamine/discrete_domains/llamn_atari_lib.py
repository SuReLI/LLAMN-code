
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf

import gym
from dopamine.discrete_domains.atari_lib import AtariPreprocessing


ExpertNetworkType = collections.namedtuple(
    'expert_network', ['features', 'q_values', 'logits', 'probabilities'])
AMNNetworkType = collections.namedtuple(
    'amn_network', ['output', 'logits', 'features'])


class Game:

  def __init__(self, game_name, sticky_actions=True):
    self.name = game_name
    self.version = 'v0' if sticky_actions else 'v4'

    self.full_name = f'{self.name}NoFrameskip-{self.version}'

    env = gym.make(self.full_name)
    self.num_actions = env.action_space.n

    self.finished = False     # used for multiprocessing

  def create(self):
    return AtariPreprocessing(gym.make(self.full_name).env)

  def __repr__(self):
    return self.name


class ExpertNetwork(tf.keras.Model):

  def __init__(self, num_actions, num_atoms, support,
               feature_size, llamn_name, name):
    super().__init__(name=name)
    activation_fn = tf.keras.activations.relu
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.support = support

    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # Defining layers.
    self.conv1 = tf.keras.layers.Conv2D(
        32, [8, 8], strides=4, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='conv_1')
    self.conv2 = tf.keras.layers.Conv2D(
        64, [4, 4], strides=2, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='conv_2')
    self.conv3 = tf.keras.layers.Conv2D(
        64, [3, 3], strides=1, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='conv_3')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
        512, activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='dense_1')
    self.dense2 = tf.keras.layers.Dense(
        feature_size, activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='dense_2')
    self.dense3 = tf.keras.layers.Dense(
        num_actions * num_atoms, kernel_initializer=self.kernel_initializer,
        name='dense_3')

    if llamn_name:
      self.llamn_network = AMNNetwork(num_actions, feature_size, llamn_name)
    else:
      self.llamn_network = None

  def call(self, state):
    state = tf.cast(state, tf.float32)
    state = tf.div(state, 255.)

    x = self.conv1(state)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)

    if self.llamn_network:
      llamn_output = self.llamn_network(state).output
      llamn_output = tf.stop_gradient(llamn_output)
      x = tf.concat([llamn_output, x], axis=1)
      # ################################################################ #
      #                               TODO                               #
      # ---------------------------------------------------------------- #
      # + Check if stop_grad is working and llamn_network is not updated #
      # ################################################################ #
      # print("llamn_output :\n", llamn_output, '\n\n', '-'*200, '\n')        # debug
      # print("x :\n", x, '\n\n', '-'*200, '\n')        # debug

    # print("self.llamn_network :\n", self.llamn_network, '\n\n', '-'*200, '\n')    # debug
    # print("x :\n", x, '\n\n', '-'*200, '\n')    # debug
    # print("features :\n", features, '\n\n', '-'*200, '\n')    # debug
    # print()        # debug
    features = self.dense2(x)
    output = self.dense3(features)
    logits = tf.reshape(output, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)

    return ExpertNetworkType(features, q_values, logits, probabilities)


class AMNNetwork(tf.keras.Model):

  def __init__(self, num_actions, feature_size, name):
    super().__init__(name=name)
    activation_fn = tf.keras.activations.relu

    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # Defining layers.
    self.conv1 = tf.keras.layers.Conv2D(
        32, [8, 8], strides=4, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='conv_1')
    self.conv2 = tf.keras.layers.Conv2D(
        64, [4, 4], strides=2, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='conv_2')
    self.conv3 = tf.keras.layers.Conv2D(
        64, [3, 3], strides=1, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='conv_3')
    self.flatten = tf.keras.layers.Flatten()

    self.dense1 = tf.keras.layers.Dense(
        512, activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='dense_1')
    self.dense_output = tf.keras.layers.Dense(
        num_actions, kernel_initializer=self.kernel_initializer,
        name='dense_out')
    self.dense_features = tf.keras.layers.Dense(
        feature_size, kernel_initializer=self.kernel_initializer,
        name='dense_feat')

  def call(self, state):
    state = tf.cast(state, tf.float32)
    state = tf.div(state, 255.)
    x = self.conv1(state)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    output = self.dense1(x)
    logits = self.dense_output(output)
    features = self.dense_features(output)

    return AMNNetworkType(output, logits, features)
