
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf


ExpertNetworkType = collections.namedtuple(
    'expert_network', ['features', 'q_values', 'logits', 'probabilities'])
AMNNetworkType = collections.namedtuple(
    'amn_network', ['output', 'q_softmax', 'features'])


class ExpertNetwork(tf.keras.Model):

  def __init__(self, num_actions, num_atoms, support,
               feature_size, llamn_network=None, name=None):
    super(ExpertNetwork, self).__init__(name=name)
    activation_fn = tf.keras.activations.relu
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.support = support
    self.llamn_network = llamn_network

    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # Defining layers.
    self.conv1 = tf.keras.layers.Conv2D(
        32, [8, 8], strides=4, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(
        64, [4, 4], strides=2, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(
        64, [3, 3], strides=1, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
        512, activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(
        num_actions * num_atoms, kernel_initializer=self.kernel_initializer,
        name='fully_connected')

    self.stop_grad = tf.keras.layers.Lambda(
        lambda x: tf.stop_gradient(x), name='stop_gradient')

  def call(self, state):
    state = tf.cast(state, tf.float32)
    state = tf.div(state, 255.)
    x = self.conv1(state)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    features = self.dense1(x)

    if self.llamn_network:
      y = self.llamn_network(state)
      y = self.stop_grad(y)
      x = tf.concat([features, y], axis=1)
      # ################################################################ #
      #                               TODO                               #
      # ---------------------------------------------------------------- #
      # + Check if concatenation is OK                                   #
      # + Check if stop_grad is working and llamn_network is not updated #
      # ################################################################ #
      print("y :\n", y, '\n\n', '-'*200, '\n')    # debug
      print("x :\n", x, '\n\n', '-'*200, '\n')    # debug

    x = self.dense2(x)
    logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)
    return ExpertNetworkType(features, q_values, logits, probabilities)


class AMNNetwork(tf.keras.Model):

  def __init__(self, num_actions, feature_size, name=None):
    super(AMNNetwork, self).__init__(name=name)
    activation_fn = tf.keras.activations.relu

    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # Defining layers.
    self.conv1 = tf.keras.layers.Conv2D(
        32, [8, 8], strides=4, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(
        64, [4, 4], strides=2, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(
        64, [3, 3], strides=1, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
        512, activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='fully_connected')
    self.dense_output = tf.keras.layers.Dense(
        num_actions, activation=tf.keras.activations.softmax,
        kernel_initializer=self.kernel_initializer, name='fully_connected')
    self.dense_feature = tf.keras.layers.Dense(
        feature_size, kernel_initializer=self.kernel_initializer,
        name='fully_connected')

  def call(self, state):
    state = tf.cast(state, tf.float32)
    state = tf.div(state, 255.)
    x = self.conv1(state)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    output = self.dense1(x)
    probabilities = self.dense_output(output)
    feature = self.dense_feature(output)

    return AMNNetworkType(output, probabilities, feature)
