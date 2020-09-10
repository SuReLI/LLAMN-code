
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
    'amn_network', ['pre_features', 'q_values', 'features'])
AMNNetworkDistributionalType = collections.namedtuple(
    'amn_network', ['pre_features', 'q_values', 'probabilities', 'features'])


def build_variant(variant):
  if variant is None:        # No variant
      return lambda image: image
  elif variant == "VHFlip":
      return lambda image: np.flip(image, (0, 1))
  elif variant == "VFlip":
      return lambda image: np.flip(image, 0)
  elif variant == "HFlip":
      return lambda image: np.flip(image, 1)
  elif variant == "Noisy":
      noise = np.random.normal(0, 3, (84, 84, 1)).astype(np.uint8)
      return lambda image: image + noise
  elif variant == "Negative":
      return lambda image: (255 - image)
  raise ValueError("Variant name invalid !")


class ModifiedAtariPreprocessing(AtariPreprocessing):
  def __init__(self, environment, variant):
    self.variant = variant
    self.process_variant = build_variant(variant)
    super().__init__(environment)

  def _pool_and_resize(self):
    image = super()._pool_and_resize()
    return self.process_variant(image)


class Game:
  variants = ('VHFlip', 'VFlip', 'HFlip', 'Noisy', 'Negative')

  def __init__(self, game_name, sticky_actions=True):
    self.variant = None
    for variant in self.variants:
        if game_name.endswith(variant):
            gym_name = game_name[:-len(variant)]
            self.variant = variant
            break
    else:
      gym_name = game_name

    self.name = game_name
    self.version = 'v0' if sticky_actions else 'v4'

    self.full_name = f'{gym_name}NoFrameskip-{self.version}'

    env = gym.make(self.full_name)
    self.num_actions = env.action_space.n

    self.finished = False     # used for multiprocessing

  def create(self):
    return ModifiedAtariPreprocessing(gym.make(self.full_name).env,
                                      self.variant)

  def __repr__(self):
    return self.name


class ExpertNetwork(tf.keras.Model):

  def __init__(self, num_actions, num_atoms, support, feature_size,
               create_llamn, init_option, distributional_night, name):
    super().__init__(name=name)
    activation_fn = tf.keras.activations.relu
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.support = support
    self.init_option = init_option

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
        feature_size, activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='dense_1')
    self.dense_output = tf.keras.layers.Dense(
        num_actions * num_atoms, kernel_initializer=self.kernel_initializer,
        name='dense_out')

    self.llamn_network = None
    if init_option == 3:
      self.dense_features = tf.keras.layers.Dense(
          feature_size, activation=activation_fn,
          kernel_initializer=self.kernel_initializer, name='dense_feat')

      if create_llamn:
        if not distributional_night:
          self.llamn_network = AMNNetwork(num_actions, None, None, feature_size, 'llamn')
        else:
          self.llamn_network = AMNNetwork(num_actions, num_atoms, support, feature_size, 'llamn')

  def call(self, state):
    state = tf.cast(state, tf.float32)
    state = state / 255

    x = self.conv1(state)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    features = self.dense1(x)

    if self.init_option == 3 and self.llamn_network:
      llamn_pre_features = self.llamn_network(state).pre_features
      llamn_pre_features = tf.stop_gradient(llamn_pre_features)
      features = tf.concat([features, llamn_pre_features], axis=1)
      features = self.dense_features(features)

    output = self.dense_output(features)
    logits = tf.reshape(output, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)

    return ExpertNetworkType(features, q_values, logits, probabilities)


class AMNNetwork(tf.keras.Model):

  def __init__(self, num_actions, num_atoms, support, feature_size, name):
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
        feature_size, activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='dense_1')
    self.dense_features = tf.keras.layers.Dense(
        feature_size, kernel_initializer=self.kernel_initializer,
        name='dense_feat')

    # Not distributional
    if not self.num_atoms:
      self.dense_output = tf.keras.layers.Dense(
          num_actions, kernel_initializer=self.kernel_initializer,
          name='dense_out')
    else:
      self.dense_output = tf.keras.layers.Dense(
          num_actions * num_atoms, kernel_initializer=self.kernel_initializer,
          name='dense_out')

  def call(self, state):
    state = tf.cast(state, tf.float32)
    state = state / 255

    x = self.conv1(state)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    pre_features = self.dense1(x)
    features = self.dense_features(pre_features)

    if not self.num_atoms:
      q_values = self.dense_output(pre_features)

      return AMNNetworkType(pre_features, q_values, features)

    else:
      output = self.dense_output(pre_features)
      logits = tf.reshape(output, [-1, self.num_actions, self.num_atoms])
      probabilities = tf.keras.activations.softmax(logits)
      q_values = tf.reduce_sum(self.support * probabilities, axis=2)

      return AMNNetworkDistributionalType(pre_features, q_values, probabilities, features)
