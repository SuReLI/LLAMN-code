
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np
import tensorflow as tf

import gin
import gym
from dopamine.discrete_domains.atari_lib import AtariPreprocessing


gin.constant('llamn_game_lib.DUCKIE_OBSERVATION_SHAPE', (60, 80))
gin.constant('llamn_game_lib.DUCKIE_OBSERVATION_DTYPE', tf.float64)


def create_game(game_name, sticky_actions=True):
  if game_name.startswith('DuckieTown'):
    return DuckieTownGame(game_name)

  else:
    return AtariGame(game_name, sticky_actions)


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


class Game(ABC):
  variants = ('VHFlip', 'VFlip', 'HFlip', 'Noisy', 'Negative')

  def __init__(self, game_name):
    self.full_name = game_name

    self.variant = None
    for variant in self.variants:
        if game_name.endswith(variant):
            self.name = game_name[:-len(variant)]
            self.variant = variant
            break
    else:
      self.name = game_name

    self.finished = False     # used for multiprocessing

  @abstractmethod
  def create(self):
    pass

  def __repr__(self):
    return self.full_name


class DuckieTownGame(Game):

  def __init__(self, game_name):
    super().__init__(game_name)

    import gym_duckietown  # noqa: F401
    import gym_sim2real    # noqa: F401

    self.num_actions = 3

  def create(self):
    env = gym.make(self.name)
    env.environment = namedtuple('environment', ['game'])(self.name)
    return env


class AtariGame(Game):

  def __init__(self, game_name, sticky_actions):
    super().__init__(game_name)

    self.version = 'v0' if sticky_actions else 'v4'
    self.env_name = f'{self.name}NoFrameskip-{self.version}'

    env = gym.make(self.env_name)
    self.num_actions = env.action_space.n

  def create(self):
    return ModifiedAtariPreprocessing(gym.make(self.env_name).env, self.variant)
