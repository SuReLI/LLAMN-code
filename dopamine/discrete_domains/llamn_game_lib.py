
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np
import tensorflow as tf

import gin
import gym
from gym.wrappers.time_limit import TimeLimit
from dopamine.discrete_domains.atari_lib import AtariPreprocessing


gin.constant('llamn_game_lib.DUCKIE_OBSERVATION_SHAPE', (60, 80))
gin.constant('llamn_game_lib.DUCKIE_OBSERVATION_DTYPE', tf.uint8)

gin.constant('llamn_game_lib.PROCGEN_STACK_SIZE', 1)
gin.constant('llamn_game_lib.PROCGEN_OBSERVATION_SHAPE', (64, 64, 3))
gin.constant('llamn_game_lib.PROCGEN_OBSERVATION_DTYPE', tf.uint8)

gin.constant('llamn_game_lib.PENDULUM_OBSERVATION_DTYPE', tf.float64)
gin.constant('llamn_game_lib.PENDULUM_STACK_SIZE', 1)


@gin.configurable
def create_games(games_names, render=False):
  games = []
  for game_name in games_names:
    # Range of levels (ProcGen)
    if ':' in game_name:
      env_name, infos = game_name.split('.')
      if '-' in infos:
        num_levels, start_level = infos.split('-')
      else:
        start_level = infos
        num_levels = None

      first_level, last_level = map(int, start_level.split(':'))
      for level in range(first_level, last_level):
        if num_levels:
          new_game_name = f"{env_name}.{num_levels}-{level}"
        else:
          new_game_name = f"{env_name}.{level}"
        games.append(create_game(new_game_name, render))

    else:
      games.append(create_game(game_name, render))
  return games


def create_game(game_name, render=False):
  if game_name.startswith('DuckieTown'):
    return DuckieTownGame(game_name)

  elif game_name.startswith('Procgen'):
    return ProcGenGame(game_name, render)

  elif game_name.startswith('Gym'):
    return GymGame(game_name, render)

  else:
    return AtariGame(game_name, True)


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


@gin.configurable
class GymPreprocessing(gym.Wrapper):
  """A Wrapper class around Gym environments."""

  def __init__(self, env, name, level, n_features, n_redundant, n_repeated, n_noisy, render=False):
    super().__init__(env)

    self.name = name
    self.level = level
    self.game_over = False
    self._render = render

    self.n_informative = self.observation_space.shape[0]
    self.n_features = n_features
    self.n_redundant = n_redundant
    self.n_repeated = n_repeated
    self.n_noisy = n_noisy
    self.n_useless = n_features - self.n_informative - n_redundant - n_repeated - n_noisy

    min_action = self.env.action_space.low[0]
    max_action = self.env.action_space.high[0]
    self.action_mapping = np.linspace(min_action, max_action, 3)[:, np.newaxis]
    self.action_space = gym.spaces.Discrete(3)

    self.min_observation = self.env.observation_space.low
    self.max_observation = self.env.observation_space.high
    self.observation_space = gym.spaces.Box(-np.ones(self.n_features),
                                            np.ones(self.n_features),
                                            (self.n_features, ))

    generator = np.random.default_rng(self.level)
    self.redundant_comatrix = 2 * generator.random((self.n_informative, self.n_redundant)) - 1
    fixed_generator = np.random.default_rng(0)
    self.noisy_redundant_comatrix = 2 * fixed_generator.random((self.n_informative, self.n_noisy)) - 1
    n = self.n_informative + self.n_redundant + self.n_noisy
    self.indices_copy = ((n - 1) * generator.random(self.n_repeated) + 0.5).astype(np.intp)

    self.eval_mode = False
    self.eval_index = 0
    theta = np.linspace(-np.pi, np.pi, 10)
    theta_dot = np.linspace(-8, 8, 5)
    theta = theta.repeat(5)
    theta_dot = np.tile(theta_dot, 10)
    self.eval_states = np.transpose([theta, theta_dot])

  def reset(self, **kwargs):
    observation = super().reset(**kwargs)

    if self.eval_mode:
      self.env.env.state = self.eval_states[self.eval_index]
      self.eval_index = (self.eval_index + 1) % len(self.eval_states)
      observation = self.env.env._get_obs()

    return self.randomize(observation)

  def render(self, mode='human'):
    if self._render:
      super().render(mode)

  def step(self, action):
    action = self.action_mapping[action]
    observation, reward, game_over, info = self.env.step(action)
    observation = self.randomize(observation)
    was_truncated = info.get('TimeLimit.truncated', False)
    game_over = game_over and not was_truncated
    self.game_over = game_over
    return observation, reward, game_over, info

  def randomize(self, observation):
    observation = (observation - self.min_observation) / (self.max_observation - self.min_observation)
    observation = 2 * observation - 1

    X = np.zeros(self.n_features)
    X[:self.n_informative] = observation

    if self.n_redundant > 0:
      X[self.n_informative: self.n_informative + self.n_redundant] = \
          np.dot(X[:self.n_informative], self.redundant_comatrix)

    if self.n_noisy > 0:
      n = self.n_informative + self.n_redundant
      X[n:n + self.n_noisy] = np.dot(X[:self.n_informative], self.noisy_redundant_comatrix)
      X[n:n + self.n_noisy] += np.random.normal(0, 0.05, self.n_noisy)

    if self.n_repeated > 0:
      n = self.n_informative + self.n_redundant + self.n_noisy
      X[n: n + self.n_repeated] = X[self.indices_copy]

    if self.n_useless > 0:
      X[-self.n_useless:] = np.random.normal(0, 1, self.n_useless)

    return X


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


class ProcGenGame(Game):

  def __init__(self, game_name, render=False):
    if '.' in game_name:
      game_name, infos = game_name.split('.')
      self.num_levels, self.start_level = map(int, infos.split('-'))
    else:
      self.num_levels, self.start_level = 0, 0

    self.render = render if render else None

    super().__init__(game_name)
    self.name = f"{game_name}.{self.num_levels}-{self.start_level}"

    self.env_name = f'procgen:{game_name.lower()}-v0'
    env = gym.make(self.env_name)
    self.num_actions = env.action_space.n

  def create(self):
    env = gym.make(self.env_name, distribution_mode='easy',
                   start_level=self.start_level, num_levels=self.num_levels,
                   render=self.render)
    env.name = self.name
    return env


class GymGame(Game):

  def __init__(self, game_name, render=False):
    if '.' in game_name:
      game_name, infos = game_name.split('.')
      self.level = int(infos)
    else:
      self.level = 1

    self.render = render

    game_name = game_name[4:]
    super().__init__(game_name)
    self.name = f"{game_name}-{self.level}"

    self.env_name = f'{game_name}-v1'

    env = gym.make(self.env_name)
    env = GymPreprocessing(env, self.name, self.level)
    self.num_actions = env.action_space.n

  def create(self):
    env = gym.make(self.env_name)
    if isinstance(env, TimeLimit):
      env = env.env
    env = GymPreprocessing(env, self.name, self.level, render=self.render)
    return env


class AtariGame(Game):

  def __init__(self, game_name, sticky_actions):
    super().__init__(game_name)

    self.version = 'v0' if sticky_actions else 'v4'
    self.env_name = f'{self.name}NoFrameskip-{self.version}'

    env = gym.make(self.env_name)
    self.num_actions = env.action_space.n

  def create(self):
    env = ModifiedAtariPreprocessing(gym.make(self.env_name).env, self.variant)
    env.name = self.name
    return env
