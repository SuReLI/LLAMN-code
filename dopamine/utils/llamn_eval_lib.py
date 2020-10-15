
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from dopamine.agents.llamn_network import expert_rainbow_agent, llamn_agent
from dopamine.discrete_domains.llamn_run_experiment import LLAMNRunner
from dopamine.discrete_domains.llamn_run_experiment import ExpertRunner
from dopamine.utils.example_viz_lib import MyDQNAgent, MyRainbowAgent
import tensorflow as tf


class MyExpertAgent(expert_rainbow_agent.ExpertAgent):
  """Sample Expert agent to visualize Q-values and rewards."""

  def _load_llamn(self):
    # Don't load llamn because the weights are saved in the checkpoint and will be loaded
    # in reload_checkpoint
    pass

  def reload_checkpoint(self, checkpoint_path):
    ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_path)
    ckpt_path = ckpt.model_checkpoint_path

    MyRainbowAgent.reload_checkpoint(self, ckpt_path)


class MyLLAMNAgent(llamn_agent.AMNAgent):
  """Sample LLAMN agent to visualize Q-values and rewards."""

  def __init__(self, sess,
               max_num_actions,
               expert_num_actions,
               expert_paths,
               llamn_path,
               summary_writer=None):

    super().__init__(sess, max_num_actions, expert_num_actions,
                     [], [], eval_mode=True, summary_writer=summary_writer)
    self.q_values_list = [[[] for _ in range(expert_num_actions[i])]
                          for i in range(self.nb_experts)]
    self.reward_list = [[] for _ in range(self.nb_experts)]

  def load_networks(self):
    pass

  def reload_checkpoint(self, checkpoint_path):
    ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_path)
    ckpt_path = ckpt.model_checkpoint_path

    MyDQNAgent.reload_checkpoint(self, ckpt_path)


class MyExpertRunner(ExpertRunner):

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    self._agent.reload_checkpoint(self._checkpoint_dir)
    self._start_iteration = 0

  def _run_one_step(self, action):
    outputs = super()._run_one_step(action)
    self._environment.render('human')
    return outputs

  def evaluate(self, num_eps):
    if not re.search(self.filter, self._base_dir, re.I) or \
        re.search(self.exclude, self._base_dir, re.I):  # noqa: E125
      return

    self._agent.eval_mode = True
    total_steps = 0
    total_reward = 0
    for _ in range(num_eps):
      steps, reward = self._run_one_episode()
      total_steps += steps
      total_reward += reward

    self._environment.close()
    game_name = self._environment.environment.game
    print("\tMean reward on", game_name, "for", num_eps, "episodes:", total_reward / num_eps)
    print("\tMean number of steps:", total_steps / num_eps)


class MyLLAMNRunner(LLAMNRunner):

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    self._agent.reload_checkpoint(self._checkpoint_dir)
    self._start_iteration = 0

  def _run_one_step(self, action):
    outputs = super()._run_one_step(action)
    self._environment.render('human')
    return outputs

  def evaluate(self, num_eps):
    for self._game_index in range(self._nb_envs):
      game_name = self._names[self._game_index]
      name = os.path.join(self._base_dir, game_name)
      if not re.search(self.filter, name, re.I) or \
          re.search(self.exclude, name, re.I):  # noqa: E125
        return

      self._agent.eval_mode = True
      total_steps = 0
      total_reward = 0
      for _ in range(num_eps):
        steps, reward = self._run_one_episode()
        total_steps += steps
        total_reward += reward

      self._environment.close()
      game_name = self._environment.environment.game
      print("\tMean reward on", game_name, "for", num_eps, "episodes:", total_reward / num_eps)
      print("\tMean number of steps:", total_steps / num_eps)


def create_expert(sess, environment, llamn_path, name, summary_writer=None):
  return MyExpertAgent(sess, num_actions=environment.action_space.n,
                       llamn_path=llamn_path, name=name)


def run(phase, nb_day, games, nb_actions, name_filter, name_exclude, num_eps, delay, root_dir):
  """Main entrypoint for running and generating visualizations"""

  phase = phase + '_' + str(nb_day)
  phase_dir = os.path.join(root_dir, phase)

  if phase.startswith('day'):
    for game in games:
      print('\033[33mDay', nb_day, '\033[0m')
      # llamn_path must be non-False if it's not the first day, but don't need to be
      # exact because we load from a checkpoint, not from a previous llamn network
      runner = MyExpertRunner(phase_dir, game, create_expert, (nb_day > 0))
      runner.filter = name_filter
      runner.exclude = name_exclude
      runner.evaluate(num_eps)

  else:
    print('\033[33mNight', nb_day, '\033[0m')
    runner = MyLLAMNRunner(phase_dir, nb_actions, games, [], MyLLAMNAgent)
    runner.filter = name_filter
    runner.exclude = name_exclude
    runner.evaluate(num_eps)
