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
"""Library used by example_viz.py to generate visualizations.

This file illustrates the following:
  - How to subclass an existing agent to add visualization functionality.
    - For DQN we visualize the cumulative rewards and the Q-values for each
      action (MyDQNAgent).
    - For Rainbow we visualize the cumulative rewards and the Q-value
      distributions for each action (MyRainbowAgent).
  - How to subclass Runner to run in eval mode, lay out the different subplots,
    generate the visualizations, and compile them into a video (MyRunner).
  - The function `run()` is the main entrypoint for running everything.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from dopamine.agents.llamn_network import expert_rainbow_agent, llamn_agent
from dopamine.discrete_domains.llamn_run_experiment import LLAMNRunner
from dopamine.discrete_domains.llamn_run_experiment import ExpertRunner
from dopamine.utils.example_viz_lib import MyDQNAgent, MyRainbowAgent, MyRunner
import numpy as np
import tensorflow as tf


class MyExpertAgent(expert_rainbow_agent.ExpertAgent):
  """Sample Expert agent to visualize Q-values and rewards."""

  def __init__(self, sess, num_actions, llamn_path, name, summary_writer=None):
    super().__init__(sess, num_actions, llamn_path, name, summary_writer=summary_writer)
    self.rewards = []

  def _load_llamn(self):
    # Don't load llamn because the weights are saved in the checkpoint and will be loaded
    # in reload_checkpoint
    pass

  def step(self, reward, observation):
    self.rewards.append(reward)
    return super().step(reward, observation)

  def reload_checkpoint(self, checkpoint_path):
    ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_path)
    ckpt_path = ckpt.model_checkpoint_path

    MyRainbowAgent.reload_checkpoint(self, ckpt_path)

  def get_probabilities(self):
    return self._sess.run(tf.squeeze(self._net_outputs.probabilities),
                          {self.state_ph: self.state})

  def get_rewards(self):
    return [np.cumsum(self.rewards)]


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

  @property
  def rewards(self):
    return self.reward_list[self.ind_expert]

  @property
  def q_values(self):
    return self.q_values_list[self.ind_expert]

  def step(self, reward, observation):
    self.rewards.append(reward)
    return super().step(reward, observation)

  def _select_action(self):
    action = super()._select_action()
    q_vals = self._sess.run(self._net_q_output[self.ind_expert],
                            {self.state_ph: self.state})[0]
    for i in range(len(q_vals)):
      self.q_values[i].append(q_vals[i])
    return action

  def reload_checkpoint(self, checkpoint_path):
    ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_path)
    ckpt_path = ckpt.model_checkpoint_path

    MyDQNAgent.reload_checkpoint(self, ckpt_path)

  def get_q_values(self):
    return self.q_values

  def get_rewards(self):
    return [np.cumsum(self.rewards)]


class MyExpertRunner(ExpertRunner):

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    self._agent.reload_checkpoint(self._checkpoint_dir)
    self._start_iteration = 0

  def visualize(self, record_path, num_global_steps=500):
    MyRunner.visualize(self, record_path, num_global_steps)


class MyLLAMNRunner(LLAMNRunner):

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    self._agent.reload_checkpoint(self._checkpoint_dir)
    self._start_iteration = 0

  def visualize(self, record_path, num_global_steps=500):

    for self._game_index in range(self._nb_envs):
      game_name = self._names[self._game_index]
      game_record_path = os.path.join(record_path, game_name, "images")
      MyRunner.visualize(self, game_record_path, num_global_steps)


def create_expert(sess, environment, llamn_path, name, summary_writer=None):
  return MyExpertAgent(sess, num_actions=environment.action_space.n,
                       llamn_path=llamn_path, name=name)


def run(phase, nb_day, games, nb_actions, num_steps, root_dir):
  """Main entrypoint for running and generating visualizations"""

  phase = phase + '_' + str(nb_day)
  phase_dir = os.path.join(root_dir, phase)

  if phase.startswith('day'):
    for game in games:
      # llamn_path must be non-False if it's not the first day, but don't need to be
      # exact because we load from a checkpoint, not from a previous llamn network
      runner = MyExpertRunner(phase_dir, game, create_expert, (nb_day > 0))

      # Image dir = results/AMN-2020.../agent_viz/day_1/expert_Pong/images
      image_dir = os.path.join(root_dir, 'agent_viz', phase, f"expert_{game.name}", "images")
      runner.visualize(image_dir, num_global_steps=num_steps)

  else:
    base_dir = os.path.join(root_dir, 'agent_viz', phase)

    runner = MyLLAMNRunner(phase_dir, nb_actions, games, [], MyLLAMNAgent)
    runner.visualize(base_dir, num_global_steps=num_steps)
