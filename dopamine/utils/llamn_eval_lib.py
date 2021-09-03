
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import time

import tensorflow as tf
import matplotlib.pyplot as plt

from dopamine.agents.llamn_network.expert_rainbow_agent import ExpertAgent
from dopamine.agents.llamn_network.llamn_agent import AMNAgent
from dopamine.discrete_domains.llamn_run_experiment import LLAMNRunner
from dopamine.discrete_domains.llamn_run_experiment import ExpertRunner
from dopamine.utils.example_viz_lib import MyDQNAgent, MyRainbowAgent
from dopamine.utils.saliency_lib import SaliencyAgent


class MyExpertAgent(ExpertAgent, SaliencyAgent):
  """Sample Expert agent to visualize Q-values and rewards."""

  def _build_replay_buffer(self, *args, **kwargs):
    pass

  def _build_networks(self):
    self.online_convnet = self._create_network(name='online')
    self._net_outputs = self.online_convnet(self.state_ph)
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]

  def _build_train_op(self):
    pass

  def _build_sync_op(self):
    pass

  def _load_llamn(self):
    # Don't load llamn because the weights are saved in the checkpoint and will be loaded
    # in reload_checkpoint
    pass

  def reload_checkpoint(self, checkpoint_path):
    ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_path)
    ckpt_path = ckpt.model_checkpoint_path

    MyRainbowAgent.reload_checkpoint(self, ckpt_path)


class MyLLAMNAgent(AMNAgent, SaliencyAgent):
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

  def _build_replay_buffers(self):
    pass

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
    if hasattr(self, 'saliency_path'):
      self.display_saliency()
    else:
      time.sleep(self.delay / 1000)
    return outputs

  def display_saliency(self):
    if not hasattr(self, 'frame_nb'):
      self.frame_nb = 0
    saliency_map = self._agent.compute_saliency(self._agent.state)

    plt.imshow(self._agent.state[0, :, :, 3], cmap='gray')
    plt.imshow(saliency_map, cmap='Reds', alpha=0.5)
    image_path = f"{self.saliency_path}_{self.frame_nb:02}.svg"
    plt.savefig(image_path)

    self.frame_nb += 1

  def evaluate(self, num_eps):
    self._agent.eval_mode = True

    game_name = self._environment.environment.game.capitalize()
    print('  \033[34m', game_name, '\033[0m', sep='')

    total_steps = 0
    total_reward = 0
    for _ in range(num_eps):
      steps, reward = self._run_one_episode()
      total_steps += steps
      total_reward += reward
      print("    Reward:", reward)

    self._environment.close()
    print("    ----------------")
    print("    Mean reward on", game_name, "for", num_eps, "episodes:", total_reward / num_eps)
    print("    Mean number of steps:", total_steps / num_eps)


class MyLLAMNRunner(LLAMNRunner):

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    self._agent.reload_checkpoint(self._checkpoint_dir)
    self._start_iteration = 0

  def _run_one_step(self, action):
    outputs = super()._run_one_step(action)
    self._environment.render('human')
    if hasattr(self, 'saliency_path'):
      self.display_saliency()
    else:
      time.sleep(self.delay / 1000)
    return outputs

  def display_saliency(self):
    if not hasattr(self, 'frame_nb'):
      self.frame_nb = [0 for _ in range(self._nb_envs)]
    saliency_map = self._agent.compute_saliency(self._agent.state)

    plt.imshow(self._agent.state[0, :, :, 3], cmap='gray')
    plt.imshow(saliency_map, cmap='Reds', alpha=0.5)
    game_name = self._names[self._game_index]
    image_path = f"{self.saliency_path}_{game_name}_{self.frame_nb[self._game_index]:02}.svg"
    plt.savefig(image_path)

    self.frame_nb[self._game_index] += 1

  def evaluate(self, num_eps):
    for self._game_index in range(self._nb_envs):
      game_name = self._names[self._game_index]
      print('  \033[34m', game_name, '\033[0m', sep='')

      self._agent.eval_mode = True
      total_steps = 0
      total_reward = 0
      for _ in range(num_eps):
        steps, reward = self._run_one_episode()
        total_steps += steps
        total_reward += reward
        print("    Reward:", reward)

      self._environment.close()
      game_name = self._environment.environment.game.capitalize()
      print("    ----------------")
      print("    Mean reward on", game_name, "for", num_eps, "episodes:", total_reward / num_eps)
      print("    Mean number of steps:", total_steps / num_eps)


def create_expert(sess, environment, llamn_path, name, summary_writer=None):
  return MyExpertAgent(sess, num_actions=environment.action_space.n,
                       llamn_path=llamn_path, name=name)


def should_evaluate(phase, game, name_filter, name_exclude):
  phase_game = os.path.join(phase, str(game))
  return re.search(name_filter, phase_game, re.I) and not re.search(name_exclude, phase_game, re.I)


def run(phase, nb_day, games, nb_actions, name_filter, name_exclude, num_eps, delay, root_dir, saliency):
  """Main entrypoint for running and generating visualizations"""

  phase = phase + '_' + str(nb_day)
  phase_dir = os.path.join(root_dir, phase)

  games = list(filter(lambda game: should_evaluate(phase, game, name_filter, name_exclude), games))
  if not games:
    return

  if phase.startswith('day'):
    print('\033[33mDay', nb_day, '\033[0m')
    for game in games:
      # llamn_path must be non-False if it's not the first day, but don't need to be
      # exact because we load from a checkpoint, not from a previous llamn network
      runner = MyExpertRunner(phase_dir, game, create_expert, (nb_day > 0))
      runner.delay = delay

      if saliency:
        saliency_dir = os.path.join(root_dir, 'agent_viz', phase, f"expert_{game.name}")
        os.makedirs(saliency_dir, exist_ok=True)
        runner.saliency_path = os.path.join(saliency_dir, "saliency")
      runner.evaluate(num_eps)

  else:
    print('\033[33mNight', nb_day, '\033[0m')
    runner = MyLLAMNRunner(phase_dir, nb_actions, games, [], MyLLAMNAgent)
    runner.delay = delay

    if saliency:
      saliency_dir = os.path.join(root_dir, 'agent_viz', phase)
      os.makedirs(saliency_dir, exist_ok=True)
      runner.saliency_path = os.path.join(saliency_dir, "saliency")
    runner.evaluate(num_eps)
