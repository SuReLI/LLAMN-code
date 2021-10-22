
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import time

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from dopamine.discrete_domains import checkpointer
from dopamine.agents.llamn_network.expert_rainbow_agent import ExpertAgent
from dopamine.agents.llamn_network.llamn_agent import AMNAgent
from dopamine.discrete_domains.llamn_run_experiment import LLAMNRunner
from dopamine.discrete_domains.llamn_run_experiment import ExpertRunner
from dopamine.utils.example_viz_lib import MyDQNAgent, MyRainbowAgent
from dopamine.utils.eval_lib import SaliencyAgent, EvalRunner


NB_STATES = 100
assert (NB_STATES % 2 == 0), "NB_STATES must be even"

NB_STATES_2 = NB_STATES**2


class MyExpertAgent(SaliencyAgent, ExpertAgent):
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

  def _build_features_op(self):
    state_shape = (NB_STATES_2//2, *self.observation_shape, 4)
    self.all_states_ph = tf.compat.v1.placeholder(self.observation_dtype, state_shape)
    self.all_outputs = self.online_convnet(self.all_states_ph)
    self.all_q_argmax = tf.argmax(self.all_outputs.q_values, axis=1)


class MyLLAMNAgent(SaliencyAgent, AMNAgent):
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

  def _build_features_op(self):
    state_shape = (NB_STATES_2//2, *self.observation_shape, 4)
    self.all_states_ph = tf.compat.v1.placeholder(self.observation_dtype, state_shape)
    self.all_outputs = self.convnet(self.all_states_ph)
    self.all_q_argmax = []

    for i in range(self.nb_experts):
      expert_mask = [n_action < self.expert_num_actions[i]
                     for n_action in range(self.llamn_num_actions)]

      partial_q_values = tf.boolean_mask(self.all_outputs.q_values, expert_mask, axis=1)
      q_argmax = tf.argmax(partial_q_values, axis=1)
      self.all_q_argmax.append(q_argmax)


class MyExpertRunner(EvalRunner, ExpertRunner):

  def evaluate(self, num_eps):
    self._agent.eval_mode = True

    game_name = self._environment.environment._game.capitalize()
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


class MyLLAMNRunner(EvalRunner, LLAMNRunner):

  def evaluate_one_agent(self, agent_index, num_eps):
    self._game_index = agent_index
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
    game_name = self._environment.environment._game.capitalize()
    print("    ----------------")
    print("    Mean reward on", game_name, "for", num_eps, "episodes:", total_reward / num_eps)
    print("    Mean number of steps:", total_steps / num_eps)


def create_expert(sess, environment, llamn_path, name, summary_writer=None):
  return MyExpertAgent(sess, num_actions=environment.action_space.n,
                       llamn_path=llamn_path, name=name)


def should_evaluate(phase, game, name_filter, name_exclude):
  phase_game = os.path.join(phase, str(game))
  return re.search(name_filter, phase_game, re.I) and not re.search(name_exclude, phase_game, re.I)


class EvalRunner:

  def __init__(self, nb_actions, name_filter, name_exclude, num_eps, delay, root_dir):
    self.nb_actions = nb_actions
    self.name_filter = name_filter
    self.name_exclude = name_exclude
    self.num_eps = num_eps
    self.delay = delay
    self.root_dir = root_dir

  def eval_expert(self, phase, phase_dir, game, nb_day, mode=False, heatmap=False):
    # llamn_path must be non-False if it's not the first day, but don't need to be
    # exact because we load from a checkpoint, not from a previous llamn network
    runner = MyExpertRunner(phase_dir, game, create_expert, (nb_day > 0))
    runner.delay = self.delay

    if mode == 'save_state':
      if not hasattr(self, 'saved_games'):
        self.saved_games = []
      if game.name in self.saved_games:
        return
      self.saved_games.append(game.name)
      print(f"  \033[34mSaving states from {game.name}\033[0m", sep='')
      runner._agent._replay = MyRainbowAgent._build_replay_buffer(runner._agent, False)
      checkpoint_nb = checkpointer.get_latest_checkpoint_number(runner._checkpoint_dir)
      runner._agent._replay.load(runner._checkpoint_dir, checkpoint_nb)
      all_states = runner._sess.run(runner._agent._replay.states)
      random_idx = np.random.choice(all_states.shape[0], NB_STATES_2, replace=False)
      sample_states = all_states[random_idx]
      os.makedirs(f'data/all_states/states_{NB_STATES}', exist_ok=True)
      state_file = os.path.join(f'data/all_states/states_{NB_STATES}', game.name+'.npy')
      np.save(state_file, sample_states)
      return

    elif mode == 'features':
      print('  \033[34m', game.name, '\033[0m', sep='')
      result_dir = os.path.join('data', *phase_dir.split('/')[1:], game.name)
      os.makedirs(result_dir, exist_ok=True)
      runner._agent._build_features_op()
      all_states = np.load(os.path.join(f'data/all_states/states_{NB_STATES}', game.name+'.npy'))

      features = np.zeros((NB_STATES_2, 512), np.float32)
      features[:NB_STATES_2//2] = runner._sess.run(runner._agent.all_outputs.features,
                                                   feed_dict={runner._agent.all_states_ph: all_states[:NB_STATES_2//2]})
      features[NB_STATES_2//2:] = runner._sess.run(runner._agent.all_outputs.features,
                                                   feed_dict={runner._agent.all_states_ph: all_states[NB_STATES_2//2:]})
      np.save(os.path.join(result_dir, f'features_{int(NB_STATES_2**0.5)}.npy'), features)

      actions = np.zeros(NB_STATES_2, np.float32)
      actions[:NB_STATES_2//2] = runner._sess.run(runner._agent.all_q_argmax,
                                                  feed_dict={runner._agent.all_states_ph: all_states[:NB_STATES_2//2]})
      actions[NB_STATES_2//2:] = runner._sess.run(runner._agent.all_q_argmax,
                                                  feed_dict={runner._agent.all_states_ph: all_states[NB_STATES_2//2:]})
      np.save(os.path.join(result_dir, f'actions_{int(NB_STATES_2**0.5)}.npy'), actions)
      return

    elif mode == 'saliency':
      saliency_dir = os.path.join(self.root_dir, 'agent_viz', phase, f"expert_{game.name}")
      os.makedirs(saliency_dir, exist_ok=True)
      runner.saliency_path = os.path.join(saliency_dir, "saliency")

    if heatmap:
      runner.features_heatmap = True

    runner.evaluate(self.num_eps)

  def eval_llamn(self, phase, phase_dir, games, agent_ind, mode=False, heatmap=False):
    runner = MyLLAMNRunner(phase_dir, self.nb_actions, games, [], MyLLAMNAgent)
    runner.delay = self.delay

    if mode == 'save_state':
      return

    elif mode == 'features':
      game = games[agent_ind]
      print('  \033[34m', game.name, '\033[0m', sep='')
      result_dir = os.path.join('data', *phase_dir.split('/')[1:], game.name)
      os.makedirs(result_dir, exist_ok=True)
      runner._agent._build_features_op()
      all_states = np.load(os.path.join(f'data/all_states/states_{NB_STATES}', game.name+'.npy'))

      features = np.zeros((NB_STATES_2, 512), np.float32)
      features[:NB_STATES_2//2] = runner._sess.run(runner._agent.all_outputs.features,
                                                   feed_dict={runner._agent.all_states_ph: all_states[:NB_STATES_2//2]})
      features[NB_STATES_2//2:] = runner._sess.run(runner._agent.all_outputs.features,
                                                   feed_dict={runner._agent.all_states_ph: all_states[NB_STATES_2//2:]})
      np.save(os.path.join(result_dir, f'features_{int(NB_STATES_2**0.5)}.npy'), features)

      actions = np.zeros(NB_STATES_2, np.float32)
      actions[:NB_STATES_2//2] = runner._sess.run(runner._agent.all_q_argmax[agent_ind],
                                                  feed_dict={runner._agent.all_states_ph: all_states[:NB_STATES_2//2]})
      actions[NB_STATES_2//2:] = runner._sess.run(runner._agent.all_q_argmax[agent_ind],
                                                  feed_dict={runner._agent.all_states_ph: all_states[NB_STATES_2//2:]})
      np.save(os.path.join(result_dir, f'actions_{int(NB_STATES_2**0.5)}.npy'), actions)
      return

    elif mode == 'saliency':
      game = games[agent_ind]
      saliency_dir = os.path.join(self.root_dir, 'agent_viz', phase, f"expert_{game.name}")
      os.makedirs(saliency_dir, exist_ok=True)
      runner.saliency_path = os.path.join(saliency_dir, "saliency")

    if heatmap:
      runner.features_heatmap = True

    runner.evaluate_one_agent(agent_ind, self.num_eps)

  def run(self, all_games, phase, nb_day, mode=False, heatmap=False):
    """Main entrypoint for running and generating visualizations"""

    phase = phase + '_' + str(nb_day)
    phase_dir = os.path.join(self.root_dir, phase)

    games = list(filter(lambda game: should_evaluate(phase, game, self.name_filter, self.name_exclude),
                        all_games))
    if not games:
      return

    if phase.startswith('day'):
      print('\033[33mDay', nb_day, '\033[0m')
      for game in games:
        self.eval_expert(phase, phase_dir, game, nb_day, mode, heatmap)

    else:
      print('\033[33mNight', nb_day, '\033[0m')
      for agent_index in range(len(games)):
        self.eval_llamn(phase, phase_dir, games, agent_index, mode, heatmap)
