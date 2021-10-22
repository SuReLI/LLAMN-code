
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from dopamine.agents.rainbow.rainbow_agent import RainbowAgent
from dopamine.discrete_domains.run_experiment import TrainRunner
from dopamine.utils.example_viz_lib import MyRainbowAgent
from dopamine.utils.eval_lib import SaliencyAgent


NB_STATES = 100
assert (NB_STATES % 2 == 0), "NB_STATES must be even"

NB_STATES_2 = NB_STATES**2


class MyTransferAgent(RainbowAgent, SaliencyAgent):
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

  def reload_checkpoint(self, checkpoint_path):
    ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_path)
    ckpt_path = ckpt.model_checkpoint_path

    MyRainbowAgent.reload_checkpoint(self, ckpt_path)

  def _build_features_op(self):
    state_shape = (NB_STATES_2//2, *self.observation_shape, 4)
    self.all_states_ph = tf.compat.v1.placeholder(self.observation_dtype, state_shape)
    self.all_outputs = self.online_convnet(self.all_states_ph)
    self.all_q_argmax = tf.argmax(self.all_outputs.q_values, axis=1)


class MyTrainRunner(TrainRunner):

  def __init__(self, *args, **kwargs):
    tf.compat.v1.reset_default_graph()
    super().__init__(*args, **kwargs)

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    self._agent.reload_checkpoint(self._checkpoint_dir)
    self._start_iteration = 0

  def _run_one_step(self, action):
    outputs = super()._run_one_step(action)
    if hasattr(self, 'saliency_path'):
      self.compute_saliency()
    else:
      self._environment.render('human')
      time.sleep(self.delay / 1000)
    return outputs

  def compute_saliency(self):
    if not hasattr(self, 'frame_nb'):
      self.frame_nb = 0
      self.fig, self.ax = plt.subplots()

    saliency_map = self._agent.compute_saliency(self._agent.state)

    image_path = f"{self.saliency_path}_{self.frame_nb:02}.png"
    self.ax.cla()
    self.ax.imshow(self._agent.state[0, :, :, 3], cmap='gray')
    self.ax.imshow(saliency_map, cmap='Reds', alpha=0.5)
    self.fig.savefig(image_path)

    self.frame_nb += 1

  def evaluate(self, num_eps):
    self._agent.eval_mode = True

    game_name = self._environment.environment._game.capitalize()
    print(f"  \033[34m{game_name}\033[0m", sep='')

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


def create_agent(sess, environment, summary_writer=None):
  return MyTransferAgent(sess, num_actions=environment.action_space.n)


def should_evaluate(game, name_filter, name_exclude):
  return re.search(name_filter, game, re.I) and not re.search(name_exclude, game, re.I)


class EvalRunner:

  def __init__(self, name_filter, name_exclude, num_eps, delay, root_dir):
    self.name_filter = name_filter
    self.name_exclude = name_exclude
    self.num_eps = num_eps
    self.delay = delay
    self.root_dir = root_dir

  def eval_expert(self, base_dir, nb_day, game, mode=False):
    runner = MyTrainRunner(base_dir, create_agent, game.create)
    runner.delay = self.delay

    if mode == 'features':
      print(f"  \033[34m{game.name}\033[0m", sep='')
      result_dir = os.path.join('data', *base_dir.split('/')[1:])
      os.makedirs(result_dir, exist_ok=True)
      runner._agent._build_features_op()
      state_file = os.path.join(f'data/all_states/states_{NB_STATES}', game.name+'.npy')
      all_states = np.load(state_file)

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
      saliency_dir = os.path.join(self.root_dir, 'agent_viz', f'day_{nb_day}', f'{game.name}')
      os.makedirs(saliency_dir, exist_ok=True)
      runner.saliency_path = os.path.join(saliency_dir, 'saliency')

    runner.evaluate(self.num_eps)

  def run(self, all_games, nb_day, mode=False):
    """Main entrypoint for running and generating visualizations"""

    games = list(filter(lambda game: should_evaluate(game.name, self.name_filter, self.name_exclude),
                        all_games))
    if not games:
      return

    print(f"\033[33mDay {nb_day}\033[0m")
    for game in games:
      if nb_day == 0:
        base_dir = os.path.join(self.root_dir, 'day_0')
      else:
        base_dir = os.path.join(self.root_dir, f'day_{nb_day}', game.name)
      self.eval_expert(base_dir, nb_day, game, mode)
