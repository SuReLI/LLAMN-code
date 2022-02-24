
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import multiprocessing

import tensorflow as tf
import numpy as np

from dopamine.discrete_domains import checkpointer
from dopamine.agents.llamn_network.expert_rainbow_agent import ExpertAgent
from dopamine.agents.llamn_network.llamn_agent import AMNAgent
from dopamine.discrete_domains.llamn_run_experiment import LLAMNRunner
from dopamine.discrete_domains.llamn_run_experiment import ExpertRunner
from dopamine.utils.example_viz_lib import MyDQNAgent, MyRainbowAgent
from dopamine.utils.eval_lib import SaliencyAgent, EvalRunner


NB_STATES = 70
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
    state_shape = (NB_STATES_2//2, *self.observation_shape, self.stack_size)
    self.all_states_ph = tf.compat.v1.placeholder(self.observation_dtype, state_shape)
    self.all_outputs = self.online_convnet(self.all_states_ph)
    self.all_q_values = self.all_outputs.q_values
    self.all_q_argmax = tf.argmax(self.all_q_values, axis=1)


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
    state_shape = (NB_STATES_2//2, *self.observation_shape, self.stack_size)
    self.all_states_ph = tf.compat.v1.placeholder(self.observation_dtype, state_shape)
    self.all_outputs = self.convnet(self.all_states_ph)
    self.all_q_values_list = []
    self.all_q_argmax_list = []

    for i in range(self.nb_experts):
      expert_mask = [n_action < self.expert_num_actions[i]
                     for n_action in range(self.llamn_num_actions)]

      partial_q_values = tf.boolean_mask(self.all_outputs.q_values, expert_mask, axis=1)
      self.all_q_values_list.append(partial_q_values)
      q_argmax = tf.argmax(partial_q_values, axis=1)
      self.all_q_argmax_list.append(q_argmax)

  @property
  def all_q_argmax(self):
    return self.all_q_argmax_list[self.ind_expert]

  @property
  def all_q_values(self):
    return self.all_q_values_list[self.ind_expert]


class MyExpertRunner(EvalRunner, ExpertRunner):

  def evaluate(self, num_eps, disp=True):
    self._agent.eval_mode = True

    game_name = self._environment.name.capitalize()
    print('  \033[34m', game_name, '\033[0m', sep='')

    total_steps = 0
    total_reward = 0
    for _ in range(num_eps):
      steps, reward = self._run_one_episode()
      total_steps += steps
      total_reward += reward
      if disp:
        print("    Reward:", reward)

    self._environment.close()
    if disp:
      print("    ----------------")
      print("    Mean reward on", game_name, "for", num_eps, "episodes:", total_reward / num_eps)
      print("    Mean number of steps:", total_steps / num_eps)
    else:
      print(f"{total_reward / num_eps:.02f}")


class MyLLAMNRunner(EvalRunner, LLAMNRunner):

  def evaluate_one_agent(self, agent_index, num_eps, disp=True):
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
      if disp is True:
        print("    Reward:", reward)

    self._environment.close()
    game_name = self._environment.name.capitalize()
    if disp is True:
      print("    ----------------")
      print("    Mean reward on", game_name, "for", num_eps, "episodes:", total_reward / num_eps)
      print("    Mean number of steps:", total_steps / num_eps)
    elif disp == "return":
      return total_reward / num_eps
    else:
      print(f"{total_reward / num_eps:.02f}")


def create_expert(sess, environment, llamn_path, name, summary_writer=None):
  return MyExpertAgent(sess, num_actions=environment.action_space.n,
                       llamn_path=llamn_path, name=name)


def should_evaluate(phase, game, name_filter, name_exclude):
  phase_game = os.path.join(phase, game.name)
  return re.search(name_filter, phase_game, re.I) and not re.search(name_exclude, phase_game, re.I)


class MainEvalRunner:

  def __init__(self, nb_actions, name_filter, name_exclude, num_eps, delay, root_dir):
    self.nb_actions = nb_actions
    self.name_filter = name_filter
    self.name_exclude = name_exclude
    self.num_eps = num_eps
    self.delay = delay
    self.root_dir = root_dir

  def save_features(self, phase_dir, runner, game):
    print('  \033[34m', game.name, '\033[0m', sep='')
    result_dir = os.path.join('data/runs', *phase_dir.split('/')[1:], game.name)
    os.makedirs(result_dir, exist_ok=True)
    runner._agent._build_features_op()
    all_states = np.load(os.path.join(f'data/all_states/states_{NB_STATES}', game.name+'.npy'))

    if game.name.startswith('Pendulum'):
      all_states = all_states[:-1, ..., np.newaxis]
      features = runner._sess.run(runner._agent.all_outputs.features,
                                  feed_dict={runner._agent.all_states_ph: all_states})
      np.save(os.path.join(result_dir, 'features.npy'), features)

      actions = runner._sess.run(runner._agent.all_q_argmax,
                                 feed_dict={runner._agent.all_states_ph: all_states})
      np.save(os.path.join(result_dir, 'actions.npy'), actions)

      q_values = runner._sess.run(runner._agent.all_q_values,
                                  feed_dict={runner._agent.all_states_ph: all_states})
      np.save(os.path.join(result_dir, 'qvalues.npy'), q_values)

    else:
      features = np.zeros((NB_STATES_2, 64), np.float32)
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

      q_values = np.zeros((NB_STATES_2, game.num_actions), np.float32)
      q_values[:NB_STATES_2//2] = runner._sess.run(runner._agent.all_q_values,
                                                   feed_dict={runner._agent.all_states_ph: all_states[:NB_STATES_2//2]})
      q_values[NB_STATES_2//2:] = runner._sess.run(runner._agent.all_q_values,
                                                   feed_dict={runner._agent.all_states_ph: all_states[NB_STATES_2//2:]})
      np.save(os.path.join(result_dir, f'qvalues_{int(NB_STATES_2**0.5)}.npy'), q_values)

  def save_saliency(self, phase_dir, runner, game):
      print('  \033[34m', game.name, '\033[0m', sep='')
      result_dir = os.path.join('data/runs', *phase_dir.split('/')[1:], game.name)
      os.makedirs(result_dir, exist_ok=True)
      all_states = np.load(os.path.join(f'data/all_states/states_{NB_STATES}', game.name+'.npy'))
      saliency_sum_file = os.path.join(result_dir, 'sum_saliency_activations.npy')
      saliency_mean_file = os.path.join(result_dir, 'mean_saliency_activations.npy')

      if game.name.startswith("Pendulum"):
        saliencies = runner._agent.compute_saliencies(all_states[..., np.newaxis])
        saliency_sums = saliencies.sum(axis=1)
        saliency_means = saliencies.mean(axis=0)

      else:
        size = 1000
        saliency_sums = np.empty(size)
        saliency_means = np.empty((size, *all_states.shape[1:-1]))
        for i, state in enumerate(all_states[:size, ...]):
          saliency = runner._agent.compute_saliency(state[np.newaxis])
          saliency_sums[i] = saliency.sum()
          saliency_means[i] = saliency
        saliency_means = saliency_means.mean(axis=0)

      np.save(saliency_sum_file, saliency_sums)
      np.save(saliency_mean_file, saliency_means)

  def save_saliency_ep(self, phase, runner, game):
    saliency_dir = os.path.join(self.root_dir, 'agent_viz', phase, f"expert_{game.name}")
    os.makedirs(saliency_dir, exist_ok=True)
    runner.saliency_path = os.path.join(saliency_dir, "saliency")

  def eval_expert(self, phase, phase_dir, game, nb_day, mode=False, heatmap=False, disp=True):
    # llamn_path must be non-False if it's not the first day, but don't need to be
    # exact because we load from a checkpoint, not from a previous llamn network
    runner = MyExpertRunner(phase_dir, game, create_expert, (nb_day > 0))
    runner.delay = self.delay
    runner._environment.eval_mode = True

    if mode == 'save_state':
      if not hasattr(self, 'saved_games'):
        self.saved_games = []
      if game.name in self.saved_games:
        return
      self.saved_games.append(game.name)
      print(f"  \033[34mSaving states from {game.name}\033[0m", sep='')
      
      # Pendulum grid state
      if game.name.startswith('Pendulum'):
        theta = np.linspace(-np.pi, np.pi, 21)
        theta_dot = np.linspace(-8, 8, 51)
        theta = theta.repeat(51)[..., np.newaxis]
        theta_dot = np.tile(theta_dot, 21)[..., np.newaxis]
        states = np.hstack((np.cos(theta), np.sin(theta), theta_dot))
        env = runner._environment
        states = (states - env.min_observation) / (env.max_observation - env.min_observation)
        states = 2 * states - 1

        X = np.zeros((states.shape[0], env.n_features))
        X[:, :env.n_informative] = states
        X[:, env.n_informative: env.n_informative + env.n_redundant] = \
            np.dot(X[:, :env.n_informative], env.redundant_comatrix)

        n = env.n_informative + env.n_redundant
        X[:, n: n+env.n_noisy] = np.dot(X[:, :env.n_informative], env.noisy_redundant_comatrix)
        X[:, n: n+env.n_noisy] += np.random.normal(0, 0.05, env.n_noisy)

        n = env.n_informative + env.n_redundant + env.n_noisy
        X[:, n: n + env.n_repeated] = X[:, env.indices_copy]

        X[:, -env.n_useless:] = np.random.normal(0, 1, env.n_useless)
        sample_states = X

      else:
        runner._agent._replay = MyRainbowAgent._build_replay_buffer(runner._agent, False)
        checkpoint_nb = checkpointer.get_latest_checkpoint_number(runner._checkpoint_dir)
        runner._agent._replay.load(runner._checkpoint_dir, checkpoint_nb)
        all_states = runner._sess.run(runner._agent._replay.states)

        random_idx = np.random.choice(all_states.shape[0], NB_STATES_2, replace=False)
        sample_states = all_states[random_idx]

      os.makedirs(f'data/all_states/states_{NB_STATES}', exist_ok=True)
      state_file = os.path.join(f'data/all_states/states_{NB_STATES}', game.name+'.v2.npy')
      np.save(state_file, sample_states)
      return

    elif mode == 'features':
      self.save_features(phase_dir, runner, game)
      return

    elif mode == 'saliency':
      self.save_saliency(phase_dir, runner, game)
      return

    elif mode == 'saliency_ep':
      self.save_saliency_ep(phase, runner, game)

    if heatmap:
      runner.features_heatmap = True

    runner.evaluate(self.num_eps, disp)

  def eval_llamn(self, phase, phase_dir, games, agent_ind, mode=False, heatmap=False, disp=True):
    runner = MyLLAMNRunner(phase_dir, self.nb_actions, games, [], MyLLAMNAgent)
    runner.delay = self.delay
    game = games[agent_ind]
    runner._agent.ind_expert = agent_ind

    if mode == 'save_state':
      return

    elif mode == 'features':
      self.save_features(phase_dir, runner, game)
      return

    elif mode == 'saliency':
      self.save_saliency(phase_dir, runner, game)
      return

    elif mode == 'saliency_ep':
      self.save_saliency_ep(phase, runner, game)

    if heatmap:
      runner.features_heatmap = True

    runner.evaluate_one_agent(agent_ind, self.num_eps, disp)

  def run(self, all_games, phase, nb_day, mode=False, heatmap=False, disp=True):
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
        self.eval_expert(phase, phase_dir, game, nb_day, mode, heatmap, disp)

    else:
      if mode == 'save_state':
        return
      print('\033[33mNight', nb_day, '\033[0m')
      for agent_index in range(len(games)):
        self.eval_llamn(phase, phase_dir, games, agent_index, mode, heatmap, disp)

  def evaluate_proc(self, agent_index):
    runner = MyLLAMNRunner(self.phase_dir, self.nb_actions, self.games, [], MyLLAMNAgent)
    runner.delay = self.delay
    runner._agent.ind_expert = agent_index
    return runner.evaluate_one_agent(agent_index, self.num_eps, "return")  

  def run_robust(self, all_games):
    phase = "night_0"
    self.phase_dir = os.path.join(self.root_dir, phase)
    self.games = list(filter(lambda game: should_evaluate(phase, game, self.name_filter, self.name_exclude),
                             all_games))
    if not self.games:
      return

    with multiprocessing.Pool() as p:
      means = p.map(self.evaluate_proc, range(len(self.games)))

    mean_path = os.path.join(os.path.dirname(self.phase_dir), 'mean.npy')
    np.save(mean_path, np.array(means))
