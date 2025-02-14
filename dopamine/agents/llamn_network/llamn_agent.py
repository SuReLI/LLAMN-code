
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random

from absl import logging

from dopamine.agents.dqn import dqn_agent
from dopamine.replay_memory import prioritized_replay_buffer
from dopamine.discrete_domains import atari_lib, llamn_atari_lib
import numpy as np
import tensorflow as tf

import gin.tf


# These are aliases which are used by other classes.
NATURE_DQN_OBSERVATION_SHAPE = atari_lib.NATURE_DQN_OBSERVATION_SHAPE
NATURE_DQN_DTYPE = atari_lib.NATURE_DQN_DTYPE


@gin.configurable
class AMNAgent:
  """An implementation of the LLAMN agent."""

  def __init__(self,
               sess,
               max_num_actions,
               expert_num_actions,
               expert_paths,
               llamn_path,
               sleeping_memory=None,
               feature_weight=0.01,
               ewc_weight=0.1,
               feature_size=512,
               observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=atari_lib.NATURE_DQN_DTYPE,
               stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
               network=llamn_atari_lib.AMNNetwork,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=1000,
               update_period=4,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=25000,
               replay_scheme='prioritized',
               distributional_night=False,
               expert_init_option=1,
               llamn_init_copy=False,
               expert_num_atoms=51,
               expert_vmax=10,
               tf_device='/cpu:*',
               eval_mode=False,
               max_tf_checkpoints_to_keep=4,
               optimizer=tf.compat.v1.train.RMSPropOptimizer(
                   learning_rate=0.00025,
                   decay=0.95,
                   momentum=0.0,
                   epsilon=0.00001,
                   centered=True),
               optimize_loss_sum=False,
               summary_writer=None,
               summary_writing_frequency=500,
               allow_partial_reload=False):

    assert isinstance(observation_shape, tuple)
    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    logging.info('\t tf_device: %s', tf_device)
    logging.info('\t optimizer: %s', optimizer)
    logging.info('\t max_tf_checkpoints_to_keep: %d',
                 max_tf_checkpoints_to_keep)

    # LLAMN Initialization
    self.llamn_num_actions = max_num_actions
    self.expert_num_actions = expert_num_actions
    self.feature_size = feature_size
    self.expert_paths = expert_paths
    self.ind_expert = 0
    self.nb_experts = len(expert_num_actions)
    self.llamn_path = llamn_path
    self.expert_init_option = expert_init_option
    self.llamn_init_copy = llamn_init_copy
    self.feature_weight = feature_weight
    self.ewc_weight = ewc_weight

    # Rainbow Init
    expert_vmax = float(expert_vmax)
    self.observation_shape = tuple(observation_shape)
    self.observation_dtype = observation_dtype
    self.stack_size = stack_size
    self.network = network
    self.gamma = gamma
    self.update_horizon = update_horizon
    self.cumulative_gamma = math.pow(gamma, update_horizon)
    self.min_replay_history = min_replay_history
    self.epsilon_fn = epsilon_fn
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_decay_period = epsilon_decay_period
    self.replay_scheme = replay_scheme
    self.update_period = update_period
    self.buffer_loaded = False

    if llamn_init_copy:
      self.epsilon_fn = dqn_agent.identity_epsilon

    # Expert parameters
    self.expert_num_atoms = expert_num_atoms
    self.expert_support = tf.linspace(-expert_vmax, expert_vmax, expert_num_atoms)

    # AMN parameters
    self.distributional_night = distributional_night
    self.llamn_num_atoms = self.expert_num_atoms if self.distributional_night else None
    self.llamn_support = self.expert_support if self.distributional_night else None

    self.eval_mode = eval_mode
    self.training_steps_list = [0] * self.nb_experts
    self.optimizer = optimizer
    self.optimize_loss_sum = optimize_loss_sum
    self.summary_writer = summary_writer if not self.eval_mode else None
    self.summary_writing_frequency = summary_writing_frequency
    self.allow_partial_reload = allow_partial_reload

    state_shape = (1, *self.observation_shape, stack_size)
    self.states = [np.zeros(state_shape) for i in range(self.nb_experts)]

    with tf.device(tf_device):
      self._build_experts()
      # self._build_prev_llamn()

      # Create a placeholder for the state input to the AMN network.
      # The last axis indicates the number of consecutive frames stacked.
      self.state_ph = tf.compat.v1.placeholder(self.observation_dtype, state_shape,
                                               name='state_ph')
      self._build_replay_buffers()

      self._build_networks()

      self._train_ops = self._build_train_ops()

      if self.summary_writer is not None:
        self._merged_summaries = [tf.compat.v1.summary.merge(s) for s in self.summaries]

    self._sess = sess

    var_map = atari_lib.maybe_transform_variable_names(
        tf.compat.v1.global_variables())
    self._saver = tf.compat.v1.train.Saver(var_list=var_map,
                                           max_to_keep=max_tf_checkpoints_to_keep)

    self._observation = None
    self._last_observation = None

  @property
  def num_actions(self):
    return self.expert_num_actions[self.ind_expert]

  @property
  def state(self):
    return self.states[self.ind_expert]

  @state.setter
  def state(self, value):
    self.states[self.ind_expert] = value

  @property
  def _replay(self):
    return self._replays[self.ind_expert]

  @property
  def q_argmax(self):
    return self._net_q_argmax[self.ind_expert]

  @property
  def _train_op(self):
    if self.optimize_loss_sum:
      return self._train_ops[0]
    else:
      return self._train_ops[self.ind_expert]

  @property
  def training_steps(self):
    return self.training_steps_list[self.ind_expert]

  @training_steps.setter
  def training_steps(self, value):
    self.training_steps_list[self.ind_expert] = value

  @property
  def merged_summary(self):
    return self._merged_summaries[self.ind_expert]

  def _build_experts(self):
    if self.eval_mode:
      return

    self.experts = []
    for num_actions, path in zip(self.expert_num_actions, self.expert_paths):
      expert_name = os.path.basename(path) + '/online'
      expert = llamn_atari_lib.ExpertNetwork(
          num_actions,
          self.expert_num_atoms,
          self.expert_support,
          self.feature_size,
          create_llamn=self.llamn_path,
          init_option=self.expert_init_option,
          distributional_night=self.distributional_night,
          name=expert_name)
      self.experts.append(expert)

  def _build_prev_llamn(self):
    if not self.eval_mode and self.llamn_path:
      self.previous_llamn = llamn_atari_lib.AMNNetwork(self.llamn_num_actions,
                                                       self.llamn_num_atoms,
                                                       self.llamn_support,
                                                       self.feature_size,
                                                       name='prev_llamn')

  def _build_replay_buffers(self):
    if self.replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self.replay_scheme))

    self._replays = []
    for _ in range(self.nb_experts):
      replay = prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
          observation_shape=self.observation_shape,
          stack_size=self.stack_size,
          update_horizon=self.update_horizon,
          gamma=self.gamma,
          observation_dtype=self.observation_dtype.as_numpy_dtype)
      self._replays.append(replay)

  def load_networks(self):
    for i in range(self.nb_experts):
      path, expert = self.expert_paths[i], self.experts[i]

      ckpt = tf.compat.v1.train.get_checkpoint_state(path + "/checkpoints")
      ckpt_path = ckpt.model_checkpoint_path

      saver = tf.compat.v1.train.Saver(var_list=expert.variables)
      saver.restore(self._sess, ckpt_path)

    if self.llamn_path and self.llamn_init_copy:
      ckpt = tf.compat.v1.train.get_checkpoint_state(self.llamn_path + "/checkpoints")
      ckpt_path = ckpt.model_checkpoint_path

      saver = tf.compat.v1.train.Saver(var_list=self.convnet.variables)
      saver.restore(self._sess, ckpt_path)

      # var_names = {var.name[5:-2]: var
      #              for var in self.previous_llamn.variables}
      # saver = tf.compat.v1.train.Saver(var_list=var_names)
      # saver.restore(self._sess, ckpt_path)


  def load_buffers(self):
    if not self.eval_mode and self.llamn_path:
      self.buffer_loaded = True
      for i in range(self.nb_experts):
        path = self.expert_paths[i] + "/checkpoints"

        ckpt = tf.compat.v1.train.get_checkpoint_state(path)
        iteration_number = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
        self._replays[i].load(path, iteration_number)

  def _create_network(self):
    network = self.network(self.llamn_num_actions, self.llamn_num_atoms,
                           self.llamn_support, self.feature_size, name='llamn')
    return network

  def _build_networks(self):
    self.convnet = self._create_network()

    self._net_q_output = []
    self._net_q_argmax = []
    self._net_q_distrib = []
    self._net_features = []
    self._expert_q_distrib = []
    self._expert_features = []
    self._previous_llamn_output = []

    net_q_values = self.convnet(self.state_ph).q_values

    for i in range(self.nb_experts):
      expert_mask = [n_action < self.expert_num_actions[i]
                     for n_action in range(self.llamn_num_actions)]

      partial_q_values = tf.boolean_mask(net_q_values, expert_mask, axis=1)
      self._net_q_output.append(partial_q_values)
      q_argmax = tf.argmax(partial_q_values, axis=1)[0]
      self._net_q_argmax.append(q_argmax)

      # Need experts only to compute the loss, no need when testing the policy
      if not self.eval_mode:
        replay_state = self._replays[i].states

        net_output = self.convnet(replay_state)
        if not self.distributional_night:
          partial_q_values = tf.boolean_mask(net_output.q_values, expert_mask, axis=1)
          net_q_distrib = tf.nn.softmax(partial_q_values, axis=1)
        else:
          net_q_distrib = tf.boolean_mask(net_output.probabilities, expert_mask, axis=1)
        self._net_q_distrib.append(net_q_distrib)
        self._net_features.append(net_output.features)

        expert_output = self.experts[i](replay_state)
        if not self.distributional_night:
          expert_q_distrib = tf.nn.softmax(expert_output.q_values, axis=1)
        else:
          expert_q_distrib = expert_output.probabilities
        self._expert_q_distrib.append(tf.stop_gradient(expert_q_distrib))
        self._expert_features.append(tf.stop_gradient(expert_output.features))

        # if self.llamn_path:
          # llamn_output = self.previous_llamn(replay_state)
          # self._previous_llamn_output.append(tf.stop_gradient(llamn_output.features))

  def _build_xent_loss(self, i_task):
    expert_q_distrib = self._expert_q_distrib[i_task]
    net_q_distrib = self._net_q_distrib[i_task]

    log_net_distrib = tf.minimum(tf.math.log(net_q_distrib + 1e-10), 0.0)
    loss = expert_q_distrib * log_net_distrib

    if not self.distributional_night:
      return -tf.reduce_sum(loss, axis=1)
    else:
      return -tf.reduce_sum(loss, axis=(1, 2))

  def _build_l2_loss(self, i_task):
    expert_features = self._expert_features[i_task]
    net_features = self._net_features[i_task]

    loss = tf.reduce_sum(tf.square(expert_features - net_features), axis=1)
    return loss

  def _build_ewc_loss(self):
    if self.llamn_path is None:
      return 0

    pass

  def _build_train_ops(self):
    if self.eval_mode:
      return tf.no_op()

    ewc_loss = self._build_ewc_loss()
    xent_losses = []
    l2_losses = []
    total_losses = []

    total_loss = 0
    update_priorities_ops = []
    train_ops = []

    for i_task in range(self.nb_experts):

      xent_loss = self._build_xent_loss(i_task)
      l2_loss = self._build_l2_loss(i_task)
      loss = xent_loss + self.feature_weight * l2_loss

      if ewc_loss:
        loss = self.ewc_weight * ewc_loss + (1 - self.ewc_weight) * loss

      xent_losses.append(xent_loss)
      l2_losses.append(l2_loss)
      total_losses.append(loss)

      if self.replay_scheme == 'prioritized':

        probs = self._replays[i_task].transition['sampling_probabilities']
        loss_weights = 1.0 / tf.sqrt(probs + 1e-10)
        loss_weights /= tf.reduce_max(loss_weights)

        update_priorities_op = self._replays[i_task].tf_set_priority(
            self._replays[i_task].indices, tf.sqrt(loss + 1e-10))

        loss = loss_weights * loss

      else:
        update_priorities_op = tf.no_op()

      if self.optimize_loss_sum:
        update_priorities_ops.append(update_priorities_op)
        total_loss += tf.reduce_mean(loss)

      else:
        with tf.control_dependencies([update_priorities_op]):
          train_ops.append(self.optimizer.minimize(tf.reduce_mean(loss)))

    if self.summary_writer is not None:
      self.summaries = [[] for i in range(self.nb_experts)]

      with tf.compat.v1.variable_scope('Losses'):
        # EWC loss summary
        if ewc_loss:
          ewc_sum = tf.compat.v1.summary.scalar('Loss_EWC', tf.reduce_mean(ewc_loss))
          for list_sum in self.summaries:
            list_sum.append(ewc_sum)

        # Other losses
        for i_task in range(self.nb_experts):
          game_name = self.expert_paths[i_task].rsplit('_', 1)[1]
          self.summaries[i_task] += [
              tf.compat.v1.summary.scalar(f'{game_name}/X_entropy', tf.reduce_mean(xent_losses[i_task])),
              tf.compat.v1.summary.scalar(f'{game_name}/L2', tf.reduce_mean(l2_losses[i_task])),
              tf.compat.v1.summary.scalar(f'{game_name}/Total_loss', tf.reduce_mean(total_losses[i_task]))
          ]

    if self.optimize_loss_sum:
      with tf.control_dependencies(update_priorities_ops):
        train_op = self.optimizer.minimize(tf.reduce_mean(total_loss))
      return [train_op]

    else:
      return train_ops

  def begin_episode(self, observation):
    self._reset_state()
    self._record_observation(observation)

    if not self.eval_mode:
      self._train_step()

    self.action = self._select_action()
    return self.action

  def step(self, reward, observation):
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(self._last_observation, self.action, reward, False)
      self._train_step()

    self.action = self._select_action()
    return self.action

  def end_episode(self, reward):
    if not self.eval_mode:
      self._store_transition(self._observation, self.action, reward, True)

  def _select_action(self):
    # Choose the action with highest Q-value at the current state.
    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(
          self.epsilon_decay_period,
          self.training_steps,
          self.min_replay_history,
          self.epsilon_train)

    if random.random() < epsilon:
      return random.randint(0, self.num_actions - 1)
    else:
      return self._sess.run(self.q_argmax, {self.state_ph: self.state})

  def _is_buffer_prefilled(self):
    min_memory_size = min([replay.memory.add_count for replay in self._replays])
    return min_memory_size > self.min_replay_history

  def _train_step(self):
    if self.buffer_loaded or self._is_buffer_prefilled():
      if self.training_steps % self.update_period == 0:
        self._sess.run(self._train_op)
        if (self.summary_writer is not None
           and self.training_steps > 0
           and self.training_steps % self.summary_writing_frequency == 0):
          summary = self._sess.run(self.merged_summary)
          self.summary_writer.add_summary(summary, self.training_steps)

    self.training_steps += 1

  def _record_observation(self, observation):
    self._observation = np.reshape(observation, self.observation_shape)

    self.state = np.roll(self.state, -1, axis=-1)
    self.state[0, ..., -1] = self._observation

  def _store_transition(self,
                        last_observation,
                        action,
                        reward,
                        is_terminal,
                        priority=None):
    if priority is None:
      if self.replay_scheme == 'uniform':
        priority = 1.
      else:
        priority = self._replay.memory.sum_tree.max_recorded_priority

    if not self.eval_mode:
      self._replay.add(last_observation, action, reward, is_terminal, priority)

  def _reset_state(self):
    self.state.fill(0)

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Returns a self-contained bundle of the agent's state.

    This is used for checkpointing. It will return a dictionary containing all
    non-TensorFlow objects (to be saved into a file by the caller), and it saves
    all TensorFlow objects into a checkpoint file.

    Args:
      checkpoint_dir: str, directory where TensorFlow objects will be saved.
      iteration_number: int, iteration number to use for naming the checkpoint
        file.

    Returns:
      A dict containing additional Python objects to be checkpointed by the
        experiment. If the checkpoint directory does not exist, returns None.
    """
    if not tf.io.gfile.exists(checkpoint_dir):
      return None
    # Call the Tensorflow saver to checkpoint the graph.
    self._saver.save(
        self._sess,
        os.path.join(checkpoint_dir, 'tf_ckpt'),
        global_step=iteration_number)
    for i in range(self.nb_experts):
      game_name = self.expert_paths[i].rsplit('_', 1)[1]
      replay_path = os.path.join(checkpoint_dir, f'replay_{game_name}')
      tf.io.gfile.mkdir(replay_path)
      self._replays[i].save(replay_path, iteration_number)
    bundle_dictionary = {}
    bundle_dictionary['states'] = self.states
    bundle_dictionary['training_steps_list'] = self.training_steps_list
    return bundle_dictionary

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    """Restores the agent from a checkpoint.

    Restores the agent's Python objects to those specified in bundle_dictionary,
    and restores the TensorFlow objects to those specified in the
    checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
      agent's state.

    Args:
      checkpoint_dir: str, path to the checkpoint saved by tf.Save.
      iteration_number: int, checkpoint version, used when restoring the replay
        buffer.
      bundle_dictionary: dict, containing additional Python objects owned by
        the agent.

    Returns:
      bool, True if unbundling was successful.
    """
    try:
      # self._replay.load() will throw a NotFoundError if it does not find all
      # the necessary files.
      for i in range(self.nb_experts):
        game_name = self.expert_paths[i].rsplit('_', 1)[1]
        replay_path = os.path.join(checkpoint_dir, f'replay_{game_name}')
        self._replays[i].load(replay_path, iteration_number)
    except tf.errors.NotFoundError:
      if not self.allow_partial_reload:
        # If we don't allow partial reloads, we will return False.
        return False
      logging.warning('Unable to reload replay buffer!')
    if bundle_dictionary is not None:
      for key in self.__dict__:
        if key in bundle_dictionary:
          self.__dict__[key] = bundle_dictionary[key]
    elif not self.allow_partial_reload:
      return False
    else:
      logging.warning("Unable to reload the agent's parameters!")
    # Restore the agent's TensorFlow graph.
    self._saver.restore(self._sess,
                        os.path.join(checkpoint_dir,
                                     'tf_ckpt-{}'.format(iteration_number)))
    return True
