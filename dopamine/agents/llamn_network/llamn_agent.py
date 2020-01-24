
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random

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
               min_replay_history=5000,
               update_period=4,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               num_atoms=51,
               vmax=10,
               tf_device='/cpu:*',
               eval_mode=False,
               max_tf_checkpoints_to_keep=4,
               optimizer=tf.train.RMSPropOptimizer(
                   learning_rate=0.00025,
                   decay=0.95,
                   momentum=0.0,
                   epsilon=0.00001,
                   centered=True),
               summary_writer=None,
               summary_writing_frequency=500,
               allow_partial_reload=False):

    assert isinstance(observation_shape, tuple)
    tf.logging.info('Creating %s agent with the following parameters:',
                    self.__class__.__name__)
    tf.logging.info('\t tf_device: %s', tf_device)
    tf.logging.info('\t optimizer: %s', optimizer)
    tf.logging.info('\t max_tf_checkpoints_to_keep: %d',
                    max_tf_checkpoints_to_keep)

    # LLAMN Initialization
    self.llamn_num_actions = max_num_actions
    self.expert_num_actions = expert_num_actions
    self.feature_size = feature_size
    self.expert_paths = expert_paths
    self.ind_expert = 0
    self.nb_experts = len(expert_paths)
    self.llamn_path = llamn_path
    self.feature_weight = feature_weight
    self.ewc_weight = ewc_weight

    # Rainbow Init
    vmax = float(vmax)
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
    self.update_period = update_period
    self.num_atoms = num_atoms
    self.support = tf.linspace(-vmax, vmax, num_atoms)
    self.eval_mode = eval_mode
    self.training_steps_list = [0] * self.nb_experts
    self.optimizer = optimizer
    self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency
    self.allow_partial_reload = allow_partial_reload

    with tf.device(tf_device):
      self._build_experts()
      self._build_prev_llamn()

      # Create a placeholder for the state input to the AMN network.
      # The last axis indicates the number of consecutive frames stacked.
      state_shape = (1, *self.observation_shape, stack_size)
      self.states = [np.zeros(state_shape) for i in range(self.nb_experts)]
      self.state_ph = tf.placeholder(self.observation_dtype, state_shape,
                                     name='state_ph')
      self._build_replay_buffers()

      self._build_networks()

      self._train_ops = self._build_train_ops()

    if self.summary_writer is not None:
      self._merged_summaries = [tf.summary.merge(s) for s in self.summaries]

    self._sess = sess

    var_map = atari_lib.maybe_transform_variable_names(tf.all_variables())
    self._saver = tf.train.Saver(var_list=var_map,
                                 max_to_keep=max_tf_checkpoints_to_keep)

    self._observation = None
    self._last_observation = None

  @property
  def state(self):
    return self.states[self.ind_expert]

  @state.setter
  def state(self, value):
    self.states[self.ind_expert] = value

  @property
  def replay(self):
    return self.replays[self.ind_expert]

  @property
  def q_argmax(self):
    return self._net_q_argmax[self.ind_expert]

  @property
  def _train_op(self):
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
    self.experts = []
    llamn_name = 'llamn' if self.llamn_path else None
    for num_actions, path in zip(self.expert_num_actions, self.expert_paths):
      expert_name = os.path.basename(path) + '/online'
      expert = llamn_atari_lib.ExpertNetwork(
          num_actions, self.num_atoms, self.support,
          self.feature_size, llamn_name=llamn_name, name=expert_name)
      self.experts.append(expert)

  def _build_prev_llamn(self):
    if self.llamn_path:
      self.previous_llamn = llamn_atari_lib.AMNNetwork(
          self.llamn_num_actions, self.feature_size, name='prev_llamn')

  def _build_replay_buffers(self):
    self.replays = []
    for i in range(self.nb_experts):
      replay = prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
          observation_shape=self.observation_shape,
          stack_size=self.stack_size,
          use_staging=True,
          update_horizon=self.update_horizon,
          gamma=self.gamma,
          observation_dtype=self.observation_dtype.as_numpy_dtype)
      self.replays.append(replay)

  def load_networks(self):
    for i in range(self.nb_experts):
      path, expert = self.expert_paths[i], self.experts[i]

      ckpt = tf.train.get_checkpoint_state(path + "/checkpoints")
      ckpt_path = ckpt.model_checkpoint_path

      saver = tf.train.Saver(var_list=expert.variables)
      saver.restore(self._sess, ckpt_path)

    if self.llamn_path:
      ckpt = tf.train.get_checkpoint_state(self.llamn_path + "/checkpoints")
      ckpt_path = ckpt.model_checkpoint_path

      var_names = {var.name[5:-2]: var
                   for var in self.previous_llamn.variables}
      saver = tf.train.Saver(var_list=var_names)
      saver.restore(self._sess, ckpt_path)

  def _create_network(self):
    network = self.network(self.llamn_num_actions, self.feature_size, name='llamn')
    return network

  def _build_networks(self):
    self.convnet = self._create_network()

    net_logits = self.convnet(self.state_ph).logits

    self._net_q_argmax = []
    self._net_q_softmax = []
    self._net_features = []
    self._expert_q_softmax = []
    self._expert_features = []
    self._previous_llamn_output = []

    for i in range(self.nb_experts):
      replay_state = self.replays[i].states
      expert_mask = [n_action < self.experts[i].num_actions
                     for n_action in range(self.llamn_num_actions)]

      partial_output = tf.boolean_mask(net_logits, expert_mask, axis=1)
      q_argmax = tf.argmax(partial_output, axis=1)[0]
      self._net_q_argmax.append(q_argmax)

      net_output = self.convnet(replay_state)
      partial_output = tf.boolean_mask(net_output.logits, expert_mask, axis=1)
      q_softmax = tf.nn.softmax(partial_output, axis=1)
      self._net_q_softmax.append(q_softmax)
      self._net_features.append(net_output.features)

      expert_output = self.experts[i](replay_state)
      expert_q_softmax = tf.nn.softmax(expert_output.q_values, axis=1)
      self._expert_q_softmax.append(expert_q_softmax)
      self._expert_features.append(expert_output.features)

      if self.llamn_path:
        llamn_output = self.previous_llamn(replay_state)
        self._previous_llamn_output.append(llamn_output.features)

  def _build_xent_loss(self, i_task):
    expert_softmax = self._expert_q_softmax[i_task]
    net_softmax = self._net_q_softmax[i_task]

    log_net_softmax = tf.minimum(tf.log(net_softmax + 1e-10), 0.0)
    loss = expert_softmax * log_net_softmax
    return tf.reduce_mean(-tf.reduce_sum(loss))

  def _build_l2_loss(self, i_task):
    expert_features = self._expert_features[i_task]
    net_features = self._net_features[i_task]

    loss = tf.nn.l2_loss(expert_features - net_features)
    return loss

  def _build_ewc_loss(self):
    if self.llamn_path is None:
      return 0

    pass

  def _build_train_ops(self):
    train_ops = []
    self.summaries = [[] for i in range(self.nb_experts)]

    with tf.variable_scope('Losses'):
      ewc_loss = self._build_ewc_loss()

      if self.summary_writer is not None and ewc_loss:
        ewc_sum = tf.summary.scalar(f'Loss_EWC', tf.reduce_mean(ewc_loss))
        for list_sum in self.summaries:
          list_sum.append(ewc_sum)

      for i_task in range(self.nb_experts):

        xent_loss = self._build_xent_loss(i_task)
        l2_loss = self._build_l2_loss(i_task)
        loss = xent_loss + self.feature_weight * l2_loss

        if ewc_loss:
          loss = self.ewc_weight * ewc_loss + (1 - self.ewc_weight) * loss

        if self.summary_writer is not None:
          game_name = self.expert_paths[i_task].rsplit('_', 1)[1]
          self.summaries[i_task] += [
              tf.summary.scalar(f'{game_name}/Loss_{i_task}/X_entropy', tf.reduce_mean(xent_loss)),
              tf.summary.scalar(f'{game_name}/Loss_{i_task}/L2', tf.reduce_mean(l2_loss)),
              tf.summary.scalar(f'{game_name}/Loss_{i_task}/Total_loss', tf.reduce_mean(loss))
          ]

        train_ops.append(self.optimizer.minimize(tf.reduce_mean(loss)))

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
      return random.randint(0, self.expert_num_actions[self.ind_expert] - 1)
    else:
      return self._sess.run(self.q_argmax, {self.state_ph: self.state})

  def _train_step(self):

    if self.replay.memory.add_count > self.min_replay_history:
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
      priority = self.replay.memory.sum_tree.max_recorded_priority

    if not self.eval_mode:
      self.replay.add(last_observation, action, reward, is_terminal, priority)

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
    if not tf.gfile.Exists(checkpoint_dir):
      return None
    # Call the Tensorflow saver to checkpoint the graph.
    self._saver.save(
        self._sess,
        os.path.join(checkpoint_dir, 'tf_ckpt'),
        global_step=iteration_number)
    for i in range(self.nb_experts):
      game_name = self.expert_paths[i].rsplit('_', 1)[1]
      replay_path = os.path.join(checkpoint_dir, f'replay_{game_name}')
      tf.gfile.MkDir(replay_path)
      self.replays[i].save(replay_path, iteration_number)
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
        self.replays[i].load(replay_path, iteration_number)
    except tf.errors.NotFoundError:
      if not self.allow_partial_reload:
        # If we don't allow partial reloads, we will return False.
        return False
      tf.logging.warning('Unable to reload replay buffer!')
    if bundle_dictionary is not None:
      for key in self.__dict__:
        if key in bundle_dictionary:
          self.__dict__[key] = bundle_dictionary[key]
    elif not self.allow_partial_reload:
      return False
    else:
      tf.logging.warning("Unable to reload the agent's parameters!")
    # Restore the agent's TensorFlow graph.
    self._saver.restore(self._sess,
                        os.path.join(checkpoint_dir,
                                     'tf_ckpt-{}'.format(iteration_number)))
    return True
