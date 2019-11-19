
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


from dopamine.discrete_domains import atari_lib, llamn_atari_lib
import numpy as np
import tensorflow as tf

import gin.tf


# These are aliases which are used by other classes.
NATURE_DQN_OBSERVATION_SHAPE = atari_lib.NATURE_DQN_OBSERVATION_SHAPE
NATURE_DQN_DTYPE = atari_lib.NATURE_DQN_DTYPE


@gin.configurable
class AMNAgent(object):
  """An implementation of the LLAMN agent."""

  def __init__(self,
               sess,
               num_actions,
               feature_size,
               expert_list,
               previous_network,
               replay_memory,
               feature_weight,
               ewc_weight,
               observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=atari_lib.NATURE_DQN_DTYPE,
               stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
               network=llamn_atari_lib.AMNNetwork,
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

    self.num_actions = num_actions
    self.feature_size = feature_size
    self.experts = expert_list
    self.previous_network = previous_network
    self._replay = replay_memory
    self.feature_weight = feature_weight
    self.ewc_weight = ewc_weight
    self.observation_shape = tuple(observation_shape)
    self.observation_dtype = observation_dtype
    self.stack_size = stack_size
    self.network = network
    self.eval_mode = eval_mode
    self.training_steps = 0
    self.optimizer = optimizer
    self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency
    self.allow_partial_reload = allow_partial_reload

    with tf.device(tf_device):
      # Create a placeholder for the state input to the AMN network.
      # The last axis indicates the number of consecutive frames stacked.
      state_shape = (1,) + self.observation_shape + (stack_size, )
      self.state = np.zeros(state_shape)
      self.state_ph = tf.placeholder(self.observation_dtype, state_shape,
                                     name='state_ph')

      self._build_networks()

      self._train_op = self._build_train_op()

    if self.summary_writer is not None:
      # All tf.summaries should have been defined prior to running this.
      self._merged_summaries = tf.summary.merge_all()
    self._sess = sess

    var_map = llamn_atari_lib.maybe_transform_variable_names(tf.all_variables())
    self._saver = tf.train.Saver(var_list=var_map,
                                 max_to_keep=max_tf_checkpoints_to_keep)

  def _create_network(self, name):
    network = self.network(self.num_actions, self.feature_size, name=name)
    return network

  def _build_networks(self):
    self.convnet = self._create_network(name='Online')
    self._net_outputs = self.convnet(self.state_ph)

    self._expert_outputs = [expert(self.state_ph) for expert in self.experts]

    self.output = self._net_outputs.output
    self.q_softmax = self._net_outputs.q_softmax
    self.features = self._net_outputs.features

    # Select action
    self.q_argmax = tf.argmax(self.q_softmax, axis=1)[0]

  def _build_xent_loss(self, i_task):
    expert_output = self._expert_outputs[i_task].q_values
    expert_softmax = tf.nn.softmax(expert_output, axis=1)
    loss = expert_softmax * tf.log(self.q_softmax)
    return tf.reduce_mean(-tf.reduce_sum(loss))

  def _build_l2_loss(self, i_task):
    expert_features = self._expert_outputs[i_task].features
    loss = tf.nn.l2_loss(expert_features - self.features)
    return tf.reduce_mean(-tf.reduce_sum(loss))

  def _build_ewc_loss(self):
    return 0

  def _build_train_op(self):
    loss = 0

    for i_task in range(len(self.experts)):

      xent_loss = self._build_xent_loss(i_task)
      l2_loss = self._build_l2_loss(i_task)

      loss += xent_loss + self.feature_weight * l2_loss

    # ewc_loss = self._build_ewc_loss()

    # loss = self.ewc_weight * loss + (1 - self.ewc_weight) * ewc_loss

    if self.summary_writer is not None:
      with tf.variable_scope('Losses'):
        tf.summary.scalar('Loss', tf.reduce_mean(loss))
    print("loss :\n", loss, '\n\n', '-'*200, '\n')    # debug
    print("tf.reduce_mean(loss) :\n", tf.reduce_mean(loss), '\n\n', '-'*200, '\n')    # debug
    return self.optimizer.minimize(tf.reduce_mean(loss))

  def begin_episode(self, state):
    return self.step(state)

  def step(self, state):

    if not self.eval_mode:
      self._train_step()

    self.action = self._select_action(state)
    return self.action

  def end_episode(self, reward):
    pass

  def _select_action(self):
    # Choose the action with highest Q-value at the current state.
    return self._sess.run(self.q_argmax, {self.state_ph: self.state})

  def _train_step(self):
    self._sess.run(self._train_op)
    if (self.summary_writer is not None and self.training_steps > 0
       and self.training_steps % self.summary_writing_frequency == 0):
      summary = self._sess.run(self._merged_summaries)
      self.summary_writer.add_summary(summary, self.training_steps)

    self.training_steps += 1

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
    # Checkpoint the out-of-graph replay buffer.
    self._replay.save(checkpoint_dir, iteration_number)
    bundle_dictionary = {}
    bundle_dictionary['state'] = self.state
    bundle_dictionary['training_steps'] = self.training_steps
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
      self._replay.load(checkpoint_dir, iteration_number)
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
