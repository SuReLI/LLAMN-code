
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import os
import pickle

import numpy as np
import tensorflow as tf

import gin.tf

ReplayElement = (
    collections.namedtuple('shape_type', ['name', 'shape', 'type']))

STORE_FILENAME_PREFIX = '$store$_'

CHECKPOINT_DURATION = 4


class SleepingReplayBuffer(object):

  def __init__(self,
               observation_shape,
               feature_shape,
               replay_capacity,
               batch_size,
               observation_dtype=np.uint8,
               feature_dtype=np.float32):

    assert isinstance(observation_shape, tuple)

    tf.logging.info(
        'Creating a %s sleeping memory with the following parameters:',
        self.__class__.__name__)
    tf.logging.info('\t observation_shape: %s', str(observation_shape))
    tf.logging.info('\t observation_dtype: %s', str(observation_dtype))
    tf.logging.info('\t replay_capacity: %d', replay_capacity)
    tf.logging.info('\t batch_size: %d', batch_size)

    self._observation_shape = observation_shape
    self._feature_shape = feature_shape
    self._state_shape = self._observation_shape
    self._replay_capacity = replay_capacity
    self._batch_size = batch_size
    self._observation_dtype = observation_dtype
    self._feature_dtype = feature_dtype
    self._create_storage()
    self.add_count = np.array(0)

  def _create_storage(self):
    self._store = {}
    for storage_element in self.get_storage_signature():
      array_shape = [self._replay_capacity] + list(storage_element.shape)
      self._store[storage_element.name] = np.empty(
          array_shape, dtype=storage_element.type)

  def get_add_args_signature(self):
    return self.get_storage_signature()

  def get_storage_signature(self):
    storage_elements = [
        ReplayElement('observation', self._observation_shape,
                      self._observation_dtype),
        ReplayElement('feature', self._feature_shape, self._feature_dtype)
    ]

    return storage_elements

  def add(self, observation, feature):
    self._check_add_types(observation, feature)
    self._add(observation, feature)

  def _add(self, *args):
    self._check_args_length(*args)
    transition = {e.name: args[idx]
                  for idx, e in enumerate(self.get_add_args_signature())}
    self._add_transition(transition)

  def _add_transition(self, transition):
    cursor = self.cursor()
    for arg_name in transition:
      self._store[arg_name][cursor] = transition[arg_name]

    self.add_count += 1

  def _check_args_length(self, *args):

    if len(args) != len(self.get_add_args_signature()):
      raise ValueError('Add expects {} elements, received {}'.format(
          len(self.get_add_args_signature()), len(args)))

  def _check_add_types(self, *args):
    self._check_args_length(*args)
    for arg_element, store_element in zip(args, self.get_add_args_signature()):
      if isinstance(arg_element, np.ndarray):
        arg_shape = arg_element.shape
      elif isinstance(arg_element, tuple) or isinstance(arg_element, list):
        # TODO(b/80536437). This is not efficient when arg_element is a list.
        arg_shape = np.array(arg_element).shape
      else:
        # Assume it is scalar.
        arg_shape = tuple()
      store_element_shape = tuple(store_element.shape)
      if arg_shape != store_element_shape:
        raise ValueError('arg has shape {}, expected {}'.format(
            arg_shape, store_element_shape))

  def is_empty(self):
    return self.add_count == 0

  def is_full(self):
    return self.add_count >= self._replay_capacity

  def cursor(self):
    return self.add_count % self._replay_capacity

  def _create_batch_arrays(self, batch_size):
    transition_elements = self.get_transition_elements(batch_size)
    batch_arrays = []
    for element in transition_elements:
      batch_arrays.append(np.empty(element.shape, dtype=element.type))
    return tuple(batch_arrays)

  def sample_index_batch(self, batch_size):
    max_id = min(self.add_count, self._replay_capacity)
    indices = np.random.randint(0, max_id, batch_size)
    return indices

  def sample_transition_batch(self, batch_size=None, indices=None):
    if batch_size is None:
      batch_size = self._batch_size
    if indices is None:
      indices = self.sample_index_batch(batch_size)
    assert len(indices) == batch_size

    batch_arrays = self._create_batch_arrays(batch_size)

    for batch_element, state_index in enumerate(indices):
      batch_arrays['state'][batch_element] = self._store['state'][state_index]
      batch_arrays['feature'][batch_element] = self._store['feature'][state_index]

    return batch_arrays

  def get_transition_elements(self, batch_size=None):
    batch_size = self._batch_size if batch_size is None else batch_size

    transition_elements = [
        ReplayElement('state', (batch_size,) + self._state_shape,
                      self._observation_dtype),
        ReplayElement('feature', (batch_size,) + self._feature_shape,
                      self._feature_dtype)
    ]
    return transition_elements

  def _generate_filename(self, checkpoint_dir, name, suffix):
    return os.path.join(checkpoint_dir, '{}_ckpt.{}.gz'.format(name, suffix))

  def _return_checkpointable_elements(self):
    """Return the dict of elements of the class for checkpointing.

    Returns:
      checkpointable_elements: dict containing all non private (starting with
      _) members + all the arrays inside self._store.
    """
    checkpointable_elements = {}
    for member_name, member in self.__dict__.items():
      if member_name == '_store':
        for array_name, array in self._store.items():
          checkpointable_elements[STORE_FILENAME_PREFIX + array_name] = array
      elif not member_name.startswith('_'):
        checkpointable_elements[member_name] = member
    return checkpointable_elements

  def save(self, checkpoint_dir, iteration_number):
    """Save the OutOfGraphReplayBuffer attributes into a file.

    This method will save all the replay buffer's state in a single file.

    Args:
      checkpoint_dir: str, the directory where numpy checkpoint files should be
        saved.
      iteration_number: int, iteration_number to use as a suffix in naming
        numpy checkpoint files.
    """
    if not tf.gfile.Exists(checkpoint_dir):
      return

    checkpointable_elements = self._return_checkpointable_elements()

    for attr in checkpointable_elements:
      filename = self._generate_filename(checkpoint_dir, attr, iteration_number)
      with tf.gfile.Open(filename, 'wb') as f:
        with gzip.GzipFile(fileobj=f) as outfile:
          # Checkpoint the np arrays in self._store with np.save instead of
          # pickling the dictionary is critical for file size and performance.
          # STORE_FILENAME_PREFIX indicates that the variable is contained in
          # self._store.
          if attr.startswith(STORE_FILENAME_PREFIX):
            array_name = attr[len(STORE_FILENAME_PREFIX):]
            np.save(outfile, self._store[array_name], allow_pickle=False)
          # Some numpy arrays might not be part of storage
          elif isinstance(self.__dict__[attr], np.ndarray):
            np.save(outfile, self.__dict__[attr], allow_pickle=False)
          else:
            pickle.dump(self.__dict__[attr], outfile)

      # After writing a checkpoint file, we garbage collect the checkpoint file
      # that is four versions old.
      stale_iteration_number = iteration_number - CHECKPOINT_DURATION
      if stale_iteration_number >= 0:
        stale_filename = self._generate_filename(checkpoint_dir, attr,
                                                 stale_iteration_number)
        try:
          tf.gfile.Remove(stale_filename)
        except tf.errors.NotFoundError:
          pass

  def load(self, checkpoint_dir, suffix):
    """Restores the object from bundle_dictionary and numpy checkpoints.

    Args:
      checkpoint_dir: str, the directory where to read the numpy checkpointed
        files from.
      suffix: str, the suffix to use in numpy checkpoint files.

    Raises:
      NotFoundError: If not all expected files are found in directory.
    """
    save_elements = self._return_checkpointable_elements()
    # We will first make sure we have all the necessary files available to avoid
    # loading a partially-specified (i.e. corrupted) replay buffer.
    for attr in save_elements:
      filename = self._generate_filename(checkpoint_dir, attr, suffix)
      if not tf.gfile.Exists(filename):
        raise tf.errors.NotFoundError(None, None,
                                      'Missing file: {}'.format(filename))
    # If we've reached this point then we have verified that all expected files
    # are available.
    for attr in save_elements:
      filename = self._generate_filename(checkpoint_dir, attr, suffix)
      with tf.gfile.Open(filename, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as infile:
          if attr.startswith(STORE_FILENAME_PREFIX):
            array_name = attr[len(STORE_FILENAME_PREFIX):]
            self._store[array_name] = np.load(infile, allow_pickle=False)
          elif isinstance(self.__dict__[attr], np.ndarray):
            self.__dict__[attr] = np.load(infile, allow_pickle=False)
          else:
            self.__dict__[attr] = pickle.load(infile)


@gin.configurable(blacklist=['observation_shape', 'feature_shape'])
class WrappedSleepingReplayBuffer(object):
  """Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism.

  Usage:
    To add a transition:  call the add function.

    To sample a batch:    Construct operations that depend on any of the
                          tensors is the transition dictionary. Every sess.run
                          that requires any of these tensors will sample a new
                          transition.
  """

  def __init__(self,
               observation_shape,
               feature_shape,
               use_staging=True,
               replay_capacity=1000000,
               batch_size=32,
               observation_dtype=np.uint8,
               feature_dtype=np.float32):

    self.batch_size = batch_size

    self.memory = SleepingReplayBuffer(
        observation_shape,
        feature_shape,
        replay_capacity,
        batch_size,
        observation_dtype=observation_dtype,
        feature_dtype=feature_dtype)

    self.create_sampling_ops(use_staging)

  def add(self, observation, feature):
    self.memory.add(observation, feature)

  def create_sampling_ops(self, use_staging):
    with tf.name_scope('sample_replay'):
      with tf.device('/cpu:*'):
        transition_type = self.memory.get_transition_elements()
        transition_tensors = tf.py_func(
            self.memory.sample_transition_batch, [],
            [return_entry.type for return_entry in transition_type],
            name='replay_sample_py_func')
        self._set_transition_shape(transition_tensors, transition_type)
        if use_staging:
          transition_tensors = self._set_up_staging(transition_tensors)
          self._set_transition_shape(transition_tensors, transition_type)

        # Unpack sample transition into member variables.
        self.unpack_transition(transition_tensors, transition_type)

  def _set_transition_shape(self, transition, transition_type):
    for element, element_type in zip(transition, transition_type):
      element.set_shape(element_type.shape)

  def _set_up_staging(self, transition):
    transition_type = self.memory.get_transition_elements()

    # Create the staging area in CPU.
    prefetch_area = tf.contrib.staging.StagingArea(
        [shape_with_type.type for shape_with_type in transition_type])

    # Store prefetch op for tests, but keep it private -- users should not be
    # calling _prefetch_batch.
    self._prefetch_batch = prefetch_area.put(transition)
    initial_prefetch = tf.cond(
        tf.equal(prefetch_area.size(), 0),
        lambda: prefetch_area.put(transition), tf.no_op)

    # Every time a transition is sampled self.prefetch_batch will be
    # called. If the staging area is empty, two put ops will be called.
    with tf.control_dependencies([self._prefetch_batch, initial_prefetch]):
      prefetched_transition = prefetch_area.get()

    return prefetched_transition

  def unpack_transition(self, transition_tensors, transition_type):
    self.transition = collections.OrderedDict()
    for element, element_type in zip(transition_tensors, transition_type):
      self.transition[element_type.name] = element

    self.states = self.transition['state']
    self.features = self.transition['feature']
    self.indices = self.transition['indices']

  def save(self, checkpoint_dir, iteration_number):
    self.memory.save(checkpoint_dir, iteration_number)

  def load(self, checkpoint_dir, suffix):
    self.memory.load(checkpoint_dir, suffix)
