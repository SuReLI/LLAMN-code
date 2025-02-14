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
r"""The entry point for running a Dopamine agent.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gin

from dopamine.discrete_domains import run_experiment
from dopamine.discrete_domains.split_transfer_run_experiment import SplitMasterRunner


flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')
flags.DEFINE_boolean('resume', False, 'Whether to resume a run or not')
flags.DEFINE_string('ckpt_dir', None, 'Checkpoint dir from which to resume')
flags.DEFINE_boolean('no_parallel', False,
                     'Whether to use multiple processes or not')

flags.DEFINE_string('phase', None, 'Training phase')
flags.DEFINE_integer('index', None, 'Index of game')

FLAGS = flags.FLAGS


def main(unused_argv):
  logging.get_absl_logger().disabled = True
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

  base_dir = FLAGS.base_dir
  os.makedirs(base_dir, exist_ok=True)

  print(f'\033[91mRunning in directory {base_dir}\033[0m')

  if FLAGS.resume:
    gin.parse_config_file(os.path.join(base_dir, 'config.gin'))

  else:
    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)

  runner = SplitMasterRunner(base_dir, phase=FLAGS.phase, index=FLAGS.index)
  runner.run_experiment()
  print()


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  flags.mark_flag_as_required('phase')
  flags.mark_flag_as_required('index')
  app.run(main)
