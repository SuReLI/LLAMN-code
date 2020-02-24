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
import glob
import datetime
import warnings
warnings.filterwarnings('ignore', r".*Passing \(type, 1\).*")


from absl import app
from absl import flags


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gin

from dopamine.discrete_domains import run_experiment
from dopamine.discrete_domains import llamn_run_experiment


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

FLAGS = flags.FLAGS


def get_base_dir(resume, ckpt_dir):
  if resume:
    if ckpt_dir is not None:
      if not os.path.exists(ckpt_dir):
        raise FileNotFoundError("No checkpoint found at this path")
      return ckpt_dir

    path = os.path.join(FLAGS.base_dir, 'AMN_*')
    expe_list = glob.glob(path)
    if not expe_list:
      raise FileNotFoundError("No checkpoint to resume")
    base_dir = max(expe_list)

  else:
    expe_time = 'AMN_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_dir = os.path.join(FLAGS.base_dir, expe_time)

  return base_dir


def main(unused_argv):
  """Main method.

  Args:
    unused_argv: Arguments (unused).
  """
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  base_dir = get_base_dir(FLAGS.resume, FLAGS.ckpt_dir)
  print(f'\033[91mRunning in directory {base_dir}\033[0m')

  if FLAGS.resume:
    gin.parse_config_file(os.path.join(base_dir, 'config.gin'))

  else:
    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)

  runner = llamn_run_experiment.MasterRunner(base_dir, not FLAGS.no_parallel)
  runner.run_experiment()
  print()


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
