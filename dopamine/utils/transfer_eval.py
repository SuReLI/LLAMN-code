
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import gin

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from absl import app
from absl import flags
from absl import logging
from dopamine.utils import transfer_eval_lib
from dopamine.discrete_domains.llamn_game_lib import create_game


flags.DEFINE_string('root_dir', 'results/', 'Root directory.')
flags.DEFINE_string('filter', '', 'Which networks to test.', short_name='f')
flags.DEFINE_string('exclude', '^$', 'Which networks to not test.', short_name='e')
flags.DEFINE_integer('num_eps', 3, 'Number of episodes to run.')
flags.DEFINE_integer('max_steps', 10000, 'Limit of steps to run.')
flags.DEFINE_integer('delay', 10, 'Number of ms to wait between steps in the environment.', short_name='d')
flags.DEFINE_enum('mode', None, ['saliency', 'features'], 'The mode of evaluation')

FLAGS = flags.FLAGS


def get_expe_dir(root_dir):
  if '/Transfer_' in root_dir:
    return root_dir

  expe_list = glob.glob(os.path.join(root_dir, 'Transfer_*'))
  if not expe_list:
    raise FileNotFoundError("No checkpoint found at this path")

  return max(expe_list)


def main(_):
  logging.get_absl_logger().disabled = True
  expe_dir = get_expe_dir(FLAGS.root_dir)
  print(f'\033[91mVisualizing networks in directory {expe_dir}\033[0m')

  gin.parse_config_file(os.path.join(expe_dir, 'config.gin'))
  first_game_name = gin.query_parameter('MasterRunner.first_game_name')
  games_names = gin.query_parameter('MasterRunner.transferred_games_names')

  all_games = [[create_game(first_game_name)],
               [create_game(game_name) for game_name in games_names]]

  if FLAGS.mode == 'saliency':
    FLAGS.num_eps = 1
    FLAGS.max_steps = min(500, FLAGS.max_steps)

  gin.bind_parameter('Runner.max_steps_per_episode', FLAGS.max_steps)

  runner = transfer_eval_lib.EvalRunner(name_filter=FLAGS.filter,
                                        name_exclude=FLAGS.exclude,
                                        num_eps=FLAGS.num_eps,
                                        delay=FLAGS.delay,
                                        root_dir=expe_dir)

  for nb_day, games in enumerate(all_games):
    runner.run(games, nb_day, FLAGS.mode)


if __name__ == '__main__':
  app.run(main)
