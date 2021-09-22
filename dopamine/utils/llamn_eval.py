
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
from dopamine.utils import llamn_eval_lib
from dopamine.discrete_domains.llamn_game_lib import create_game


flags.DEFINE_string('root_dir', 'results/', 'Root directory.')
flags.DEFINE_string('filter', '', 'Which subnetworks to test.', short_name='f')
flags.DEFINE_string('exclude', '^$', 'Which subnetworks to not test.', short_name='e')
flags.DEFINE_integer('num_eps', 3, 'Number of episodes to run.')
flags.DEFINE_integer('max_steps', 10000, 'Limit of steps to run.')
flags.DEFINE_integer('delay', 10, 'Number of ms to wait between steps in the environment.', short_name='d')
flags.DEFINE_enum('mode', None, ['save_state', 'saliency', 'features'], 'The mode of evaluation')

FLAGS = flags.FLAGS


def get_expe_dir(root_dir):
  if '/AMN_' in root_dir:
    return root_dir

  expe_list = glob.glob(os.path.join(root_dir, 'AMN_*'))
  if not expe_list:
    raise FileNotFoundError("No checkpoint found at this path")

  return max(expe_list)


def main(_):
  logging.get_absl_logger().disabled = True
  expe_dir = get_expe_dir(FLAGS.root_dir)
  print(f'\033[91mVisualizing networks in directory {expe_dir}\033[0m')

  gin.parse_config_file(os.path.join(expe_dir, 'config.gin'))
  games_names = gin.query_parameter('MasterRunner.games_names')

  all_games = [[create_game(game_name) for game_name in list_names]
               for list_names in games_names]

  nb_actions = max([game.num_actions for game_list in all_games
                    for game in game_list])

  if FLAGS.mode == 'save_state':
    gin.bind_parameter('WrappedPrioritizedReplayBuffer.batch_size', llamn_eval_lib.NB_STATES)

  elif FLAGS.mode == 'saliency':
    FLAGS.num_eps = 1
    FLAGS.max_steps = min(500, FLAGS.max_steps)

  gin.bind_parameter('Runner.max_steps_per_episode', FLAGS.max_steps)
  gin.bind_parameter('LLAMNRunner.max_steps_per_episode', FLAGS.max_steps)

  runner = llamn_eval_lib.EvalRunner(nb_actions=nb_actions,
                                     name_filter=FLAGS.filter,
                                     name_exclude=FLAGS.exclude,
                                     num_eps=FLAGS.num_eps,
                                     delay=FLAGS.delay,
                                     root_dir=expe_dir)

  for day, games in enumerate(all_games):
    for phase in ('day', 'night'):
      runner.run(games, phase, day, FLAGS.mode)


if __name__ == '__main__':
  app.run(main)
