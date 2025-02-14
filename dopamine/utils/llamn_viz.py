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
r"""Sample file to generate visualizations.

To run, point FLAGS.restore_checkpoint to the TensorFlow checkpoint of a
trained agent. As an example, you can download to `/tmp/checkpoints` the files
linked below:
  # pylint: disable=line-too-long
  * https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/SpaceInvaders_v4/checkpoints/tf_ckpt-199.data-00000-of-00001
  * https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/SpaceInvaders_v4/checkpoints/tf_ckpt-199.index
  * https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/SpaceInvaders_v4/checkpoints/tf_ckpt-199.meta
  # pylint: enable=line-too-long

You can then run the binary with:

```
python example_viz.py \
        --agent='rainbow' \
        --game='SpaceInvaders' \
        --num_steps=1000 \
        --root_dir='/tmp/dopamine' \
        --restore_checkpoint=/tmp/checkpoints/colab_samples_rainbow_SpaceInvaders_v4_checkpoints_tf_ckpt-199
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import gin

from absl import app
from absl import flags
from absl import logging
from dopamine.utils import llamn_viz_lib
from dopamine.discrete_domains.llamn_game_lib import create_games


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


flags.DEFINE_string('root_dir', 'results/', 'Root directory.')
flags.DEFINE_string('filter', '', 'Which subnetworks to test.', short_name='f')
flags.DEFINE_string('exclude', '^$', 'Which subnetworks to not test.', short_name='e')
flags.DEFINE_integer('num_steps', 2000, 'Number of steps to run.')

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

  games = [create_games(list_names) for list_names in games_names]

  nb_actions = max([game.num_actions for game_list in games
                    for game in game_list])

  for day, games in enumerate(games):
    for phase in ('day', 'night'):
      llamn_viz_lib.run(phase=phase,
                        nb_day=day,
                        games=games,
                        nb_actions=nb_actions,
                        name_filter=FLAGS.filter,
                        name_exclude=FLAGS.exclude,
                        num_steps=FLAGS.num_steps,
                        root_dir=expe_dir)


if __name__ == '__main__':
  app.run(main)
