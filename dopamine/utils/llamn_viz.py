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
import ast
import glob

from absl import app
from absl import flags
from dopamine.utils import llamn_viz_lib
from dopamine.discrete_domains.llamn_atari_lib import Game

flags.DEFINE_string('root_dir', 'results/', 'Root directory.')
flags.DEFINE_integer('num_steps', 2000, 'Number of steps to run.')

FLAGS = flags.FLAGS


def get_expe_dir(root_dir):
  if '/AMN_' in root_dir:
    return root_dir

  expe_list = glob.glob(os.path.join(root_dir, 'AMN_*'))
  if not expe_list:
    raise FileNotFoundError("No checkpoint found at this path")

  return max(expe_list)


def get_config(root_dir):
  prev_gin_config = os.path.join(root_dir, 'config.gin')
  with open(prev_gin_config, 'r') as config_file:
    config = config_file.read().split('\n')

  feature_size_line = ''
  expert_init_option_lines = ''
  games_line = ''
  for index, line in enumerate(config):
    if line.startswith('feature_size = '):
      feature_size_line = line + '\n'
    if 'init_option = ' in line:
      expert_init_option_lines += line + '\n'
    if line.startswith('MasterRunner.games_names = '):
      games_line = line
      while not config[index+1][0].isalnum():
        games_line += config[index+1]
        index += 1

  if not feature_size_line or not expert_init_option_lines or not games_line:
    raise ValueError("Feature size, xpert init option or game list not found"
                     "in saved config file")

  config = (f"{feature_size_line}"
            f"{expert_init_option_lines}"
             "WrappedReplayBuffer.replay_capacity = 300")

  games_line = games_line[games_line.index('['):]
  games_names = ast.literal_eval(games_line)
  games = [[Game(game_name) for game_name in list_names]
           for list_names in games_names]

  max_num_actions = max([game.num_actions for game_list in games
                         for game in game_list])

  return config, games, max_num_actions


def main(_):
  expe_dir = get_expe_dir(FLAGS.root_dir)
  print(f'\033[91mVisualizing networks in directory {expe_dir}\033[0m')

  config, game_list, nb_actions = get_config(expe_dir)
  for day, games in enumerate(game_list):
    for phase in ('day', 'night'):
      llamn_viz_lib.run(phase=phase,
                        nb_day=day,
                        games=games,
                        nb_actions=nb_actions,
                        num_steps=FLAGS.num_steps,
                        root_dir=expe_dir,
                        config=config)

if __name__ == '__main__':
  app.run(main)
