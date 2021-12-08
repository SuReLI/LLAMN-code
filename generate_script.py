#!/usr/bin/env python

import argparse
import datetime
import gin
import glob
import os
import subprocess


@gin.configurable
class SplitMasterRunner:
    pass


SLURM_DIR = "slurm_scripts"
SBATCH_SCRIPT = """\
#!/bin/bash

#SBATCH --job-name={name}

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

#SBATCH --gres=gpu:1
#SBATCH --mem={memory}

#SBATCH --begin=now

#SBATCH --time={time}

{dependency_line}

#SBATCH --mail-user=valentin.guillet@isae-supaero.fr
#SBATCH --mail-type=FAIL,END
#SBATCH -o /tmpdir/p21001gv/slurm/slurm.%j.out # STDOUT
#SBATCH -e /tmpdir/p21001gv/slurm/slurm.%j.err # STDERR

module load python/3.9.5
module load cuda/11.2

source $HOME/.venv/main/bin/activate

$HOME/LLAMN-code/split_main --base_dir {base_dir} --gin_files {gin_file} --phase {phase} --index {index}
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='results/', help='Base directory to host all required sub-directories.')
    parser.add_argument('--gin_file', help='Gin file', required=True)
    parser.add_argument('--resume', action='store_true', help='Resume experiment')
    parser.add_argument('--ckpt_dir', default=None, help='Checkpoint to resume')

    return parser.parse_args()


def get_base_dir(base_dir, resume, ckpt_dir):
  if resume:
    if ckpt_dir is not None:
      if not os.path.exists(ckpt_dir):
        raise FileNotFoundError("No checkpoint found at this path")
      return ckpt_dir

    path = os.path.join(base_dir, 'AMN_*')
    expe_list = glob.glob(path)
    if not expe_list:
      raise FileNotFoundError("No checkpoint to resume")
    base_dir = max(expe_list)

  else:
    expe_time = 'AMN_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_dir = os.path.join(base_dir, expe_time) + os.sep

  return base_dir


def write_sbatch_script(base_dir, gin_file, phase_index, game_index, dependencies):

    # Day
    if game_index >= 0:
        phase = f'day_{phase_index}'
        job_name = f"day-{phase_index}_game-{game_index}"
        memory = 15000
        time = '02:00:00'

    # Night
    else:
        phase = f'night_{phase_index}'
        job_name = f"night-{phase_index}"
        memory = 60000
        time = '14:00:00'

    name = 'AMN_' + job_name

    if dependencies is not None:
        dependency_line = "#SBATCH --dependency=afterok:" + ','.join(map(str, dependencies))
    else:
        dependency_line = ''

    script = SBATCH_SCRIPT.format(base_dir=base_dir,
                                  gin_file=gin_file,
                                  phase=phase,
                                  index=game_index,
                                  dependency_line=dependency_line,
                                  time=time,
                                  memory=memory,
                                  name=name
                                  )

    if not os.path.exists(SLURM_DIR):
        os.mkdir(SLURM_DIR)

    script_name = job_name + '.slurm'
    file_name = os.path.join(SLURM_DIR, script_name)
    with open(file_name, 'w') as file:
        file.write(script)

    return file_name


def submit_sbatch(script_file):
    process = subprocess.run(["sbatch", "--parsable", script_file],
                             check=True, capture_output=True)
    job_id = int(process.stdout.decode())
    print("\tjob id", job_id)
    return job_id


def main():

    args = parse_args()

    gin.parse_config_file(args.gin_file, skip_unknown=True)
    games = gin.query_parameter("SplitMasterRunner.games_names")

    base_dir = get_base_dir(args.base_dir, args.resume, args.ckpt_dir)
    print(f'\033[91mRunning in directory {base_dir}\033[0m')

    day_dependency = None
    for phase_index, phase_games in enumerate(games):

        # ProcGen Conversion
        new_phase_games = []
        for game_name in phase_games:
            if ':' in game_name:
                env_name, infos = game_name.split('.')
                num_levels, start_level = infos.split('-')
                first_level, last_level = map(int, start_level.split(':'))
                for level in range(first_level, last_level):
                    new_game_name = f"{env_name}.{num_levels}-{level}"
                    new_phase_games.append(new_game_name)
            else:
                new_phase_games.append(game_name)
        phase_games = new_phase_games

        # Day
        night_dependencies = []
        for game_index, game in enumerate(phase_games):
            print("Building and running sbatch for game", game)
            script_file = write_sbatch_script(base_dir, args.gin_file,
                                              phase_index, game_index,
                                              day_dependency)
            night_dependencies.append(submit_sbatch(script_file))

        # Night
        print("Building and running night", ', '.join(phase_games))
        script_file = write_sbatch_script(base_dir, args.gin_file,
                                          phase_index, -1,
                                          night_dependencies)
        day_dependency = [submit_sbatch(script_file)]


if __name__ == '__main__':
    main()
