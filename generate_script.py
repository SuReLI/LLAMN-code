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

{main_script} --base_dir {base_dir} --gin_files {gin_file} --phase {phase} --index {index}
"""

BASH_SCRIPT = """\
#!/bin/bash

game_ids={game_ids}
mkdir -p parallel_logs

for phase_id in {phase_ids}
do
    # Day
    parallel -j {nb_jobs} {main_script} --base_dir {base_dir} \\
             --gin_files {gin_file} --phase day_$phase_id \\
             --index {{}} '>' parallel_logs/day_${{phase_id}}-game_{{}}.log \\
             ::: $(seq 0 ${{game_ids[$phase_id]}})

    # Night
    {main_script} --base_dir {base_dir} --gin_files {gin_file} \\
                  --phase night_$phase_id --index -1 > parallel_logs/night_${{phase_id}}.log
done
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='results/', help="Base directory to host all required sub-directories.")
    parser.add_argument('--gin_file', required=True, help="Gin file")
    parser.add_argument('--resume', action='store_true', help="Resume experiment")
    parser.add_argument('--ckpt_dir', default=None, help="Checkpoint to resume")
    parser.add_argument('--transfer', action='store_true', help="Generate scripts for transfer")
    parser.add_argument('--no_slurm', action='store_true', help="Produce a batch script that does not run on slurm")
    parser.add_argument('--jobs', '-j', default=2, help="Maximal number of parallel jobs (in no_slurm mode)")

    return parser.parse_args()


def convert_procgen_games(games):
    new_phase_games = []
    for game_name in games:
        if ':' in game_name:
            env_name, infos = game_name.split('.')
            num_levels, start_level = infos.split('-')
            first_level, last_level = map(int, start_level.split(':'))
            for level in range(first_level, last_level):
                new_game_name = f"{env_name}.{num_levels}-{level}"
                new_phase_games.append(new_game_name)
        else:
            new_phase_games.append(game_name)
    return new_phase_games


def get_base_dir(base_dir, is_transfer, resume, ckpt_dir):
  expe_name = "Transfer" if is_transfer else "AMN"
  if resume:
    if ckpt_dir is not None:
      if not os.path.exists(ckpt_dir):
        raise FileNotFoundError("No checkpoint found at this path")
      return ckpt_dir

    path = os.path.join(base_dir, f'{expe_name}_*')
    expe_list = glob.glob(path)
    if not expe_list:
      raise FileNotFoundError("No checkpoint to resume")
    base_dir = max(expe_list)

  else:
    expe_time = expe_name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_dir = os.path.join(base_dir, expe_time) + os.sep

  return base_dir


def write_sbatch_script(base_dir, gin_file, is_transfer, phase_index, game_index, dependencies):

    # Day
    if game_index >= 0:
        phase = f'day_{phase_index}'
        job_name = f"day-{phase_index}_game-{game_index}"
        memory = 15000
        time = '03:30:00' if is_transfer else '02:00:00'

    # Night
    else:
        phase = f'night_{phase_index}'
        job_name = f"night-{phase_index}"
        memory = 60000
        time = '16:00:00'

    if is_transfer:
        name = "Transfer_" + job_name
        main_script = "$HOME/LLAMN-code/split_transfer"

    else:
        name = "AMN_" + job_name
        main_script = "$HOME/LLAMN-code/split_main"

    if dependencies is not None:
        dependency_line = "#SBATCH --dependency=afterok:" + ','.join(map(str, dependencies))
    else:
        dependency_line = ''

    script = SBATCH_SCRIPT.format(base_dir=base_dir,
                                  gin_file=gin_file,
                                  main_script=main_script,
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


def write_run_script(games, base_dir, args, is_transfer=False):
    if is_transfer:
        main_script = "./split_transfer"
        phase_ids = "0 1"
        game_ids = f"(0 {len(games)-1})"
    else:
        main_script = "./split_main"
        phase_ids = "{0.." + str(len(games)-1) + "}"
        game_ids = "(" + ' '.join([str(len(game) - 1) for game in games]) + ")"

    script_file = BASH_SCRIPT.format(
        base_dir=base_dir,
        main_script=main_script,
        gin_file=args.gin_file,
        phase_ids=phase_ids,
        nb_jobs=args.jobs,
        game_ids=game_ids,
    )

    with open("run_all.sh", 'w') as file:
        file.write(script_file)


def submit_sbatch(script_file):
    process = subprocess.run(["sbatch", "--parsable", script_file],
                             check=True, capture_output=True)
    job_id = int(process.stdout.decode())
    print("\tjob id", job_id)
    return job_id


def run_amn(base_dir, games, args):
    games = list(map(convert_procgen_games, games))
    if args.no_slurm:
        write_run_script(games, base_dir, args)
        return

    day_dependency = None
    for phase_index, phase_games in enumerate(games):

        # Day
        night_dependencies = []
        for game_index, game in enumerate(phase_games):
            print("Building and running sbatch for game", game)
            script_file = write_sbatch_script(base_dir, args.gin_file, False,
                                              phase_index, game_index,
                                              day_dependency)
            night_dependencies.append(submit_sbatch(script_file))

        # Night
        print("Building and running night", ', '.join(phase_games))
        script_file = write_sbatch_script(base_dir, args.gin_file, False,
                                          phase_index, -1,
                                          night_dependencies)
        day_dependency = [submit_sbatch(script_file)]


def run_transfer(base_dir, first_game, transferred_games, args):
    transferred_games = convert_procgen_games(transferred_games)
    if args.no_slurm:
        write_run_script(transferred_games, base_dir, args, is_transfer=True)
        return

    script_file = write_sbatch_script(base_dir, args.gin_file, True, 0, 0, None)
    first_game_dependency = [submit_sbatch(script_file)]

    for game_index, game in enumerate(transferred_games):
        print("Building and running sbatch for game", game)
        script_file = write_sbatch_script(base_dir, args.gin_file, True,
                                          1, game_index, first_game_dependency)
        submit_sbatch(script_file)


def main():
    args = parse_args()

    gin.parse_config_file(args.gin_file, skip_unknown=True)

    base_dir = get_base_dir(args.base_dir, args.transfer, args.resume, args.ckpt_dir)
    print(f'\033[91mRunning in directory {base_dir}\033[0m')

    if args.transfer:
        first_game = gin.query_parameter("split_transfer_run_experiment.SplitMasterRunner.first_game_name")
        transferred_games = gin.query_parameter("split_transfer_run_experiment.SplitMasterRunner.transferred_games_names")
        run_transfer(base_dir, first_game, transferred_games, args)

    else:
        games = gin.query_parameter("SplitMasterRunner.games_names")
        run_amn(base_dir, games, args)


if __name__ == '__main__':
    main()
