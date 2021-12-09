#!/usr/bin/env python

import argparse
import collections
import functools
import gym
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.tri as tri

from sklearn import decomposition, cross_decomposition
from sklearn.preprocessing import StandardScaler, OneHotEncoder


MERGING = {"Pong": [(1, 0), (4, 2), (5, 3)]}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize PCA on features')
    parser.add_argument('files', nargs='+', help="Feature files to visualize")
    parser.add_argument('-n', '--nb_components', default=2, type=int,
                        help="Number of components in the PCA")
    parser.add_argument('-w', '--write', nargs='?', const='default',
                        type=str, dest='save_file', help="Save figure instead of showing")
    parser.add_argument('-m', '--merge', type=str, action='append',
                        help="Actions to merge together")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pls', action='store_true',
                       help="Display PLS instead of PCA")
    group.add_argument('-s', '--saliency', action='store_true',
                       help="Compute sum of saliency activations")
    group.add_argument('-v', '--variance', action='store_true',
                        help="Compare only variance")
    group.add_argument('-c', '--correlation', action='store_true',
                        help="Compute correlation")
    return parser.parse_args()

@functools.cache
def load_states(game, n):
    state_file = f"all_states/states_{n}/{game}.npy"
    assert os.path.exists(state_file), f"State file not found: {state_file}"
    states = np.load(state_file)
    return states[:, :, :, 3]

def onpick(state_imgs, event):
    state = state_imgs[event.ind[0]]
    fig = plt.figure("state_display")
    fig.gca().cla()
    fig.gca().imshow(state, cmap='gray')
    plt.draw()
    plt.pause(0.01)

def disp_pca(feature_file, action_file, n_components=2, save_file=None, merge=None, blocking=True):
    features = np.load(feature_file)
    actions = np.load(action_file).astype(np.int32)
    game_name = feature_file.split('/')[-2]

    if merge:
        try:
            if merge == ['default']:
                merge = MERGING.get(game_name, [])
            else:
                merge = [tuple(map(int, elt.split(','))) for elt in merge]
        except ValueError:
            raise ValueError("Invalid syntax for action merge") from None

        for elt in merge:
            actions[actions == elt[0]] = elt[1]

    pca = decomposition.PCA(n_components=n_components)
    features_std = StandardScaler().fit_transform(features)
    components = pca.fit_transform(features_std)

    exp_var = [v*100 for v in pca.explained_variance_ratio_]
    total_var = sum(pca.explained_variance_ratio_) * 100
    print(f"Experiment {os.path.dirname(feature_file)}")
    print(f"Explained variance: {total_var:.2f}%")
    print(f"Component wise: ", end='')
    for v in pca.explained_variance_ratio_:
        print(f"{v*100:.2f}%, ", end='')
    print('\n')

    fig = plt.figure(feature_file)
    if n_components == 2:
        ax = fig.add_subplot()
        scatter = ax.scatter(components[:, 0], components[:, 1], c=actions, picker=True)
        # scatter = ax.tricontour(components[:, 0], components[:, 1], actions)

    elif n_components == 3:
        ax = fig.add_subplot(projection='3d')
        scatter = ax.scatter(components[:, 0], components[:, 1], components[:, 2], c=actions)

    else:
        print("Can't visualize in more than 3D")
        return

    title = f"Explained variance: {total_var:.2f}%\nPer component: "
    for var in exp_var:
        title += f"{var:.2f}%, "
    ax.set_title(title)

    env = gym.make(f"{game_name}-v0")
    meanings = np.array(env.unwrapped.get_action_meanings())
    meanings = meanings[np.unique(actions)]
    ax.legend(handles=scatter.legend_elements()[0], labels=meanings.tolist())
    if save_file:
        if save_file == 'default':
            n = int(features.shape[0] ** 0.5)
            save_file = f"all_figs/{os.path.dirname(feature_file)}/feature{'3D' if n_components == 3 else ''}_{n}.pickle"
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        print("Saving in", save_file)
        with open(save_file, 'wb') as file:
            pickle.dump(fig, file)
    else:
        n_states = int(actions.shape[0]**0.5)
        states = load_states(game_name, n_states)

        callback = functools.partial(onpick, states)
        fig.canvas.mpl_connect('pick_event', callback)
        plt.show(block=blocking)
        plt.pause(0.01)

def disp_pls(feature_file, action_file, qvalues_file, n_components=2,
             save_file=None, merge=None, blocking=True):
    features = np.load(feature_file)
    actions = np.load(action_file).astype(np.int32)
    qvalues = np.load(qvalues_file)
    game_name = feature_file.split('/')[-2]

    if merge:
        try:
            if merge == ['default']:
                merge = MERGING.get(game_name, [])
            else:
                merge = [tuple(map(int, elt.split(','))) for elt in merge]
        except ValueError:
            raise ValueError("Invalid syntax for action merge") from None

        for elt in merge:
            actions[actions == elt[0]] = elt[1]

    pls = cross_decomposition.PLSRegression(n_components=n_components)
    if False:
        target = qvalues
    else:
        target = OneHotEncoder().fit_transform(actions.reshape(-1, 1)).toarray()
    components = pls.fit_transform(features, target)[0]

    breakpoint()
    explained_variance_ratio = np.var(components, axis=0) / np.var(features, axis=0).sum()
    exp_var = [v*100 for v in explained_variance_ratio]
    total_var = sum(explained_variance_ratio) * 100
    print(f"Experiment {os.path.dirname(feature_file)}")
    print(f"Explained variance: {total_var:.2f}%")
    print(f"Component wise: ", end='')
    for v in explained_variance_ratio:
        print(f"{v*100:.2f}%, ", end='')
    print('\n')


    fig = plt.figure(feature_file)
    if n_components == 2:
        ax = fig.add_subplot()
        scatter = ax.scatter(components[:, 0], components[:, 1], c=actions, picker=True)
        # scatter = ax.tricontour(components[:, 0], components[:, 1], actions)

    elif n_components == 3:
        ax = fig.add_subplot(projection='3d')
        scatter = ax.scatter(components[:, 0], components[:, 1], components[:, 2], c=actions)

    else:
        print("Can't visualize in more than 3D")
        return

    env = gym.make(f"{game_name}-v0")
    meanings = np.array(env.unwrapped.get_action_meanings())
    meanings = meanings[np.unique(actions)]
    ax.legend(handles=scatter.legend_elements()[0], labels=meanings.tolist())
    if save_file:
        if save_file == 'default':
            n = int(features.shape[0] ** 0.5)
            save_file = f"all_figs/{os.path.dirname(feature_file)}/feature{'3D' if n_components == 3 else ''}_{n}.pickle"
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        print("Saving in", save_file)
        with open(save_file, 'wb') as file:
            pickle.dump(fig, file)
    else:
        n_states = int(actions.shape[0]**0.5)
        states = load_states(game_name, n_states)

        callback = functools.partial(onpick, states)
        fig.canvas.mpl_connect('pick_event', callback)
        plt.show(block=blocking)
        plt.pause(0.01)

def disp_var(feature_dir, index_phase, n_components=2, merge=None, diff_vars=None):
    day_feature_dir = os.path.join(feature_dir, f"day_{index_phase}")
    night_feature_dir = os.path.join(feature_dir, f"night_{index_phase}")

    for game in os.listdir(day_feature_dir):
        day_features_file = os.path.join(day_feature_dir, game, "features.npy")
        night_features_file = os.path.join(night_feature_dir, game, "features.npy")
        day_features = np.load(day_features_file)
        night_features = np.load(night_features_file)

        day_pca = decomposition.PCA(n_components=n_components)
        day_features_std = StandardScaler().fit_transform(day_features)
        day_components = day_pca.fit_transform(day_features_std)
        day_exp_var = [v*100 for v in day_pca.explained_variance_ratio_]
        day_total_var = sum(day_pca.explained_variance_ratio_) * 100

        night_pca = decomposition.PCA(n_components=n_components)
        night_features_std = StandardScaler().fit_transform(night_features)
        night_components = night_pca.fit_transform(night_features_std)
        night_exp_var = [v*100 for v in night_pca.explained_variance_ratio_]
        night_total_var = sum(night_pca.explained_variance_ratio_) * 100

        if diff_vars is not None:
            diff_vars[game].append(night_total_var - day_total_var)

        print(f"Experiment {feature_dir} game {game} phase {index_phase}")
        print(f"Day/Night explained variance: {day_total_var:.2f}% / {night_total_var:.2f}%")
        print(f"Day components: ")
        for v in day_pca.explained_variance_ratio_:
            print(f"{v*100:.2f}%, ", end='')
        print(f"\nNight components: ")
        for v in night_pca.explained_variance_ratio_:
            print(f"{v*100:.2f}%, ", end='')
        print("\n")

def disp_corr(feature_files, qvalues_files, save_file=None):
    correlations, lines, game_names = [], [], []
    for feature_file, qvalues_file in zip(feature_files, qvalues_files):
        features = np.load(feature_file)
        qvalues = np.load(qvalues_file)
        game_names.append(qvalues_file.split('/')[-2])

        tab = np.hstack((features, qvalues)).T
        correlations.append(np.corrcoef(tab)[:512, 512:].T)
        lines.append(qvalues.shape[1])

    indices = np.argsort(correlations[0][0])
    correlations = [corr[:, indices] for corr in correlations]
    corr = np.vstack(correlations)

    fig, ax = plt.subplots()

    if True:  # with NaN colors
        masked_corr = np.ma.array(corr, mask=np.isnan(corr))
        cmap = matplotlib.cm.get_cmap("bwr").copy()
        cmap.set_bad('green', 1)

        ax.imshow(masked_corr, cmap=cmap, interpolation='nearest', aspect='auto')
    else:
        ax.imshow(corr, cmap='bwr', interpolation='nearest', aspect='auto')

    ylines = np.cumsum(lines)
    maj_hlines = ax.hlines(ylines[:-1]-0.5, 0, 512, colors='black')
    maj_hlines.set_linewidth(3)
    min_hlines = ax.hlines(np.arange(0.5, ylines[-1]-0.5), 0, 512, colors='black')
    min_hlines.set_linewidth(0.2)
    ax.set_xlim(0, 512)

    ticks = [ylines[0] / 2]
    for i in range(1, len(ylines)):
        ticks.append((ylines[i] + ylines[i-1]) / 2)
    ax.yaxis.set_ticks(ticks)
    ax.yaxis.set_ticklabels(game_names, rotation=90)
    plt.show()

def disp_saliencies(phase_id, day_dir, night_dir, blocking=True):
    games = os.listdir(day_dir)

    fig, axes = plt.subplots(len(games))
    fig.suptitle(f"Phase {phase_id}")
    for i, game in enumerate(games):
        day_data = np.load(os.path.join(day_dir, game, 'saliency_activations.npy'))
        night_data = np.load(os.path.join(night_dir, game, 'saliency_activations.npy'))
        axes[i].plot(day_data, label='Day')
        axes[i].plot(night_data, label='Night')
        axes[i].set_title(game)
        axes[i].legend()

    plt.show(block=blocking)


def main():
    args = parse_args()
    if args.variance:
        diff_vars = collections.defaultdict(list)
        for feature_dir in args.files:
            if 'Transfer_' in feature_dir:
                print(f"Can't compare variance for Transfer experiment {feature_dir}")
                continue

            phases = os.listdir(feature_dir)
            days = list(filter(lambda s: 'day' in s, phases))
            nb_phases = len(days)
            for i in range(nb_phases):
                if not os.path.isdir(os.path.join(feature_dir, f"night_{i}")):
                    continue
                disp_var(feature_dir, i, args.nb_components, args.merge, diff_vars)

        ind, vals = [], []
        keys = diff_vars.keys()
        boxplots = []
        for i, key in enumerate(keys):
            ind.extend([i+1] * len(diff_vars[key]))
            vals.extend(diff_vars[key])
            boxplots.append(diff_vars[key] if len(diff_vars[key]) > 1 else [])

        fig, ax = plt.subplots()
        ax.scatter(ind, vals, c=ind)
        ax.boxplot(boxplots)
        ax.set_xticks(range(1, len(keys)+1))
        ax.set_xticklabels(keys)
        ax.hlines(0, 0, len(keys), colors='black')
        ax.set_ylabel("Difference total variance explained Night-Day (in %)")
        plt.show()

    elif args.correlation:
        feature_files = []
        qvalues_files = []
        for feature_dir in args.files:
            if not (os.path.isdir(feature_dir) and 'qvalues.npy' in os.listdir(feature_dir)
                    and 'features.npy' in os.listdir(feature_dir)):
                raise ValueError("Invalid directory: directory must contain both a 'features.npy'"
                                 " and a 'qvalues.npy' files")

            feature_files.append(os.path.join(feature_dir, 'features.npy'))
            qvalues_files.append(os.path.join(feature_dir, 'qvalues.npy'))

        disp_corr(feature_files, qvalues_files, args.save_file)

    elif args.saliency:
        expe_dirs = []
        for expe_dir in args.files:
            nb_phases = len(os.listdir(expe_dir)) // 2

            for phase_id in range(nb_phases):
                blocking = (nb_phases > 4) or phase_id == nb_phases - 1
                day_dir = os.path.join(expe_dir, f'day_{phase_id}')
                night_dir = os.path.join(expe_dir, f'night_{phase_id}')

                disp_saliencies(phase_id, day_dir, night_dir, blocking)

    else:
        for feature_file in args.files:
            blocking = (len(args.files) > 4) or feature_file == args.files[-1]

            # Directory
            if os.path.isdir(feature_file):
                if 'features.npy' in os.listdir(feature_file):
                    feature_file = os.path.join(feature_file, 'features.npy')

            # Numpy feature file
            if feature_file.endswith('.npy'):
                assert os.path.exists(feature_file), f"Feature file not found: {feature_file}"
                action_file = feature_file.replace('features.', 'actions.')
                assert os.path.exists(action_file), f"No action file found: {action_file}"

                if not args.pls:
                    disp_pca(feature_file, action_file, args.nb_components,
                             args.save_file, args.merge, blocking)

                else:
                    qvalues_file = feature_file.replace('features.', 'qvalues.')
                    disp_pls(feature_file, action_file, qvalues_file, args.nb_components,
                             args.save_file, args.merge, blocking)

            # Matplotlib figure
            elif feature_file.endswith('.pickle'):
                with open(feature_file, 'rb') as file:
                    fig = pickle.load(file)
                ax = fig.gca()
                ax.set_title(feature_file + "\n" + ax.get_title())
                plt.show(block=blocking)
                plt.pause(0.01)

            else:
                raise ValueError("Invalid file: not a feature file nor a figure file")


if __name__ == '__main__':
    main()
