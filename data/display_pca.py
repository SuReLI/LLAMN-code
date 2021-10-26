#!/usr/bin/env python

import argparse
import functools
import gym
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from sklearn import decomposition
from sklearn.preprocessing import StandardScaler


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
    global STATES

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
    print(f"Explained variance: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    print(f"Component wise: ", end='')
    for v in pca.explained_variance_ratio_:
        print(f"{v*100:.2f}%, ", end='')
    print()

    fig = plt.figure(feature_file)
    if n_components == 2:
        ax = fig.add_subplot()
        scatter = ax.scatter(components[:, 0], components[:, 1], c=actions, picker=True)
        # ax.tricontour(components[:, 0], components[:, 1], actions)

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

def main():
    args = parse_args()
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

            disp_pca(feature_file, action_file, args.nb_components,
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
