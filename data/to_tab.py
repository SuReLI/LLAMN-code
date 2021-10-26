#!/usr/bin/env python3

import glob
import os
import sys

import numpy as np


def find_feature_files(path):
    files = []
    if os.path.isfile(path) and path.endswith('.npy') and (
            os.path.basename(path).startswith('features_') or
            os.path.basename(path).startswith('actions_')):
        files.append(path)

    if os.path.isdir(path):
        for subpath in glob.glob(os.path.join(path, '*')):
            files.extend(find_feature_files(subpath))
    return files

def convert_file(file):
    data = np.load(file)
    if os.path.basename(file).startswith('features_'):
        return '\n'.join(['\t'.join(map(str, data[row].tolist())) for row in range(data.shape[0])])
    else:
        return '\n'.join(map(str, data.astype(np.int64).tolist()))

def main():
    if len(sys.argv) < 2:
        print("Usage: python to_tab.py path [...]")
        sys.exit()

    files = []
    for path in sys.argv[1:]:
        files.extend(find_feature_files(path))

    for file in files:
        print(f"Converting file {file}")
        tab_file = convert_file(file)
        new_path = file.replace('.npy', '.tsv')
        with open(new_path, 'w') as new_file:
            new_file.write(tab_file)

if __name__ == '__main__':
    main()
