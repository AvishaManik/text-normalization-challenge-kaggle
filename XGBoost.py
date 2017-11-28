import pandas as pd
import numpy as np
from pathlib import Path
import random
import gc
import sys

usage = 'XGBoost.py /path/to/input/file.csv'

if len(sys.argv) < 2:
    print('not enough arguments\n')
    print(usage)
    sys.exit(1)
if len(sys.argv) > 2:
    print('too many arguments\n')
    print(usage)
    sys.exit(1)


def main(argv):
    input_data_path = Path(argv[1])
    train_data = pd.read_csv(input_data_path)

    max_num_features = 10
    space_letter = 0

    x_data = []
    gc.collect()
    for x in train_data['before'].values:
        x_row = np.ones(max_num_features, dtype=int) * space_letter
        for xi, i in zip(list(str(x)), np.arange(max_num_features)):
            x_row[i] = ord(xi)
        x_data.append(x_row)

    pre = np.zeros(max_num_features)
    pre = [pre for x in np.arange(1)]
    print(pre)

    pad_size = 1
    boundary_letter = -1
    space_letter = 0

    x_data = x_data[:20]
    #y_data = y_data[:20]
    x_data = np.array(context_window_transform(
        data=x_data,
        pad_size=pad_size,
        max_num_features=max_num_features,
        boundary_letter=boundary_letter))


def context_window_transform(data, pad_size, max_num_features, boundary_letter):
    pre = np.zeros(max_num_features)
    pre = [pre for x in np.arange(pad_size)]
    data = pre + data + pre
    neo_data = []
    for i in np.arange(len(data) - pad_size * 2):
        row = []
        for x in data[i : i + pad_size * 2 + 1]:
            row.append([boundary_letter])
            row.append(x)
        row.append([boundary_letter])
        neo_data.append([int(x) for y in row for x in y])
    return neo_data

main(sys.argv)
