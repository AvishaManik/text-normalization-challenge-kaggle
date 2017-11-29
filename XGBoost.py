import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import xgboost as xgb
import re
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
    y_data = pd.factorize(train_data['class'])
    labels = y_data[1]
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

    x_data = np.array(context_window_transform(
        data=x_data,
        pad_size=pad_size,
        max_num_features=max_num_features,
        boundary_letter=boundary_letter))

    x_train = x_data
    y_train = y_data
    gc.collect()

    x_train, x_valid, y_train, y_valid= train_test_split(x_train, y_train, test_size=0.1, random_state=2017)
    gc.collect()
    num_class = len(labels)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)
    watchlist = [(dvalid, 'valid'), (dtrain, 'train')]

    param = {'objective':'multi:softmax',
             'eta':'0.3', 'max_depth':10,
             'silent':1, 'nthread':-1,
             'num_class':num_class,
             'eval_metric':'merror'}
    model = xgb.train(param, dtrain, 50, watchlist, early_stopping_rounds=20,
                      verbose_eval=10)
    gc.collect()

    pred = model.predict(dvalid)
    pred = [labels[int(x)] for x in pred]
    y_valid = [labels[x] for x in y_valid]
    x_valid = [ [ chr(x) for x in y[2 + max_num_features: 2 + max_num_features * 2]] for y in x_valid]
    x_valid = [''.join(x) for x in x_valid]
    x_valid = [re.sub('a+$', '', x) for x in x_valid]

    gc.collect()



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
