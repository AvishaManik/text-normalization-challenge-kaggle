from pathlib import Path
from sklearn.model_selection import train_test_split
from pprint import pprint
import pandas as pd
import numpy as np
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
    training_data = pd.read_csv(input_data_path)

    max_num_features = 10
    space_letter = 0

    x_data = []
    y_data = pd.factorize(training_data['class'])
    labels = y_data[1]
    gc.collect()
    #TODO zip the sentence id into x_data so we can use it when making context frames
    for x in training_data['before'].values:
        x_row = np.ones(max_num_features, dtype=int) * space_letter
        for xi, i in zip(list(str(x)), np.arange(max_num_features)):
            x_row[i] = ord(xi)
        x_data.append(x_row)

    x_data = np.array(context_window_transform(
        data=x_data,
        pad_size=2,
        max_num_features=max_num_features,
        boundary_letter=-1))

    x_train = x_data
    y_train = y_data
    gc.collect()

#    print('x_train:')
#    pprint(x_train)
#    print('x_train:')
#    pprint(y_train)
#
#    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=2017)
#    gc.collect()
#    num_class = len(labels)
#    dtrain = xgb.DMatrix(x_train, label=y_train)
#    dvalid = xgb.DMatrix(x_valid, label=y_valid)
#    watchlist = [(dvalid, 'valid'), (dtrain, 'train')]
#
#    param = {'objective':'multi:softmax',
#             'eta':'0.3', 'max_depth':10,
#             'silent':1, 'nthread':-1,
#             'num_class':num_class,
#             'eval_metric':'merror'}
#    model = xgb.train(param, dtrain, 50, watchlist, early_stopping_rounds=20,
#                      verbose_eval=10)
#    gc.collect()
#
#    pred = model.predict(dvalid)
#    pred = [labels[int(x)] for x in pred]
#    y_valid = [labels[x] for x in y_valid]
#    x_valid = [ [ chr(x) for x in y[2 + max_num_features: 2 + max_num_features * 2]] for y in x_valid]
#    x_valid = [''.join(x) for x in x_valid]
#    x_valid = [re.sub('a+$', '', x) for x in x_valid]
#
#    gc.collect()


def context_window_transform(data, pad_size, max_num_features, boundary_letter):
    #get array of zeros
    pad = np.zeros(shape=max_num_features)
    #create array of pad arrays
    pads = list()
    for temp in range(pad_size):
        pads.append(pad)
    #example of what we're doing:
    #0   123   0   01230
    #0 + 232 + 0 = 02320
    #0   321   0   03210
    data = pads + data + pads
    flattened_context_windows = []
    for lower_bound in range(len(data) - pad_size * 2):
        flattened_context_window = []
        #calc how many tokens (lines) to look at
        context_window_size = pad_size * 2 + 1
        upper_bound = lower_bound + context_window_size
        #get a window of the data (this is a frame that moves by 1 each iteration)
        context_window = data[lower_bound:upper_bound]
        for word in context_window:
            flattened_context_window.append(boundary_letter)
            flattened_context_window.extend(word)
        flattened_context_window.append(boundary_letter)
        flattened_context_windows.append(flattened_context_window)
    return flattened_context_windows





main(sys.argv)
