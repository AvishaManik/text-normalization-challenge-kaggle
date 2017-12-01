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
    #change classes to array of numeric encodings of classes
    encoded_classes = pd.factorize(training_data['class'])
    #get pandas index (basically a special array) of label names for the encoded classes
    labels = encoded_classes[1]
    gc.collect()
    #TODO zip the sentence id into x_data so we can use it when making context frames
    for x in training_data['before'].values:
        x_row = np.ones(max_num_features, dtype=int) * space_letter
        for xi, i in zip(list(str(x)), np.arange(max_num_features)):
            x_row[i] = ord(xi)
        x_data.append(x_row)

    x_data = np.array(make_flat_context_windows(
        data=x_data,
        pad_size=1,
        max_num_features=max_num_features,
        boundary_letter=-1))

    x_train = x_data
    y_train = encoded_classes[0]
    gc.collect()

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=2017)
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

    encoded_predictions = model.predict(dvalid)
    predictions = list()
    observations = list()
    for encoded_prediction, encoded_observation in zip(encoded_predictions, y_valid):
        prediction = labels[int(encoded_prediction)]
        observation = labels[int(encoded_observation)]
        predictions.append(prediction)
        observations.append(observation)

    #TODO figure out what this is for
#    x_valid = [ [ chr(x) for x in y[2 + max_num_features: 2 + max_num_features * 2]] for y in x_valid]
    x_valid2 = list()
    for y in x_valid:
        for x in y[2 + max_num_features: 2 + max_num_features * 2]:
            x_valid2.append(chr(int(x)))
    print('x_valid2: ', len(x_valid2), ' ', x_valid2)

    #TODO delete? this line converts the type so he can regex it, but why remove all 'a' characters?
#    x_valid = [''.join(x) for x in x_valid2]
#    x_valid = [re.sub('a+$', '', x) for x in x_valid]
    x_valid = list()
    for char in x_valid2:
        converted_char = ''.join(char)
        not_an_a_char = re.sub('a+$', '', converted_char)
        x_valid.append(not_an_a_char)
    print('x_valid: ', len(x_valid), ' ', x_valid)

    gc.collect()


def make_flat_context_windows(data, pad_size, max_num_features, boundary_letter):
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
        #append window list to list of window lists
        flattened_context_windows.append(flattened_context_window)
    return flattened_context_windows





main(sys.argv)
