from pathlib import Path
from sklearn.model_selection import train_test_split
from pprint import pprint
import pandas as pd
import numpy as np
import xgboost as xgb
import gc
import sys

usage = 'XGBoost.py /path/to/input/file.csv /path/to/save/model/'

if len(sys.argv) < 3:
    print('not enough arguments\n')
    print(usage)
    sys.exit(1)
if len(sys.argv) > 3:
    print('too many arguments\n')
    print(usage)
    sys.exit(1)


def main(argv):
    input_data_path = Path(argv[1])
    model_output_data_path = Path(argv[2])
    training_data = pd.read_csv(input_data_path)

    max_num_features = 10

    space_padded_tokens = []
    #change classes to array of numeric encodings of classes
    encoded_classes = pd.factorize(training_data['class'])
    #get pandas index (basically a special array) of label names for the encoded classes
    labels = encoded_classes[1]
    #write indexed labels. This ordering will be needed to decode any predictions made by this model
    labels.to_frame(index=False).to_csv(path_or_buf=str(Path('label_index.csv')), header=False)
    gc.collect()

    training_data = make_encoded_space_padded_tokens(
        data=training_data['before'].values,
        max_num_features=10,
        space_char=0
    )

    training_data = np.array(make_flat_context_windows(
        data=training_data,
        pad_size=1,
        max_num_features=max_num_features,
        boundary_letter=-1))

    x_train = training_data
    y_train = encoded_classes[0]
    gc.collect()

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=1045)
    gc.collect()
    num_class = len(labels)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)
    watchlist = [(dvalid, 'valid'), (dtrain, 'train')]

    xgbregressor_params = {
        'objective':'multi:softmax',
        'learning_rate':'0.3',
        'max_depth':10,
        'silent':False,
        'n_jobs':64, #num threads to use
        'num_class':num_class,
        'eval_metric':'merror'
    }
    model = xgb.train(params=xgbregressor_params,
                      dtrain=dtrain,
                      num_boost_round=50,
                      evals=watchlist,
                      early_stopping_rounds=20,
                      verbose_eval=1)
    gc.collect()

    encoded_predictions = model.predict(dvalid)
    predictions = list()
    observations = list()
    for encoded_prediction, encoded_observation in zip(encoded_predictions, y_valid):
        prediction = labels[int(encoded_prediction)]
        observation = labels[int(encoded_observation)]
        predictions.append(prediction)
        observations.append(observation)

    gc.collect()

    model.save_model(model_output_data_path.joinpath(Path('booster_model')).name)

def make_encoded_space_padded_tokens(data, max_num_features, space_char):
    space_padded_tokens = list()
    for before_value in data:
        #initialize array of space characters
        space_padded_token = np.ones(max_num_features, dtype=int) * space_char
        #split the token (word) into a list of characters (much like a string in C)
        before_value_c_str = list(str(before_value))
        for before_value_char, i in zip(before_value_c_str, range(max_num_features)):
            #get the unicode code point of given character
            space_padded_token[i] = ord(before_value_char)
        space_padded_tokens.append(space_padded_token)
    return space_padded_tokens

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
