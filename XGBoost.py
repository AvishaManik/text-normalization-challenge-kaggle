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
    encoded_space_char = 0

    space_padded_tokens = []
    #change classes to array of numeric encodings of classes
    encoded_classes = pd.factorize(training_data['class'])
    #get pandas index (basically a special array) of label names for the encoded classes
    labels = encoded_classes[1]
    gc.collect()
    for before_value in training_data['before'].values:
        #initialize array of space characters
        space_padded_token = np.ones(max_num_features, dtype=int) * encoded_space_char
        #split the token (word) into a list of characters (much like a string in C)
        before_value_c_str = list(str(before_value))
        for before_value_char, i in zip(before_value_c_str, range(max_num_features)):
            #get the unicode code point of given character
            space_padded_token[i] = ord(before_value_char)
        space_padded_tokens.append(space_padded_token)

    flat_context_windows = np.array(makeFlatContextWindows(
        data=space_padded_tokens,
        pad_size=1,
        max_num_features=max_num_features,
        boundary_letter=-1,
        sentence_ids=training_data['sentence_id'].values,
        sentence_lengths_by_id=training_data.groupby(by='sentence_id').size(),
        token_ids=training_data['token_id'].values
    ))
    print('flat c w:', len(flat_context_windows))

    x_train = flat_context_windows
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
             'silent':False, 'n_jobs':64,
             'num_class':num_class,
             'eval_metric':'merror'}
    model = xgb.train(param, dtrain, 50, watchlist, early_stopping_rounds=20,
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


def makeFlatContextWindows(data, pad_size, max_num_features, boundary_letter, sentence_ids, sentence_lengths_by_id, token_ids):
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
    print('len(data): ', len(data))
    #calc how many tokens (lines) to look at
    context_window_size = pad_size * 2 + 1
    print('context_window_size: ', context_window_size)
    print('len(sentence_ids): ', len(sentence_ids))
    previous_sentence_id = -1
    for lower_bound in range(len(data) - pad_size * 2):
        #get a window of the data (this is a frame that moves by 1 each iteration)
        upper_bound = lower_bound + context_window_size
        context_window = data[lower_bound:upper_bound]
        #check if window is spanning multiple sentences
        if lower_bound >= pad_size and upper_bound < len(sentence_ids + pad_size) and sentence_ids[lower_bound - pad_size] != sentence_ids[upper_bound - pad_size - 1]:
            lower_sentence_id = sentence_ids[lower_bound - pad_size]
            num_upper_padding_needed = context_window_size - sentence_lengths_by_id[lower_sentence_id]
            #check if the sentence length is too small to ever be included in a full frame so we include it with padding
            #only do this for the first occurence of the sentence
            if num_upper_padding_needed > 0 and previous_sentence_id != lower_sentence_id:
                #TODO check if this can be changed to correect shaped np.zeroes
                pads = list()
                for temp in range(num_upper_padding_needed):
                    pads.append(pad)
                context_window += pads
            else:
                #skip this invalid frame
#                pprint(context_window)
#                print('lower_bound: ', lower_bound,
#                      'upper_bound: ', upper_bound,
#                      'lower_sentence_id: ', lower_sentence_id,
#                      'upper_sentence_id: ', sentence_ids[upper_bound - 1],
#                      'lower_token: ', data[lower_bound],
#                      'upper_token: ', data[upper_bound - 1])
#                print('\n')
                continue
            previous_sentence_id = lower_sentence_id
        flattened_context_window = flattenContextWindow(context_window=context_window, boundary_letter=boundary_letter)
        #append window list to list of window lists
        flattened_context_windows.append(flattened_context_window)
    return flattened_context_windows


def flattenContextWindow(context_window, boundary_letter):
    flattened_context_window = []
    for word in context_window:
        flattened_context_window.append(boundary_letter)
        flattened_context_window.extend(word)
    flattened_context_window.append(boundary_letter)
    return flattened_context_window



main(sys.argv)
