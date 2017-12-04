from pathlib import Path
from pprint import pprint
import pandas as pd
import numpy as np
import xgboost as xgb
import sys

usage = 'applyBoosted.py /path/to/model/file /path/to/label_index/file.csv /path/to/input/file.csv /path/to/output/file.csv'

if len(sys.argv) < 5:
    print('not enough arguments\n')
    print(usage)
    sys.exit(1)
if len(sys.argv) > 5:
    print('too many arguments\n')
    print(usage)
    sys.exit(1)


def main(argv):
    model_filepath = Path(argv[1])
    label_encodings_filepath = Path(argv[2])
    input_data_path = Path(argv[3])
    output_data_path = Path(argv[4])

    max_num_features = 10

    model = xgb.Booster(model_file=str(model_filepath))

    xgb.plot_tree(model, fmap='', num_trees=0, rankdir='UT', ax=None)

    label_encodings = pd.read_csv(label_encodings_filepath, header=None, index_col=None)[1]
    print('label_encodings: ')
    pprint(label_encodings)

    input_data = pd.read_csv(input_data_path)
    test_data = input_data['before'].values
    test_data = make_encoded_space_padded_tokens(
        data=test_data,
        max_num_features=max_num_features,
        space_char=0
    )
    test_data = np.array(make_flat_context_windows(
        data=test_data,
        pad_size=1,
        max_num_features=max_num_features,
        boundary_letter=-1
    ))

    test_dm = xgb.DMatrix(data=test_data)
    encoded_predictions = model.predict(data=test_dm)

    labeled_data = decode_join(data=input_data, encoded_labels=encoded_predictions, label_encodings=label_encodings)
    print(labeled_data)
    labeled_data.to_csv(output_data_path)


def decode_join(data, encoded_labels, label_encodings):
    labels = list()
    for encoded_label in encoded_labels:
        label = label_encodings[int(encoded_label)]
        labels.append(label)
    labeled_data = data
    labeled_data['class'] = pd.Series(labels).values
    return labeled_data


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
