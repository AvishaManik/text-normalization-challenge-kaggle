from pathlib import Path
from sklearn.model_selection import train_test_split
from pprint import pprint
import pandas as pd
import numpy as np
import sys
import gc
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import RMSprop

usage = 'NNWithClassAndContext.py /path/to/input/file.csv /path/to/save/model/'

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

    preprocessed_before_tokens = make_encoded_space_padded_tokens(
        data=training_data['before'].values,
        max_num_features=10,
        space_char=0
    )

    preprocessed_before_tokens = pd.DataFrame(make_flat_context_windows(
        data=preprocessed_before_tokens,
        pad_size=1,
        max_num_features=max_num_features,
        boundary_letter=-1))

    preprocessed_after_tokens = make_encoded_space_padded_tokens(
        data=training_data['after'].values,
        max_num_features=50,
        space_char=0
    )

    x_train = pd.DataFrame(keras.utils.to_categorical(encoded_classes[0], len(labels)), columns=labels)
    x_train = x_train.join(preprocessed_before_tokens, lsuffix='', rsuffix='_before')
#    x_train['sentence_id'] = training_data['sentence_id']
    y_train = pd.DataFrame(preprocessed_after_tokens)

    gc.collect()

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=1045)
    gc.collect()

    x_train = x_train.as_matrix()
    x_valid = x_valid.as_matrix()
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))

    model = Sequential()
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=((max_num_features * 3) + 4 + len(labels), 1)))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=256, epochs=50, verbose=1, validation_data=(x_valid, y_valid))

    score = model.evaluate(x_valid, y_valid, verbose=0)
    print("Accuracy on validaton dataset")
    print(score[1])


    #model.save_model(model_output_data_path.joinpath(Path('booster_model')).name)

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
