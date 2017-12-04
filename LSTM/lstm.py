from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import gc
import re
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt

def make_encoded_space_padded_tokens(data, max_num_features, space_char):
    space_padded_tokens = list()
    for before_value in data:
        space_padded_token = np.ones(max_num_features, dtype=int) * space_char
        before_value_c_str = list(str(before_value))
        for before_value_char, i in zip(before_value_c_str, range(max_num_features)):
            space_padded_token[i] = ord(before_value_char)
        space_padded_tokens.append(space_padded_token)
    return space_padded_tokens

def make_flat_context_windows(data, pad_size, max_num_features, boundary_letter):
    pad = np.zeros(shape=max_num_features)
    pads = [pad for _ in  np.arange(pad_size)]
    data = pads + data + pads
    flattened_context_windows = []
    for lower_bound in np.arange(len(data) - pad_size * 2):
        flattened_context_window = []
        context_window_size = pad_size * 2 + 1
        upper_bound = lower_bound + context_window_size
        context_window = data[lower_bound:upper_bound]
        for word in context_window:
            flattened_context_window.append([boundary_letter])
            flattened_context_window.append(word)
        flattened_context_window.append([boundary_letter])
        flattened_context_windows.append([int(word) for row in flattened_context_window for word in row])
    return flattened_context_windows


max_num_features = 10
space_letter = 0
space_padded_tokens = []
training_data = pd.read_csv("en_train.csv")
max_data_size = len(training_data)
print(training_data['class'].unique())
encoded_classes = pd.factorize(training_data['class'])
x_data = []
labels = encoded_classes[1]
y_data = encoded_classes[0]
gc.collect()
count = 0

for before_value in training_data['before'].values:
    row = np.ones(max_num_features, dtype=int) * space_letter
    for before_value_char, i in zip(list(str(before_value)), np.arange(max_num_features)):
        row[i] = ord(before_value_char)
    count+=1
    x_data.append(row)

x_data = x_data[:max_data_size]
y_data = y_data[:max_data_size]
x_data = np.array(make_flat_context_windows(x_data, pad_size = 1, max_num_features= max_num_features, boundary_letter=-1))
gc.collect()

x_train = np.array(x_data)
y_train = np.array(y_data)
gc.collect()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)
num_classes = len(labels)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
print(x_train.shape)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))

model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=((max_num_features * 3) + 4, 1)))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(labels), activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=256, epochs=50, verbose=1, validation_data=(x_valid, y_valid))

gc.collect()

score = model.evaluate(x_valid, y_valid, verbose=0)
print("Accuracy on validaton dataset")
print(score[1])

plt.plot(history.history['acc'])
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.show()

plt.plot(history.history['loss'])
plt.title('Loss')
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()

predicted_valid = model.predict(x_valid)
predicted_valid = [labels[np.argmax(x)] for x in predicted_valid]
x_valid = [[chr(x) for x in y[2 + max_num_features: 2 + max_num_features * 2]] for y in x_valid]
x_valid = [''.join(x) for x in x_valid]
x_valid = [re.sub('a+$', '', x) for x in x_valid]
gc.collect()

df_predicted_valid = pd.DataFrame(columns=['data', 'predict'])
df_predicted_valid['data'] = x_valid
df_predicted_valid['predict'] = predicted_valid
df_predicted_valid.to_csv('validation_pred_lstm.csv')

df_predicted_valid.head()

test_dataset = pd.read_csv("en_test.csv")
test_data = test_dataset['before'].values

test_data = make_encoded_space_padded_tokens(data=test_data, max_num_features=max_num_features, space_char=0)
test_data = np.array(make_flat_context_windows(data=test_data, pad_size=1, max_num_features=max_num_features, boundary_letter=-1))

test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))

predicted_data = model.predict(test_data)

predicted_data = [labels[np.argmax(char)] for char in predicted_data]
test_data = [[chr(wrd) for wrd in char[2 + max_num_features: 2 + max_num_features * 2]] for char in test_data]
test_data = [''.join(wrd) for wrd in test_data]
test_data = [re.sub('a+$', '', wrd) for wrd in test_data]

gc.collect()

df_predicted = pd.DataFrame(columns=['data', 'predict'])
df_predicted['data'] = test_data
df_predicted['predict'] = predicted_data
df_predicted.to_csv('pred_lstm.csv')

all_classes = set(training_data['class'].unique())
predicted_classes = set(df_predicted['predict'].unique())
missing_classes = all_classes - predicted_classes

print("Missing Classes:")
for cls in missing_classes: print(cls)
