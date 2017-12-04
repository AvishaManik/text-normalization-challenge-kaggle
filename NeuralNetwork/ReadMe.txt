Language Used: Python

Packages Used:
from sklearn.model_selection import train_test_split

from pprint import pprint

import pandas as pd

import numpy as np

import sys

import os

import gc

import re

import keras

from keras.models import Sequential

from keras.layers import LSTM, Dense, Dropout

from keras.optimizers import RMSprop

import warnings

from sklearn.neural_network import MLPClassifier

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


From the directory containing Neural_Network_predict_after-values.ipynb and en_train.csv, 
execute the Ipython notebook.

Alternate Execution:
python3 Neural_Network_predict_after-values.py

The predicted outputs are saved in pred_train.csv and pred_validation.csv

