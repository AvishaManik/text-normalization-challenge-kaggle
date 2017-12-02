from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import sys

usage = 'applyBoosted.py /path/to/model/file /path/to/input/file.csv'

if len(sys.argv) < 3:
    print('not enough arguments\n')
    print(usage)
    sys.exit(1)
if len(sys.argv) > 3:
    print('too many arguments\n')
    print(usage)
    sys.exit(1)


def main(argv):
    model_path = Path(argv[1])
    model = xgb.Booster.load_model(model_path)
    test_data_path = Path(argv[2])
    test_data = pd.read_csv(test_data_path)





main(sys.argv)
