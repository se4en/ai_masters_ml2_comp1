import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.data_laoder import load_data
from src.validation.timeline_val import train_test_split


if __name__ == "__main__":
    train_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "../../data/train.csv")
    )
    test_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "../../data/test.csv")
    )

    X_train, y_train = load_data(train_df)
    X_test, y_test = load_data(test_df)

    for train_idxs, test_idxs in train_test_split(X_train, y_train, shuffle=True):
        print(train_idxs)
        print(test_idxs)
        print()
