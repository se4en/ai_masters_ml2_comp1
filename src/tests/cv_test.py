import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.data_laoder import load_data
from src.validation.timeline_val import TimelineKFold


if __name__ == "__main__":
    train_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "../../data/train.csv")
    )
    test_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "../../data/test.csv")
    )

    X_train, y_train = load_data(train_df)
    X_test, y_test = load_data(test_df)

    cv = TimelineKFold(test_size=2000, shuffle=False, val_type="rolling-window")

    for train_idxs, test_idxs in cv.split(X_train, y_train):
        print(train_idxs)
        print(test_idxs)
        print()
