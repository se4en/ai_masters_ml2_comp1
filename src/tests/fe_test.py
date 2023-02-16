import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.data_laoder import load_data
from src.validation.timeline_val import train_test_split
from src.data.feature_engineering import FeatureGenerator

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)


if __name__ == "__main__":
    train_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "../../data/train.csv")
    )
    test_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "../../data/test.csv")
    )

    X_train, y_train = load_data(train_df)
    X_test, y_test = load_data(test_df)
    assert y_train is not None

    fg = FeatureGenerator()

    train_idxs, val_idxs = next(train_test_split(X_train, y_train, shuffle=True))
    _X_train, _y_train = X_train.iloc[train_idxs], y_train.iloc[train_idxs]
    _X_val, _y_val = X_train.iloc[val_idxs], y_train.iloc[val_idxs]

    print(_X_train.shape, _X_val.shape, X_test.shape)

    _X_train, _X_val, X_test = fg.process_features(_X_train, _X_val, X_test)

    print(_X_train.shape, _X_val.shape, X_test.shape)
    _X_train.to_csv("bruh.csv")
