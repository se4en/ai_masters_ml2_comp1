import typing as t
from datetime import datetime

import numpy as np
import pandas as pd


def train_test_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n: int = 5,
    test_size: int = 3239,
    shuffle: bool = False,
    random_seed: int = 42,
    val_type: t.Literal["rolling-window", "growing-window"] = "growing-window",
    timeline_col_name: str = "timestamp",
) -> t.Iterator[t.Tuple[np.ndarray, np.ndarray]]:
    assert len(X) > n * test_size

    X[timeline_col_name] = X[timeline_col_name].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d")
    )
    X = X.sort_values(by=[timeline_col_name], ascending=[True])

    if val_type == "rolling-window":
        for i in range(n):
            if n - i - 1 == 0:
                last_idx = len(X)
            else:
                last_idx = -(n - i - 1) * test_size
            if shuffle:
                yield X[i * test_size : (n - i) * test_size].sample(
                    frac=1, random_state=random_seed
                ).index, X[-(n - i) * test_size : last_idx].sample(
                    frac=1, random_state=random_seed
                ).index
            else:
                yield X[i * test_size : (n - i) * test_size].index, X[
                    -(n - i) * test_size : last_idx
                ].index
    elif val_type == "growing-window":
        for i in range(n):
            if n - i - 1 == 0:
                last_idx = len(X)
            else:
                last_idx = -(n - i - 1) * test_size
            if shuffle:
                yield X[: -(n - i) * test_size].sample(
                    frac=1, random_state=random_seed
                ).index, X[-(n - i) * test_size : last_idx].sample(
                    frac=1, random_state=random_seed
                ).index
            else:
                yield X[: -(n - i) * test_size].index, X[
                    -(n - i) * test_size : last_idx
                ].index
