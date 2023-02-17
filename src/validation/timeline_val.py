import typing as t
from datetime import datetime

import numpy as np
import pandas as pd


class TimelineKFold:
    def __init__(
        self,
        n: int = 5,
        test_size: int = 3239,
        shuffle: bool = False,
        random_seed: int = 42,
        val_type: t.Literal["rolling-window", "growing-window"] = "growing-window",
        timeline_col_name: str = "timestamp",
    ) -> None:
        self.__n = n
        self.__test_size = test_size
        self.__shuffle = shuffle
        self.__random_seed = random_seed
        self.__val_type = val_type
        self.__timeline_col_name = timeline_col_name

    def split(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> t.Iterator[t.Tuple[np.ndarray, np.ndarray]]:
        assert len(X) > self.__n * self.__test_size

        X[self.__timeline_col_name] = X[self.__timeline_col_name].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d")
        )
        X = X.sort_values(by=[self.__timeline_col_name], ascending=[True])

        if self.__val_type == "rolling-window":
            for i in range(self.__n):
                if self.__n - i - 1 == 0:
                    last_idx = len(X)
                else:
                    last_idx = -(self.__n - i - 1) * self.__test_size
                if self.__shuffle:
                    yield X[
                        i * self.__test_size : -(self.__n - i) * self.__test_size
                    ].sample(frac=1, random_state=self.__random_seed).index, X[
                        -(self.__n - i) * self.__test_size : last_idx
                    ].sample(
                        frac=1, random_state=self.__random_seed
                    ).index
                else:
                    yield X[
                        i * self.__test_size : -(self.__n - i) * self.__test_size
                    ].index, X[-(self.__n - i) * self.__test_size : last_idx].index
        elif self.__val_type == "growing-window":
            for i in range(self.__n):
                if self.__n - i - 1 == 0:
                    last_idx = len(X)
                else:
                    last_idx = -(self.__n - i - 1) * self.__test_size
                if self.__shuffle:
                    yield X[: -(self.__n - i) * self.__test_size].sample(
                        frac=1, random_state=self.__random_seed
                    ).index, X[-(self.__n - i) * self.__test_size : last_idx].sample(
                        frac=1, random_state=self.__random_seed
                    ).index
                else:
                    yield X[: -(self.__n - i) * self.__test_size].index, X[
                        -(self.__n - i) * self.__test_size : last_idx
                    ].index
