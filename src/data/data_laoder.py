import typing as t

import pandas as pd


def load_data(
    fname: str, target_col_name: str = "result_price"
) -> t.Tuple[pd.DataFrame, t.Optional[pd.Series]]:
    X = pd.read_csv(fname)
    if target_col_name in X:
        y = X.pop("result_price")
    else:
        y = None
    return X, y
