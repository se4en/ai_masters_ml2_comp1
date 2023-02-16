from datetime import datetime
import typing as t

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FeatureGenerator:
    def __init__(self):
        pass

    @staticmethod
    def __correct_types(
        train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: t.Optional[pd.DataFrame]
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame, t.Optional[pd.DataFrame]]:
        dfs = [train_df, val_df] if test_df is None else [train_df, val_df, test_df]
        for df in dfs:
            df["timestamp"] = df["timestamp"].apply(
                lambda x: datetime.strptime(x, "%Y-%m-%d") if isinstance(x, str) else x
            )
            df["timestamp"] = df["timestamp"].apply(
                lambda x: datetime.strptime(x, "%Y-%m-%d") if isinstance(x, str) else x
            )

        return train_df, val_df, test_df

    @staticmethod
    def __remove_outliers(
        train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: t.Optional[pd.DataFrame]
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame, t.Optional[pd.DataFrame]]:
        dfs = [train_df, val_df] if test_df is None else [train_df, val_df, test_df]
        for df in dfs:
            # flat info
            df.loc[df["floor"] == 0, "floor"] = np.nan
            df.loc[
                (df["rooms_num"] == 0) | (df["rooms_num"] >= 10), "rooms_num"
            ] = np.nan
            df.loc[
                (df["living_area"] > df["total_area"]) | (df["living_area"] > 500),
                "living_area",
            ] = np.nan
            df.loc[
                (df["kitchen_area"] >= df["total_area"]) | (df["kitchen_area"] > 500),
                "kitchen_area",
            ] = np.nan
            df.loc[
                (df["total_area"] < 5) | (df["total_area"] > 300), "total_area"
            ] = np.nan

            # others
            df.loc[df["state"] == 4, "state"] = np.nan
            df.loc[df["year_of_construction"] < 1000, "year_of_construction"] = np.nan

        return train_df, val_df, test_df

    @staticmethod
    def __encode_categorial(
        train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: t.Optional[pd.DataFrame]
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame, t.Optional[pd.DataFrame]]:
        for cat_col_name in ["district_name", "product_type"]:
            le = LabelEncoder()
            labels = train_df[cat_col_name].tolist() + val_df[cat_col_name].tolist()
            if test_df is not None:
                labels += test_df[cat_col_name].tolist()
            le.fit(labels)

            dfs = [train_df, val_df] if test_df is None else [train_df, val_df, test_df]
            for df in dfs:
                df[cat_col_name] = pd.Series(
                    le.transform(df[cat_col_name]), dtype="category"
                )

        return train_df, val_df, test_df

    @staticmethod
    def __encode_time(
        train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: t.Optional[pd.DataFrame]
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame, t.Optional[pd.DataFrame]]:
        dfs = [train_df, val_df] if test_df is None else [train_df, val_df, test_df]
        for df in dfs:
            df.loc[:, "year"] = df["timestamp"].apply(lambda x: x.year)
            df.loc[:, "month"] = df["timestamp"].apply(lambda x: x.month)

        return train_df, val_df, test_df

    @staticmethod
    def __fill_na(
        train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: t.Optional[pd.DataFrame]
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame, t.Optional[pd.DataFrame]]:
        return train_df, val_df, test_df

    @staticmethod
    def __drop_columns(
        train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: t.Optional[pd.DataFrame]
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame, t.Optional[pd.DataFrame]]:
        dfs = [train_df, val_df] if test_df is None else [train_df, val_df, test_df]
        for df in dfs:
            df.drop(columns=["timestamp"], inplace=True)

        return train_df, val_df, test_df

    def process_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: t.Optional[pd.DataFrame] = None,
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame, t.Optional[pd.DataFrame]]:
        X_train, X_val = X_train.copy(), X_val.copy()
        X_test = X_test.copy() if X_test is not None else None

        for encode_func in [
            self.__correct_types,
            self.__remove_outliers,
            self.__fill_na,
            self.__encode_categorial,
            self.__encode_time,
            self.__drop_columns,
        ]:
            X_train, X_val, X_test = encode_func(X_train, X_val, X_test)

        return X_train, X_val, X_test
