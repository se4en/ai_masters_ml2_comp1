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

            train_df.loc[:, cat_col_name] = le.transform(train_df[cat_col_name])
            train_df[cat_col_name] = train_df[cat_col_name].astype("category")

            val_df.loc[:, cat_col_name] = le.transform(val_df[cat_col_name])
            val_df[cat_col_name] = val_df[cat_col_name].astype("category")

            if test_df is not None:
                test_df.loc[:, cat_col_name] = le.transform(test_df[cat_col_name])
                test_df[cat_col_name] = test_df[cat_col_name].astype("category")

        return train_df, val_df, test_df

    @staticmethod
    def __encode_time(
        train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: t.Optional[pd.DataFrame]
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame, t.Optional[pd.DataFrame]]:
        dfs = [train_df, val_df] if test_df is None else [train_df, val_df, test_df]
        for df in dfs:
            df.loc[:, "age"] = np.nan
            _not_na_ids = ~df["year_of_construction"].isna()
            df.loc[_not_na_ids, "age"] = (
                df[_not_na_ids]["timestamp"].apply(lambda x: x.year)
                - df[_not_na_ids]["year_of_construction"]
            )
            # df.loc[:, "year"] = df["timestamp"].apply(lambda x: x.year)
            # df.loc[:, "month"] = df["timestamp"].apply(lambda x: x.month)
            # df.loc[:, "day"] = df["timestamp"].apply(lambda x: x.day)

        return train_df, val_df, test_df

    @staticmethod
    def __fill_na_value(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: t.Optional[pd.DataFrame],
        feature_name: str,
        value: t.Any,
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame, t.Optional[pd.DataFrame]]:
        train_df.loc[:, feature_name] = train_df[feature_name].fillna(value)
        val_df.loc[:, feature_name] = val_df[feature_name].fillna(value)
        if test_df is not None:
            test_df.loc[:, feature_name] = test_df[feature_name].fillna(value)
        return train_df, val_df, test_df

    @staticmethod
    def __fill_na(
        train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: t.Optional[pd.DataFrame]
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame, t.Optional[pd.DataFrame]]:
        dfs = [train_df, val_df] if test_df is None else [train_df, val_df, test_df]
        for df in dfs:
            df.loc[df["year_of_construction"] == 0, "year_of_construction"] = 2000
            df.loc[df["year_of_construction"] == 1, "year_of_construction"] = 2001

            df.loc[:, "floor"] = df["floor"].fillna(1)

        year_mean = round(
            np.mean(df[~df["year_of_construction"].isna()]["year_of_construction"])
        )
        train_df, val_df, test_df = FeatureGenerator.__fill_na_value(
            train_df, val_df, test_df, "year_of_construction", year_mean
        )

        return train_df, val_df, test_df

    @staticmethod
    def __drop_columns(
        train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: t.Optional[pd.DataFrame]
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame, t.Optional[pd.DataFrame]]:
        dfs = [train_df, val_df] if test_df is None else [train_df, val_df, test_df]
        for df in dfs:
            df.drop(columns=["timestamp", "id"], inplace=True)

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
