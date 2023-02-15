from datetime import datetime
import typing as t

import numpy as np
import pandas as pd


class FeatureGenerator:
    def __init__(self, columns_to_remove: t.List[str] = []):
        self.__columns_to_remove = columns_to_remove

    # def __correct_types(train_df: pd.DataFrame, test_df: pd.DataFrame):
    #     for df in [train_df, test_df]:
    #         df["timestamp"] = df["timestamp"].apply(
    #             lambda x: datetime.strptime(x, "%Y-%m-%d")
    #         )
    #         df["timestamp"] = df["timestamp"].apply(
    #             lambda x: datetime.strptime(x, "%Y-%m-%d")
    #         )

    #     return train_df, test_df

    @staticmethod
    def __encode_age(
        train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        for df in [train_df, test_df]:
            # print(df.columns)
            df["age"] = np.nan
            notna_idxs = ~df["year_of_construction"].isna()
            df.loc[notna_idxs, "age"] = df[notna_idxs].apply(
                lambda x: x["timestamp"].year - x["year_of_construction"],
                axis=1,
            )
            df.loc[df["age"] < 0, "age"] = np.nan

        return train_df, test_df

    def __remove_columns(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = train_df.drop(columns=self.__columns_to_remove)
        test_df = test_df.drop(columns=self.__columns_to_remove)
        return train_df, test_df

    def process_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> t.Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        for encode_func in [self.__encode_age, self.__remove_columns]:
            X_train, X_test = encode_func(X_train, X_test)

        return X_train, y_train, X_test, y_test
