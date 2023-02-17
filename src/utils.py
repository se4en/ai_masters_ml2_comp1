import typing as t

from omegaconf import DictConfig, ListConfig
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{k}", v)
            else:
                mlflow.log_param(f"{parent_name}.{k}", v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f"{parent_name}.{i}", v)


def create_submission(ids: pd.Series, y_pred: pd.Series, fname: str = "submission.csv"):
    subm_df = pd.DataFrame({"id": ids})
    subm_df["result_price"] = y_pred

    subm_df.to_csv(fname, index=False)


def save_feature_importance(
    model: t.Union[LGBMRegressor, CatBoostRegressor], f_names: t.List[str], fname: str
) -> None:
    feature_imp = pd.DataFrame(
        {"Value": model.feature_importances_, "Feature": f_names}
    )
    plt.figure(figsize=(20, 10))
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False),
    )
    plt.savefig(fname)
    plt.close()
