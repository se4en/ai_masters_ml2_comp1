from omegaconf import DictConfig, ListConfig
import mlflow
import pandas as pd


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


def create_submission(
    X: pd.DataFrame, y_pred: pd.Series, fname: str = "submission.csv"
):
    subm_df = pd.DataFrame({"id": X["id"].tolist()})
    subm_df["result_price"] = y_pred

    subm_df.to_csv(fname, index=False)
