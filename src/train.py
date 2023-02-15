import os
from time import sleep
from omegaconf import DictConfig

import pandas as pd
import numpy as np
import hydra
from hydra.utils import instantiate
import mlflow

from utils import log_params_from_omegaconf_dict
from data.data_laoder import load_data


@hydra.main(version_base="1.3.1", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    mlflow.set_tracking_uri(os.path.join(os.path.dirname(__file__), "../mlruns"))
    mlflow.set_experiment(cfg.mlflow.runname)

    train_df = pd.read_csv(cfg.pipeline.data.train_path)
    test_df = pd.read_csv(cfg.pipeline.data.test_path)

    X_train, y_train = load_data(train_df)
    X_test, y_test = load_data(test_df)

    cv = instantiate(cfg.pipeline.cv)
    feature_generator = instantiate(cfg.pipeline.feature_generator)
    metric = instantiate(cfg.pipeline.metric)

    with mlflow.start_run():
        log_params_from_omegaconf_dict(cfg)
        train_scores = []
        val_scores = []

        for train_idxs, val_idxs in cv(X_train, y_train):
            _X_train, _y_train = X_train.iloc[train_idxs], y_train.iloc[train_idxs]
            _X_val, _y_val = X_train.iloc[val_idxs], y_train.iloc[train_idxs]
            _X_train, _y_train, _X_val, _y_val = feature_generator.process_features(
                _X_train, _y_train, _X_val, _y_val
            )
            model = instantiate(cfg.pipeline.model)

            # print(f"start fit catboost {_X_train.shape}")
            model.fit(_X_train, _y_train)
            # print("end fit catboost")
            _y_pred_train = model.predict(_X_train)
            _y_pred_val = model.predict(_X_val)

            train_scores.append(metric(_y_train, _y_pred_train))
            val_scores.append(metric(_y_val, _y_pred_val))

        train_score = np.mean(train_scores)
        val_score = np.mean(val_scores)
        mlflow.log_metric("train_metric", train_score)
        mlflow.log_metric("val_metric", val_score)

    return val_score


if __name__ == "__main__":
    main()
