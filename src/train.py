import os
import warnings

import pandas as pd
import numpy as np
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import mlflow

from utils import (
    log_params_from_omegaconf_dict,
    create_submission,
    save_feature_importance,
)
from data.data_laoder import load_data

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)


@hydra.main(version_base="1.3.1", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    mlflow.set_tracking_uri(os.path.join(os.path.dirname(__file__), "../mlruns"))
    mlflow.set_experiment(cfg.general.runname)

    X_train, y_train = load_data(cfg.data.train_path)
    X_test, y_test = load_data(cfg.data.test_path)
    test_ids = X_test["id"].copy()

    cv = instantiate(cfg.cv)
    feature_generator = instantiate(cfg.feature_generator)
    metric = instantiate(cfg.metric)

    with mlflow.start_run() as cur_mlflow_run:
        log_params_from_omegaconf_dict(cfg)
        train_scores = []
        val_scores = []

        for train_idxs, val_idxs in cv.split(X_train, X_train["product_type"]):
            _X_train, _y_train = X_train.iloc[train_idxs], y_train.iloc[train_idxs]
            _X_val, _y_val = X_train.iloc[val_idxs], y_train.iloc[val_idxs]
            _X_train, _X_val, _ = feature_generator.process_features(
                _X_train, _X_val, X_test
            )
            model = instantiate(cfg.model)
            model.fit(_X_train, _y_train)
            _y_pred_train = np.absolute(model.predict(_X_train))
            _y_pred_val = np.absolute(model.predict(_X_val))

            train_scores.append(metric(_y_train, _y_pred_train))
            val_scores.append(metric(_y_val, _y_pred_val))

        train_score = np.mean(train_scores)
        val_score = np.mean(val_scores)
        mlflow.log_metric("train_metric", train_score)
        for i, cv_value in enumerate(val_scores):
            mlflow.log_metric(f"_val_metric_{i}", cv_value)
        mlflow.log_metric("val_metric", val_score)

        # create submission
        model = instantiate(cfg.model)
        X_train, X_test, _ = feature_generator.process_features(X_train, X_test)
        model.fit(X_train, y_train)

        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        submission_path = os.path.join(
            hydra_cfg["runtime"]["output_dir"],
            f"subm_{cur_mlflow_run.info.run_name}.csv",
        )
        create_submission(
            test_ids,
            model.predict(X_test),
            submission_path,
        )
        mlflow.log_artifact(submission_path)

        importance_path = os.path.join(
            hydra_cfg["runtime"]["output_dir"], "importance.png"
        )
        save_feature_importance(
            model,
            X_train.columns.tolist(),
            importance_path,
        )
        mlflow.log_artifact(importance_path)

    return val_score


if __name__ == "__main__":
    main()
