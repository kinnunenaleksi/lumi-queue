from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)

from train import utils

from train.params_models import SEED


@dataclass
class Result:
    df_feature_selection: pl.DataFrame
    df_cv_results: pl.DataFrame
    df_feature_importance: pl.DataFrame
    df_accuracy_metrics: pl.DataFrame
    df_validation: pl.DataFrame
    best_model: Any


def predict(
    df: pl.DataFrame,
    y_col: str,
    x_cols: list,
    no_features: int,
    test_size: float,
    model: str,
    search_method: str,
    split_method: str,
    log_transform_policy: str,
    scaling_policy: str,
    model_configs: Any,
):

    df = df.sort(pl.col("start_ts"), descending=False)

    df = utils.log_transform_input(
        df, target=[y_col], log_transform_policy=log_transform_policy, inverse=False
    )

    X, y, df_feature_selection, selected_cols = utils.select_features(
        df, y_col, x_cols, k=no_features
    )

    X_train, X_test, y_train, y_test = utils.split_data(
        X, y, test_size=test_size, split_method=split_method
    )

    df_feature_importance, df_cv_results, df_metrics, df_validation, best_model = (
        train_model(
            X_train,
            X_test,
            y_train,
            y_test,
            selected_cols,
            search_method=search_method,
            model_type=model,
            log_transform_policy=log_transform_policy,
            scaling_policy=scaling_policy,
            model_configs=model_configs,
        )
    )

    res = Result(
        df_feature_selection=df_feature_selection,
        df_cv_results=df_cv_results,
        df_feature_importance=df_feature_importance,
        df_accuracy_metrics=df_metrics,
        df_validation=df_validation,
        best_model=best_model,
    )

    return res


def train_model(
    X_train,
    X_test,
    y_train,
    y_test,
    selected_cols: list,
    model_type: str,
    log_transform_policy: str,
    scaling_policy: str,
    search_method: str,
    model_configs: Any,
    seed: int = SEED,
):

    config = model_configs[model_type]

    model = config.estimator(random_state=seed, **config.estimator_kwargs)

    X_train, X_test, y_train, y_test, x_scaler, y_scaler = utils.scale_input(
        X_train, X_test, y_train, y_test, scaling_policy=scaling_policy
    )

    if search_method == "grid":
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=config.param_grid,
            cv=config.cv_folds,
            n_jobs=-1,
            verbose=3,
            **config.grid_search_kwargs,
        )

    elif search_method == "random":
        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=config.param_grid,
            cv=config.cv_folds,
            n_jobs=-1,
            verbose=3,
            **config.grid_search_kwargs,
        )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = grid_search.predict(X_test)

    if scaling_policy in ["all_variables", "only_target"]:
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    if scaling_policy == "all_variables":
        X_test = x_scaler.inverse_transform(X_test)

    df_validation = pl.DataFrame(X_test, schema=selected_cols).with_columns(
        [
            pl.Series("realized_wait_time", y_test),
            pl.Series("estimated_wait_time", y_pred),
        ]
    )

    if log_transform_policy in ["all_variables", "only_target"]:
        y_pred = np.expm1(y_pred)
        y_test = np.expm1(y_test)

        df_validation = utils.log_transform_input(
            df_validation,
            inverse=True,
            log_transform_policy=log_transform_policy,
            target=["realized_wait_time", "estimated_wait_time"],
        )

    df_metrics = utils.calc_performance_metrics(y_test, y_pred)

    if config.use_permutation_importance:
        perm_importance = permutation_importance(
            best_model, X_test, y_test, n_repeats=10, random_state=seed
        )
        importances = perm_importance.get("importances_mean")
    else:
        importances = best_model.feature_importances_

    df_feature_importance = utils.fetch_feature_importance(importances, selected_cols)

    df_cv_results = utils.fetch_cv_results(grid_search.cv_results_)

    return (
        df_feature_importance,
        df_cv_results,
        df_metrics,
        df_validation,
        best_model,
    )
