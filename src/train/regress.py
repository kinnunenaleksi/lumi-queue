from dataclasses import dataclass
from re import search
from typing import Any

import numpy as np
import polars as pl
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from train.model_params import MODEL_CONFIGS


@dataclass
class Result:
    feature_selection: pl.DataFrame
    cv_results: pl.DataFrame
    feature_importance: pl.DataFrame
    accuracy_metrics: pl.DataFrame
    y_pred: np.ndarray
    y_test: np.ndarray


def predict(
    df: pl.DataFrame,
    y_col: str,
    x_cols: list,
    no_features: int,
    test_size: float = 0.2,
    model: str = "rf",
    search_method: str = "grid",
    log_transform_policy: str = "all_variables",
    scaling_policy: str = "all_variables",
):

    df = log_transform_input(
        df, target=y_col, log_transform_policy=log_transform_policy
    )

    X, y, feature_selection, selected_cols = select_features(
        df, y_col, x_cols, k=no_features
    )

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

    df_feature_importance, df_cv_results, df_metrics, y_pred, y_test = train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        selected_cols,
        search_method=search_method,
        model_type=model,
        log_transform_policy=log_transform_policy,
        scaling_policy=scaling_policy,
    )

    res = Result(
        feature_selection=feature_selection,
        cv_results=df_cv_results,
        feature_importance=df_feature_importance,
        accuracy_metrics=df_metrics,
        y_pred=y_pred,
        y_test=y_test,
    )

    return res


def log_transform_input(
    df: pl.DataFrame,
    log_transform_policy: str = "all_variables",
    target: str = "elapsed_seconds",
):
    predictor_cols = [
        c
        for c in df.columns
        if c.startswith("allocated") or c.startswith("active") or c.startswith("queued")
    ]

    if log_transform_policy == "all_variables":
        cols = [target] + predictor_cols
    elif log_transform_policy == "only_target":
        cols = [target]
    elif log_transform_policy == "none":
        return df
    else:
        raise ValueError(
            "log_transform_policy must be one of "
            "'all_variables', 'only_target', 'none'"
        )

    return df.with_columns([pl.col(c).log1p().alias(c) for c in cols])


def split_data(X, y, test_size: float = 0.2, seed: int = 49):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def select_features(df: pl.DataFrame, y_col: str, x_cols: list, k: int = 2):
    X = df.select(pl.col(x_cols)).to_numpy()
    y = df.select(pl.col(y_col)).to_numpy().ravel()

    selector = SelectKBest(f_regression, k=k)
    X_new = selector.fit_transform(X, y)
    scores = selector.scores_

    n = len(scores)
    if k > n:
        k = n

    selected = [1] * k + [0] * (n - k)

    res = (
        pl.DataFrame({"feature": x_cols, "score": [float(s) for s in scores]})
        .sort("score", descending=True)
        .with_columns(pl.Series("selected", selected))
    )

    selected_cols = (
        res.filter(pl.col("selected") == 1)
        .select(pl.col("feature"))
        .to_series()
        .to_list()
    )

    return X_new, y, res, selected_cols


def scale_input(
    X_train, X_test, y_train, y_test, scaling_policy: str = "all_variables"
):

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    if scaling_policy == "all_variables":

        X_train = x_scaler.fit_transform(X_train)
        X_test = x_scaler.transform(X_test)
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    elif scaling_policy == "only_target":
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    elif scaling_policy == "none":
        pass

    else:
        raise ValueError("policy must be in all_variables, only_target, or none")

    return X_train, X_test, y_train, y_test, x_scaler, y_scaler


def train_model(
    X_train,
    X_test,
    y_train,
    y_test,
    selected_cols: list,
    model_type: str,
    log_transform_policy: str,
    scaling_policy: str,
    seed: int = 49,
    search_method: str = "grid",
):

    config = MODEL_CONFIGS[model_type]

    model = config.estimator(random_state=seed, **config.estimator_kwargs)

    X_train, X_test, y_train, y_test, x_scaler, y_scaler = scale_input(
        X_train, X_test, y_train, y_test, scaling_policy=scaling_policy
    )

    if search_method == "grid":
        grid_search = GridSearchCV(
            # grid_search = RandomizedSearchCV(
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
            verbose=2,
            **config.grid_search_kwargs,
        )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = grid_search.predict(X_test)

    if scaling_policy in ["all_variables", "only_target"]:
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    if log_transform_policy in ["all_variables", "only_target"]:
        y_pred = np.expm1(y_pred)
        y_test = np.expm1(y_test)

    df_metrics = calc_performance_metrics(
        y_test, y_pred, log_transform_policy=log_transform_policy
    )

    if config.use_permutation_importance:
        perm_importance = permutation_importance(
            best_model, X_test, y_test, n_repeats=10, random_state=seed
        )
        importances = perm_importance.get("importances_mean")
    else:
        importances = best_model.feature_importances_

    df_feature_importance = fetch_feature_importance(importances, selected_cols)

    df_cv_results = pl.DataFrame(grid_search.cv_results_).sort(
        "rank_test_score", descending=True
    )

    return (df_feature_importance, df_cv_results, df_metrics, y_pred, y_test)


def fetch_feature_importance(importances: Any, selected_cols: list):

    dict_feature_importance = {
        col: float(score) for col, score in zip(selected_cols, importances)
    }

    df_feature_importance = (
        pl.DataFrame(dict_feature_importance)
        .transpose(include_header=True, header_name="feature", column_names=["value"])
        .sort("value", descending=True)
    )

    return df_feature_importance


def calc_performance_metrics(
    y_test: np.ndarray, y_pred: np.ndarray, log_transform_policy: str
):

    rmse = root_mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    med = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    abs_err = abs

    abs_err = abs(y_test - y_pred)

    mask_1min = abs_err <= 60
    mask_3min = abs_err <= 180
    mask_5min = abs_err <= 300
    mask_10min = abs_err <= 600
    mask_30min = abs_err <= 1800

    df_metrics = pl.DataFrame(
        {
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "med_seconds": med,
            "perc_err_under_1min": mask_1min.mean(),
            "perc_err_under_3min": mask_3min.mean(),
            "perc_err_under_5min": mask_5min.mean(),
            "perc_err_under_10min": mask_10min.mean(),
            "perc_err_under_30min": mask_30min.mean(),
        }
    ).transpose(include_header=True, header_name="metric", column_names=["value"])

    return df_metrics
