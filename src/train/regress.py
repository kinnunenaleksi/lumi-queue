from dataclasses import dataclass
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
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

from train.model_params import MODEL_CONFIGS


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

    df = log_transform_input(
        df, target=[y_col], log_transform_policy=log_transform_policy, inverse=False
    )

    X, y, df_feature_selection, selected_cols = select_features(
        df, y_col, x_cols, k=no_features
    )

    X_train, X_test, y_train, y_test = split_data(
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


def log_transform_input(
    df: pl.DataFrame,
    log_transform_policy: str,
    target: list,
    inverse: bool,
):
    predictor_cols = [
        c
        for c in df.columns
        if c.startswith("allocated") or c.startswith("active") or c.startswith("queued")
    ]

    if log_transform_policy == "all_variables":
        cols = target + predictor_cols
    elif log_transform_policy == "only_target":
        cols = target
    elif log_transform_policy == "none":
        return df
    else:
        raise ValueError(
            "log_transform_policy must be one of 'all_variables', 'only_target', 'none'"
        )

    if inverse:
        df = df.with_columns(
            [(pl.col(c).exp() - 1).clip(lower_bound=0).alias(c) for c in cols]
        )
    else:
        df = df.with_columns([pl.col(c).log1p().alias(c) for c in cols])

    return df


def split_data(
    X, y, test_size: float = 0.2, seed: int = 49, split_method: str = "random"
):

    if split_method == "random":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

    elif split_method == "timeseries":
        split_point = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]

    return X_train, X_test, y_train, y_test


def select_features(df: pl.DataFrame, y_col: str, x_cols: list, k: int = 2):
    X = df.select(pl.col(x_cols)).to_numpy()
    y = df.select(pl.col(y_col)).to_numpy().ravel()

    selector = SelectKBest(f_regression, k=k)
    X_new = selector.fit_transform(X, y)
    scores = selector.scores_
    mask = selector.get_support()

    n = len(scores)
    if k > n:
        k = n

    selected = [1] * k + [0] * (n - k)

    res = (
        pl.DataFrame({"feature": x_cols, "score": [float(s) for s in scores]})
        .sort("score", descending=True)
        .with_columns(pl.Series("selected", selected))
    )

    selected_cols = [col for col, m in zip(x_cols, mask) if m]

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
    model_configs: Any = MODEL_CONFIGS,
):

    config = model_configs[model_type]

    model = config.estimator(random_state=seed, **config.estimator_kwargs)

    X_train, X_test, y_train, y_test, x_scaler, y_scaler = scale_input(
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
            verbose=2,
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

        df_validation = log_transform_input(
            df_validation,
            inverse=True,
            log_transform_policy=log_transform_policy,
            target=["realized_wait_time", "estimated_wait_time"],
        )

    df_metrics = calc_performance_metrics(y_test, y_pred)

    if config.use_permutation_importance:
        perm_importance = permutation_importance(
            best_model, X_test, y_test, n_repeats=10, random_state=seed
        )
        importances = perm_importance.get("importances_mean")
    else:
        importances = best_model.feature_importances_

    df_feature_importance = fetch_feature_importance(importances, selected_cols)

    df_cv_results = fetch_cv_results(grid_search.cv_results_)

    return (
        df_feature_importance,
        df_cv_results,
        df_metrics,
        df_validation,
        best_model,
    )


def fetch_cv_results(cv_results: dict):

    df_cv_results = pl.DataFrame(cv_results)

    obj_cols = [
        c
        for c, dt in zip(df_cv_results.columns, df_cv_results.dtypes)
        if dt == pl.Object
    ]

    df_cv_results = pl.DataFrame(df_cv_results).with_columns(
        [
            pl.col(c)
            .map_elements(
                lambda x: None if x is None else str(x),
                return_dtype=pl.Utf8,
            )
            .alias(c)
            for c in obj_cols
        ]
    )

    df_cv_results = (
        pl.DataFrame(df_cv_results).sort("rank_test_score", descending=True)
    ).unnest("params")

    return df_cv_results


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


def calc_performance_metrics(y_test: np.ndarray, y_pred: np.ndarray):

    rmse = root_mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    med = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

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
