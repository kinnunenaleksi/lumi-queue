from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from train import utils
from train.params.params_models import SEED


@dataclass
class Result:
    """Model training results.

    This is the ouput of the `predict` function, that concludes the necessary training
    metrics.

    Attributes:
        df_feature_selection: Results of naive feature selection.
        df_cv_results: Results from the hyperparameter tuning.
        df_feature_importance: Results from feature importance calculations.
        df_accuracy_metrics: Accuracy metrics for the best model.
        y_pred: Array of the model predictions for the validation set.
        validation_indices: Indeces for validation set from the preprocessed data.
        best_model: Best performing Sklearn model from the tuning.
    """

    df_feature_selection: pl.DataFrame
    df_cv_results: pl.DataFrame
    df_feature_importance: pl.DataFrame
    df_accuracy_metrics: pl.DataFrame
    y_pred: np.ndarray | None
    validation_indices: np.ndarray | None
    best_model: Any


def adjust_target(df: pl.DataFrame, y_col: str, lower_bound: int, upper_bound):

    return df.with_columns(
        pl.col(y_col)
        .clip(lower_bound=lower_bound, upper_bound=upper_bound)
        .alias(y_col)
    )


def predict(
    df: pl.DataFrame,
    y_col: str,
    x_cols: list,
    no_features: int,
    test_size: float,
    model: str,
    split_method: str,
    model_configs: Any,
):
    """Main function of `regress`. Trains and tunes a model for a single partition.

    Args:
        df: Preprocessed dataframe for one partition. See `input.input.create_datasets`.
        y_col: Target variable, generally `wait_time_seconds` in this analysis.
        x_cols: Selected explanatory features.
        no_features: Number of features wanted.
        test_size: Percentage of the data used for validation.

        model: Predictive algorithm model. Can be one of:
            - "rf": Random Forest
            - "gb": Gradient Boosing
            - "xgb": XGBoost
            - "mlp": Neural Networks

        split_method: How dataset is divided into training and validation sets. Can be:
            - "random": Traditional sklearn `train_test_split`
            - "timeseries": (1-test_size)% of data in chronological order.

        model_configs: Dictionary of model parameters. See `train.params.params_model`.

    Returns:
        Result
    """
    config = model_configs[model]

    df = df.sort(pl.col("start_ts"), descending=False)

    # df = df.with_columns(pl.col("wait_time_seconds").clip(lower_bound=1))

    df = adjust_target(
        df, y_col=y_col, lower_bound=config.lower_bound, upper_bound=config.upper_bound
    )

    df = utils.log_transform_input(
        df,
        target=[y_col],
        log_transform_policy=config.log_transform_policy,
        inverse=False,
    )

    X, y, df_feature_selection, selected_cols = utils.select_features(
        df, y_col, x_cols, k=no_features
    )

    X_train, X_test, y_train, y_test, validation_indices = utils.split_data(
        X, y, test_size=test_size, split_method=split_method
    )

    sample_weights = utils.compute_sample_weights(
        y_train, method=config.sample_weight_method
    )

    df_feature_importance, df_cv_results, df_metrics, y_pred, best_model = train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        selected_cols,
        y_col,
        model=model,
        model_configs=model_configs,
        sample_weights=sample_weights,
    )

    res = Result(
        df_feature_selection=df_feature_selection,
        df_cv_results=df_cv_results,
        df_feature_importance=df_feature_importance,
        df_accuracy_metrics=df_metrics,
        y_pred=y_pred,
        validation_indices=validation_indices,
        best_model=best_model,
    )

    return res


def train_model(
    X_train,
    X_test,
    y_train,
    y_test,
    selected_cols: list,
    y_col: str,
    model: str,
    model_configs: Any,
    sample_weights: np.ndarray | None = None,
    seed: int = SEED,
):
    """Auxillary function for `predict`, does the training."""
    config = model_configs[model]

    model = config.estimator(**config.estimator_kwargs)

    X_train, X_test, y_train, y_test, x_scaler, y_scaler = utils.scale_input(
        X_train, X_test, y_train, y_test, scaling_policy=config.scaling_policy
    )

    grid_search = utils.search_cv(
        model=model, config=config, search_method=config.search_method
    )

    fit_params = {}
    if sample_weights is not None:
        fit_params["sample_weight"] = sample_weights

    grid_search.fit(X_train, y_train, **fit_params)
    best_model = grid_search.best_estimator_
    y_pred = grid_search.predict(X_test)

    if y_col == "wait_time_bin":
        y_proba = best_model.predict_proba(X_test)

    if config.scaling_policy in ["all_variables", "only_target"]:
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    if config.scaling_policy == "all_variables":
        X_test = x_scaler.inverse_transform(X_test)

    if config.log_transform_policy in ["all_variables", "only_target"]:
        y_pred = np.expm1(y_pred)
        y_test = np.expm1(y_test)

    if y_col in ["wait_time_seconds", "wait_time_minutes"]:
        df_metrics = utils.calc_performance_metrics(y_test, y_pred, y_col)
    elif y_col in ["wait_time_bin"]:
        df_metrics = utils.calc_classification_metrics(y_test, y_pred, y_proba)

    df_feature_importance = utils.fetch_feature_importance(
        best_model,
        X_test,
        y_test,
        selected_cols,
        use_permutation_importance=config.use_permutation_importance,
        seed=seed,
    )

    df_cv_results = utils.fetch_cv_results(grid_search.cv_results_)

    return (
        df_feature_importance,
        df_cv_results,
        df_metrics,
        y_pred,
        best_model,
    )
