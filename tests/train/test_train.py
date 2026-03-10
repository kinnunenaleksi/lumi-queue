import shutil

import polars as pl
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from input.input import EXPORT_PATH
from train.params.params_features import FEATURE_SETS
from train.params.params_models import ModelConfig
from train.train import train_models

EXPORT_PATH = "test_results"

TEST_MODEL_CONFIGS = {
    "rf": ModelConfig(
        estimator=RandomForestRegressor,
        estimator_kwargs={"n_jobs": -1},
        param_grid={
            "n_estimators": [10, 20],
        },
        cv_folds=2,
        use_permutation_importance=False,
    ),
    "xgb": ModelConfig(
        estimator=xgb.XGBRegressor,
        param_grid={
            "max_depth": [1, 3],
        },
        cv_folds=2,
        use_permutation_importance=True,
    ),
}

TEST_MODELS = {
    "small-g": {
        "models": ["rf"],
        "feature_sets": ["perfect", "baseline"],
    },
    "standard": {
        "models": ["rf", "xgb"],
        "feature_sets": ["perfect"],
    },
}


def test_train_models():
    res = train_models(
        train_dict=TEST_MODELS,
        feature_sets=FEATURE_SETS,
        model_configs=TEST_MODEL_CONFIGS,
        y_col="wait_time_seconds",
        test_size=0.4,
        truncate_pct=0.02,
        split_method="timeseries",
        use_local_data=False,
        export_path=EXPORT_PATH,
    )

    assert res

    print(
        res.df_accuracy_metrics.filter(pl.col("metric") == "perc_err_under_10min").sort(
            "result_name"
        )
    )

    shutil.rmtree(EXPORT_PATH)
