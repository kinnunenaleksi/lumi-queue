import shutil

import polars as pl
import pytest
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from input.input import create_datasets
from train.params.params_features import FEATURE_SETS
from train.params.params_models import ModelConfig
from train.train import train_models

INPUT_PATH = "test_data/"
EXPORT_PATH = "test_results/"

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

PARTITIONS = list(TEST_MODELS.keys())


@pytest.fixture(scope="module", autouse=True)
def prepare_datasets():
    create_datasets(
        partitions=PARTITIONS,
        export_path=INPUT_PATH,
        truncate_pct=0.02,
    )
    yield
    shutil.rmtree(INPUT_PATH, ignore_errors=True)


def test_train_models():
    result_dir = train_models(
        train_dict=TEST_MODELS,
        feature_sets=FEATURE_SETS,
        model_configs=TEST_MODEL_CONFIGS,
        y_col="wait_time_seconds",
        test_size=0.4,
        split_method="timeseries",
        export_path=EXPORT_PATH,
        input_path=INPUT_PATH,
    )

    print(result_dir)
    # assert res
    #
    # print(
    #     res.df_accuracy_metrics.filter(pl.col("metric") == "perc_err_under_10min").sort(
    #         "result_name"
    #     )
    # )

    shutil.rmtree(EXPORT_PATH)
