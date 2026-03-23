import shutil

import polars as pl

from analyze.analyze import create_reports
from analyze.utils import recreate_dataset
from input.input import create_datasets
from train.params.params_features import FEATURE_SETS
from train.params.params_models import TEST_MODEL_CONFIGS
from train.train import train_models

DATA_PATH = "../anonJobs.parquet"
INPUT_PATH = "test_data"
EXPORT_PATH = "test_results"

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


def test_pipeline():

    written_paths = create_datasets(
        partitions=PARTITIONS,
        export_path=INPUT_PATH,
        truncate_pct=0.02,
        data_path=DATA_PATH,
    )

    assert written_paths == [
        f"{INPUT_PATH}/small-g.parquet",
        f"{INPUT_PATH}/standard.parquet",
    ]

    result_dir = train_models(
        train_dict=TEST_MODELS,
        feature_sets=FEATURE_SETS,
        model_configs=TEST_MODEL_CONFIGS,
        input_path=INPUT_PATH,
        export_path=EXPORT_PATH,
        # y_col="wait_time_seconds",
        y_col="wait_time_seconds",
        test_size=0.4,
        split_method="timeseries",
        compression="pkl",
    )

    df = recreate_dataset(
        results_dir=result_dir,
        partition="standard",
        input_path=INPUT_PATH,
        compression="pkl",
    )

    print(df.select(pl.col("wait_time_seconds", "wait_time_minutes")).describe())

    print(df.columns)

    res, accuracy_results, cv_results = create_reports(
        results_dir=result_dir, compression="pkl"
    )

    print(accuracy_results[0])
    print(cv_results[0])
    print(res.df_feature_importance.head())

    # shutil.rmtree(INPUT_PATH)
    # shutil.rmtree(EXPORT_PATH)
