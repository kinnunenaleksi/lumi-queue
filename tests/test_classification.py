import shutil

import polars as pl

from analyze.analyze import create_reports
from analyze.utils import recreate_dataset
from input.input import create_datasets
from train.params.params_classification import TEST_CLASSIFIER_CONFIG
from train.params.params_features import FEATURE_SETS
from train.train import train_models

DATA_PATH = "../anonJobs.parquet"
INPUT_PATH = "test_data"
EXPORT_PATH = "test_classification_results"

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
        model_configs=TEST_CLASSIFIER_CONFIG,
        input_path=INPUT_PATH,
        export_path=EXPORT_PATH,
        # y_col="wait_time_seconds",
        y_col="wait_time_bin",
        test_size=0.4,
        split_method="random",
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
        results_dir=result_dir, compression="pkl", prediction_type="classification"
    )

    print(accuracy_results[0])
    print(cv_results[0])
    print(res.df_feature_importance.head())

    # print(df.select(pl.col("y_pred_rf_perfect")).to_series().value_counts())

    pred_vals = df.select(pl.col("y_pred_rf_perfect")).to_series().value_counts()
    true_vals = df.select(pl.col("wait_time_bin")).to_series().value_counts()
    df_join = pred_vals.join(
        true_vals, left_on="y_pred_rf_perfect", right_on="wait_time_bin"
    )
    print(df_join.sort("y_pred_rf_perfect"))

    print(df.select(pl.col("hour")).to_series().value_counts().sort("count").head(10))

    # print(df.columns)

    # shutil.rmtree(INPUT_PATH)
    # shutil.rmtree(EXPORT_PATH)
