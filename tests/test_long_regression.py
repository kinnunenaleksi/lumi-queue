import shutil

import polars as pl

from analyze.analyze import create_reports
from analyze.utils import recreate_dataset
from input.input import create_datasets
from train.params.params_features import FEATURE_SETS
from train.params.params_models import TEST_MODEL_CONFIGS
from train.regress import predict
from train.train import train_models

DATA_PATH = "../anonJobs.parquet"
INPUT_PATH = "long_data"
EXPORT_PATH = "long_results"

TEST_MODELS = {
    "standard": {
        "models": ["rf", "xgb"],
        "feature_sets": ["perfect", "system", "baseline", "naive"],
    },
}

PARTITIONS = list(TEST_MODELS.keys())


def test_pipeline():

    written_paths = create_datasets(
        partitions=PARTITIONS,
        export_path=INPUT_PATH,
        truncate_pct=0.8,
        data_path=DATA_PATH,
        filter=True,
    )

    result_dir = train_models(
        train_dict=TEST_MODELS,
        feature_sets=FEATURE_SETS,
        model_configs=TEST_MODEL_CONFIGS,
        input_path=INPUT_PATH,
        export_path=EXPORT_PATH,
        y_col="wait_time_seconds",
        test_size=0.2,
        split_method="random",
        compression="pkl",
    )

    df = recreate_dataset(
        results_dir=result_dir,
        partition="standard",
        input_path=INPUT_PATH,
        compression="pkl",
    )

    print(df.shape[0])

    print(df.select(pl.col("wait_time_seconds", "wait_time_minutes")).describe())

    print(df.columns)

    res, accuracy_results, cv_results, feature_importance_results = create_reports(
        results_dir=result_dir, compression="pkl", prediction_type="regression"
    )

    print(accuracy_results[0])
    print(cv_results[0])
    print(res.df_feature_importance.head())
    print(feature_importance_results[0])

    shutil.rmtree(INPUT_PATH)
    shutil.rmtree(EXPORT_PATH)
