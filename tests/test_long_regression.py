import shutil

import polars as pl

from analyze.analyze import create_reports
from input.input import create_datasets
from train.params.params_features import FEATURE_SETS
from train.params.params_models import TEST_MODEL_CONFIGS
from train.regress import predict
from train.train import train_models

DATA_PATH = "../anonJobs.parquet"
INPUT_PATH = "long_data"
EXPORT_PATH = "long_results"

TEST_MODELS = {
    "standard-g": {
        "models": ["rf"],
        "feature_sets": ["full", "baseline", "naive", "minimal"],
    },
    # "small-g": {
    #     "models": ["rf"],
    #     "feature_sets": ["perfect", "system"],
    # },
}

PARTITIONS = list(TEST_MODELS.keys())


def test_pipeline():

    written_paths = create_datasets(
        partitions=PARTITIONS,
        export_path=INPUT_PATH,
        truncate_pct=0.05,
        data_path=DATA_PATH,
        filter_geq_minutes=10,
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

    _ = create_reports(
        results_dir=result_dir,
        input_path=INPUT_PATH,
        partitions=PARTITIONS,
        prediction_type="regression",
        compression="pkl",
    )

    shutil.rmtree(INPUT_PATH)
    shutil.rmtree(EXPORT_PATH)
