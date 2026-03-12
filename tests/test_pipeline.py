import shutil

from analyze.analyze import create_reports
from input.input import create_datasets
from train.params.params_features import FEATURE_SETS
from train.params.params_models import TEST_MODEL_CONFIGS
from train.train import train_models

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
        partitions=PARTITIONS, export_path=INPUT_PATH, truncate_pct=0.02
    )

    assert written_paths == [
        f"{INPUT_PATH}/small-g.parquet",
        f"{INPUT_PATH}/standard.parquet",
    ]

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

    res, accuracy_results, cv_results = create_reports(results_dir=result_dir)

    print(accuracy_results[0])
    print(cv_results[0])
    print(res.df_feature_importance.head())

    # shutil.rmtree(INPUT_PATH)
    # shutil.rmtree(EXPORT_PATH)
