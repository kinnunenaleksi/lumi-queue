from analyze.analyze import create_reports
from train.params.params_features import FEATURE_SETS
from train.params.params_models import MODEL_CONFIGS, TEST_MODEL_CONFIGS
from train.train import train_models

TEST_MODELS = {
    "small": {
        "models": ["rf", "xgb"],
        "feature_sets": ["full", "baseline", "naive", "minimal"],
    },
    "small-g": {
        "models": ["rf", "xgb"],
        "feature_sets": ["full", "baseline", "naive", "minimal"],
    },
    "standard": {
        "models": ["rf", "xgb"],
        "feature_sets": ["full", "baseline", "naive", "minimal"],
    },
    "standard-g": {
        "models": ["rf", "xgb"],
        "feature_sets": ["full", "baseline", "naive", "minimal"],
    },
}

INPUT_PATH = "filtered_data"
EXPORT_PATH = "filtered_model_results"
PARTITIONS = list(TEST_MODELS.keys())


def main():

    results_dir = train_models(
        train_dict=TEST_MODELS,
        feature_sets=FEATURE_SETS,
        model_configs=TEST_MODEL_CONFIGS,
        y_col="wait_time_seconds",
        test_size=0.2,
        split_method="random",
        export_path=EXPORT_PATH,
        input_path=INPUT_PATH,
    )

    _ = create_reports(
        results_dir=results_dir,
        input_path=INPUT_PATH,
        partitions=PARTITIONS,
        prediction_type="regression",
    )


if __name__ == "__main__":
    main()
