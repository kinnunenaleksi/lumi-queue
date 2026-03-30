from analyze.analyze import create_reports
from train.params.params_features import FEATURE_SETS
from train.params.params_models import ABLATION_CONFIG
from train.train import train_models

TEST_MODELS = {
    "standard": {
        "models": ["xgb", "rf"],
        "feature_sets": ["full", "baseline", "naive", "minimal"],
    },
}

INPUT_PATH = "data"
EXPORT_PATH = "regression_results"
PARTITIONS = list(TEST_MODELS.keys())


def main():

    results_dir = train_models(
        train_dict=TEST_MODELS,
        feature_sets=FEATURE_SETS,
        model_configs=ABLATION_CONFIG,
        y_col="wait_time_seconds",
        test_size=0.2,
        split_method="random",
        export_path=EXPORT_PATH,
        input_path=INPUT_PATH,
    )

    _ = create_reports(
        results_dir=results_dir,
        prediction_type="regression",
        input_path=INPUT_PATH,
        partitions=PARTITIONS,
    )


if __name__ == "__main__":
    main()
