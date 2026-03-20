from analyze.analyze import create_reports
from train.params.params_features import FEATURE_SETS
from train.params.params_models import BASELINE_CONFIGS
from train.train import train_models

TEST_MODELS = {
    # "small-g": {
    #     "models": ["rf", "xgb"],
    #     "feature_sets": ["perfect", "baseline"],
    # },
    # # "small": {
    # #     "models": ["rf", "xgb"],
    # #     "feature_sets": ["perfect", "baseline"],
    # # },
    # "standard": {
    #     "models": ["rf", "xgb"],
    #     "feature_sets": ["baseline", "perfect"],
    # },
    # "standard-g": {
    #     "models": ["rf", "xgb"],
    #     "feature_sets": ["baseline", "perfect"],
    # },
    # "small": {
    #     "models": ["rf", "xgb"],
    #     "feature_sets": ["baseline"],
    # },
    # "standard": {
    #     # "models": ["rf", "xgb"],
    #     # "models": ["mlp"],
    #     "models": ["rf", "xgb", "mlp"],
    #     "feature_sets": ["baseline"],
    # },
    "small-g": {
        # "models": ["rf", "xgb"],
        # "models": ["mlp"],
        "models": ["rf", "xgb", "mlp"],
        "feature_sets": ["baseline"],
    },
}

INPUT_PATH = "data"
EXPORT_PATH = "test_model_results"
PARTITIONS = list(TEST_MODELS.keys())


def main():

    results_dir = train_models(
        train_dict=TEST_MODELS,
        feature_sets=FEATURE_SETS,
        model_configs=BASELINE_CONFIGS,
        y_col="wait_time_seconds",
        test_size=0.3,
        split_method="timeseries",
        export_path=EXPORT_PATH,
        input_path=INPUT_PATH,
    )

    _, _, _ = create_reports(results_dir=results_dir)


if __name__ == "__main__":
    main()
