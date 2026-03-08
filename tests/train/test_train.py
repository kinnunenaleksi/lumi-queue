import polars as pl
from train.train import train_models
from train.params_training import TEST_MODELS
from train.params_models import TEST_MODEL_CONFIGS
from train.params_features import FEATURE_SETS


def test_train_models():
    res = train_models(
        train_dict=TEST_MODELS,
        feature_sets=FEATURE_SETS,
        model_configs=TEST_MODEL_CONFIGS,
        y_col="wait_time_seconds",
        test_size=0.4,
        truncate_pct=0.02,
        scaling_policy="none",
        log_transform_policy="all_variables",
        split_method="timeseries",
        search_method="random",
    )

    assert res

    print(
        res.df_accuracy_metrics.filter(pl.col("metric") == "perc_err_under_10min").sort(
            "result_name"
        )
    )
