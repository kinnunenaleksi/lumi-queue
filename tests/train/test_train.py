import polars as pl
from train.train import train_models
from train.train_params import FEATURE_SETS
from train.model_params import TEST_MODEL_CONFIGS


def test_train_models():
    res = train_models(
        # partitions=["largemem", "lumid"],
        # partitions=["lumid"],
        partitions=["small-g", "standard"],
        models=["rf", "xgb"],
        feature_sets=FEATURE_SETS,
        y_col="wait_time_seconds",
        test_size=0.4,
        save_results=True,
        truncate_pct=0.02,
        model_configs=TEST_MODEL_CONFIGS,
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
