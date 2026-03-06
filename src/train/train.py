from joblib import dump, load
import logging
import os
from pathlib import Path
import polars as pl
import datetime

from input.input import get_partition
from train.regress import predict
from train.train_params import (
    PARTITION_LIST_TEST,
    MODELS_LIST_TEST,
    FEATURE_SETS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def create_prefix(
    models: list = MODELS_LIST_TEST, partitions: list = PARTITION_LIST_TEST
):
    timestamp = str(datetime.datetime.now().strftime("%Y%m%dT%H%M"))
    models_string = "-".join([m for m in models])
    partitions_string = "-".join([p for p in partitions])
    prefix = models_string + "." + partitions_string + "." + timestamp
    return prefix


def train_models(
    partitions: list = PARTITION_LIST_TEST,
    models: list = MODELS_LIST_TEST,
    feature_sets: dict = FEATURE_SETS,
    y_col: str = "wait_time_seconds",
    test_size: float = 0.3,
):

    # timestamp = str(datetime.datetime.now().date())
    prefix = create_prefix()
    os.makedirs(f"results/{prefix}/", exist_ok=True)

    logger.info(
        f"Starting training with {len(partitions)} partitions and {len(models)} models"
    )

    for partition in partitions:
        logger.info(f"Processing partition: {partition}")

        df = get_partition(partition=partition, type="with_features")
        logger.info(f"Loaded data for partition {partition}: {len(df)} rows")

        for model in models:
            logger.info(f"Training model: {model}")

            partition_ablation_sets = feature_sets.get(partition, {})

            for ablation_name, ablation_features in partition_ablation_sets.items():
                logger.info(
                    f"Ablation set: {ablation_name} ({len(ablation_features)} features)"
                )

                filename = (
                    f"results/{prefix}/res_{partition}_{model}_{ablation_name}.pkl"
                )

                model_res = predict(
                    df,
                    y_col=y_col,
                    x_cols=ablation_features,
                    no_features=len(ablation_features),
                    test_size=test_size,
                    model=model,
                )

                dump(value=model_res, filename=filename)
                logger.info(f"Saved results to {filename}")
            logger.info(f"Training completed for {partition, model}")

        # combine_results()


def combine_results(results_dir: str = "results/"):

    results_path = Path(results_dir)
    combined_results = {}

    for file in results_path.glob("*.pkl"):
        name = file.stem
        combined_results[name] = load(file)

    buckets: dict[str, list[pl.DataFrame]] = {}

    for name, res in combined_results.items():
        for attr, val in vars(res).items():
            if isinstance(val, pl.DataFrame):
                buckets.setdefault(attr, []).append(
                    val.with_columns(pl.lit(name).alias("result_name"))
                )

    combined_dfs = {
        attr: pl.concat(dfs, how="diagonal_relaxed") for attr, dfs in buckets.items()
    }

    return combined_results, combined_dfs
