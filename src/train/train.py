import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

import jsonpickle
import polars as pl
from joblib import dump, load

from input.input import get_partition
from train.regress import Result, predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def train_models(
    train_dict: dict,
    feature_sets: dict,
    y_col: str,
    test_size: float,
    truncate_pct: float,
    model_configs: Any,
    split_method: str,
    use_local_data: bool,
    export_path: str = "results/",
    input_path: str = "data/",
):
    """Trains multiple models across partitions and feature-sets.

    This function

    Args:
        train_dict: Denotes what models and feature-sets are considered.
            See `train.params.params_training`.
        feature_sets: Denotes features for each partition and feature-set.
            See
        model_configs: Denotes model-specifications.
        y_col: Target column, i.e. wait_time_seconds.
        test_size: How large portion of partition's data is used for validation.
        truncate_pct: How much raw data is truncated, useful for testing purposes.
        split_method: How the data is splitted into training and testing sets.
            Can be either `random` that uses `sklearn.train_test_split`, or
            `timeseries`, where data is split chronologically.

    Returns:
    """
    models = list({m for cfg in train_dict.values() for m in cfg["models"]})
    partitions = list(train_dict.keys())

    prefix = create_prefix(
        models=models,
        partitions=partitions,
    )

    logger.info(
        f"Starting training with {len(partitions)} partitions and {len(models)} models"
    )

    for partition, config in train_dict.items():
        logger.info(f"Processing partition: {partition}")

        if use_local_data:
            df = pl.read_parquet(f"{input_path}/{partition}.parquet")
        else:
            df = get_partition(
                partition=partition, type="with_features", truncate_pct=truncate_pct
            )

        logger.info(f"Loaded data for partition {partition}: {len(df)} rows")

        for model in config["models"]:
            logger.info(f"Training model: {model}")

            partition_ablation_sets = feature_sets.get(partition, {})

            for ablation_name in config["feature_sets"]:
                ablation_features = partition_ablation_sets[ablation_name]
                logger.info(
                    f"Ablation set: {ablation_name} ({len(ablation_features)} features)"
                )

                filename = f"{export_path}/{prefix}/res_{partition}_{model}_{ablation_name}.pkl"

                model_res = predict(
                    df,
                    y_col=y_col,
                    x_cols=ablation_features,
                    no_features=len(ablation_features),
                    test_size=test_size,
                    model=model,
                    model_configs=model_configs,
                    split_method=split_method,
                )

                os.makedirs(f"{export_path}/{prefix}/", exist_ok=True)
                dump(value=model_res, filename=filename)
                logger.info(f"Saved results to {filename}")

            logger.info(f"Training completed for {partition, model}")

    with open(f"{export_path}/{prefix}/features.json", "w") as fp:
        json.dump(feature_sets, fp)

    with open(f"{export_path}/{prefix}/model_parameters.json", "w") as f:
        f.write(jsonpickle.encode(model_configs, indent=2))

    res = combine_results(results_dir=f"{export_path}/{prefix}")

    return res


def recreate_dataset(
    results_dir: str,
    partition: str,
    use_local_data: bool,
    truncate_pct: float = 1.0,
    input_path: str = "data/",
):
    """Recreates the validation-set dataframe for a partition with all model
    predictions.

    Args:
        results_dir: Path to the directory containing serialized Result objects.
        partition: The partition name to reconstruct.
        use_local_data: Whether to load data from local parquet files.
        truncate_pct: Truncation percentage for remote data loading.

    Returns:
        A polars DataFrame containing the validation rows with y_pred columns
        from each model/ablation combination.
    """
    results_path = Path(results_dir)
    result_files = sorted(results_path.glob(f"res_{partition}_*.pkl"))

    if not result_files:
        raise FileNotFoundError(
            f"No result files found for partition '{partition}' in {results_dir}"
        )

    if use_local_data:
        df = pl.read_parquet(f"{input_path}/{partition}.parquet")
    else:
        df = get_partition(
            partition=partition, type="with_features", truncate_pct=truncate_pct
        )

    df = df.sort(pl.col("start_ts"), descending=False).with_row_index("__row_idx")

    df_validation = None

    for file in result_files:
        res = load(file)
        name = file.stem.replace(f"res_{partition}_", "")

        if df_validation is None:
            df_validation = df[res.validation_indices.tolist()]

        df_validation = df_validation.with_columns(
            pl.Series(f"y_pred_{name}", res.y_pred)
        )

    df_validation = df_validation.drop("__row_idx")

    return df_validation


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

    res = Result(
        df_feature_selection=combined_dfs.get("df_feature_selection"),
        df_cv_results=combined_dfs.get("df_cv_results"),
        df_feature_importance=combined_dfs.get("df_feature_importance"),
        df_accuracy_metrics=combined_dfs.get("df_accuracy_metrics"),
        y_pred=None,
        validation_indices=None,
        # df_validation=combined_dfs.get("df_validation"),
        best_model=None,
    )

    return res


def create_prefix(models: list, partitions: list):
    timestamp = str(datetime.datetime.now().strftime("%Y%m%dT%H%M"))
    models_string = "-".join([m for m in models])
    partitions_string = "-".join([p for p in partitions])
    prefix = models_string + "__" + partitions_string + "__" + timestamp
    return prefix
