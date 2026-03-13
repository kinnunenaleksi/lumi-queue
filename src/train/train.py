import datetime
import json
import logging
import os
from typing import Any

import jsonpickle
import polars as pl
from joblib import dump

from train.regress import predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def train_models(
    train_dict: dict,
    feature_sets: dict,
    model_configs: Any,
    y_col: str,
    test_size: float,
    split_method: str,
    export_path: str = "model_results/",
    input_path: str = "data/",
):
    """Main function of `train` module. Trains multiple models across
    partitions and feature-sets.

    This function loops over the `train.regress.predict` function for determined
    partitions, models, and feature-sets.

    Args:
        train_dict: Denotes what models and feature-sets are considered.
            See `train.params.params_training`.
        feature_sets: Denotes features for each partition and feature-set.
            See
        model_configs: Denotes model-specifications.
        y_col: Target column, i.e. wait_time_seconds.
        test_size: How large portion of partition's data is used for validation.
        split_method: How the data is splitted into training and testing sets.
            Can be either `random` that uses `sklearn.train_test_split`, or
            `timeseries`, where data is split chronologically.
        input_path: Relative path to the folder where preprocessed datasets are.
        export_path: Relative path to the folder in which to store model results.
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

    written_paths = []

    for partition, config in train_dict.items():
        logger.info(f"Processing partition: {partition}")

        df = pl.read_parquet(f"{input_path}/{partition}.parquet")

        logger.info(f"Loaded data for partition {partition}: {len(df)} rows")

        for model in config["models"]:
            logger.info(f"Training model: {model}")

            partition_ablation_sets = feature_sets.get(partition, {})

            for ablation_name in config["feature_sets"]:
                ablation_features = partition_ablation_sets[ablation_name]
                logger.info(
                    f"Ablation set: {ablation_name} ({len(ablation_features)} features)"
                )

                model_name = f"res_{partition}_{model}_{ablation_name}.xz"
                filename = f"{export_path}/{prefix}/{model_name}"

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

                written_paths.append(model_name)

            logger.info(f"Training completed for {partition, model}")

    with open(f"{export_path}/{prefix}/features.json", "w") as fp:
        json.dump(feature_sets, fp)

    with open(f"{export_path}/{prefix}/model_parameters.json", "w") as f:
        f.write(jsonpickle.encode(model_configs, indent=2))

    result_dir = f"{export_path}/{prefix}/"
    return result_dir


def create_prefix(models: list, partitions: list):
    """Creates the folder name for the training-results.

    The prefix is in the format:

    ```bash
    model1-model2__partition1-partition2__YYYYMMDDTHHMM
    ```

    Args:
        models: List of models that are trained.
        partitions: List of partitions the models are trained for.

    Returns:
        String for the folder-name.
    """
    timestamp = str(datetime.datetime.now().strftime("%Y%m%dT%H%M"))
    models_string = "-".join([m for m in models])
    partitions_string = "-".join([p for p in partitions])
    prefix = models_string + "__" + partitions_string + "__" + timestamp
    return prefix
