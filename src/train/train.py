from joblib import dump, load
import logging
import os
from pathlib import Path
import polars as pl
import datetime
import json
import jsonpickle

from typing import Any

from input.input import get_partition
from train.regress import predict, Result
# from train.model_params import MODEL_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def create_prefix(models: list, partitions: list):
    timestamp = str(datetime.datetime.now().strftime("%Y%m%dT%H%M"))
    models_string = "-".join([m for m in models])
    partitions_string = "-".join([p for p in partitions])
    prefix = models_string + "__" + partitions_string + "__" + timestamp
    return prefix


def train_models(
    train_dict: dict,
    feature_sets: dict,
    y_col: str,
    test_size: float,
    save_results: bool,
    truncate_pct: float,
    model_configs: Any,
    scaling_policy: str,
    log_transform_policy: str,
    split_method: str,
    search_method: str,
):

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
                    model_configs=model_configs,
                    scaling_policy=scaling_policy,
                    log_transform_policy=log_transform_policy,
                    split_method=split_method,
                    search_method=search_method,
                )

                if save_results:
                    os.makedirs(f"results/{prefix}/", exist_ok=True)
                    dump(value=model_res, filename=filename)
                    logger.info(f"Saved results to {filename}")
                else:
                    logger.info("Results are not saved.")

            logger.info(f"Training completed for {partition, model}")

    if save_results:
        with open(f"results/{prefix}/features.json", "w") as fp:
            json.dump(feature_sets, fp)

        with open(f"results/{prefix}/model_parameters.json", "w") as f:
            f.write(jsonpickle.encode(model_configs, indent=2))

        res = combine_results(results_dir=f"results/{prefix}")

        return res
    # else:
    #     pass


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
        df_validation=combined_dfs.get("df_validation"),
        best_model=None,
    )

    return res
