from pathlib import Path

import polars as pl
from joblib import load

from train.regress import Result


def recreate_dataset(
    results_dir: str,
    partition: str,
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

    df = pl.read_parquet(f"{input_path}/{partition}.parquet")

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
        best_model=None,
    )

    return res
