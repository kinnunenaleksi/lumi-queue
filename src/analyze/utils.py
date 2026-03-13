import re
from pathlib import Path

import pandas as pd
import polars as pl
from joblib import load

from train.regress import Result


def combine_results(results_dir: str = "results/"):

    results_path = Path(results_dir)
    combined_results = {}

    for file in results_path.glob("*.xz"):
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


def format_accuracy_metrics(df: pl.DataFrame):
    return df.with_columns(
        pl.when(pl.col("metric").str.starts_with("perc"))
        .then((pl.col("value") * 100).round(2).cast(pl.Utf8) + "%")
        .otherwise(pl.col("value").round(2))
        .alias("value")
    )


def explode_result_name(df: pl.DataFrame, res_name: str = "result_name"):

    return (
        df.with_columns(
            pl.col(res_name)
            .str.replace("^res_", "")
            .str.split_exact("_", 2)
            .struct.rename_fields(["partition", "model", "feature_set"])
            .alias("parsed")
        )
        .unnest("parsed")
        .select(
            ["partition", "model", "feature_set"]
            + [c for c in df.columns if c != "result_name"]
        )
    )


def print_table(df: pl.DataFrame, partition: str, model: str = None):

    df = pd.DataFrame(df, columns=df.columns)
    md = df.to_markdown(index=False, numalign="left")
    typst = df.reset_index(drop=True).style.to_typst()

    typst = typst.replace("columns: 3,", "columns: 2,", 1)
    typst = typst.replace("[], [", "[", 1)
    typst = re.sub(r"\[\d+\],\s*", "", typst)

    if model is None:
        res = f"----- {partition} -----\n\n{md}\n\n{typst}\n\n"
    else:
        res = f"----- {partition}, {model} -----\n\n{md}\n\n{typst}\n\n"

    print(res)

    return res
