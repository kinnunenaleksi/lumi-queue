from collections.abc import Mapping

import pandas as pd
import polars as pl

from analyze.utils import (
    combine_results,
    evaluate_by_wait_time_bins,
    explode_res_name,
    explode_result_name,
    format_accuracy_metrics,
    print_table,
    recreate_dataset,
)


def create_reports(
    results_dir: str,
    input_path: str,
    partitions: list,
    prediction_type: str,
    compression: str = "xz",
):
    """Main function of `analyze` module. Creates combined model-training results.

    This function creates the following files for reporting purposes:

        `accuracy_results.txt`: Combined accuracy results.
        `cv_results.txt`: Combined cross-validation results.

    Args:
        results_dir: Directory where with the training-results.
    """
    res = combine_results(results_dir=results_dir, compression=compression)

    accuracy_results = create_accuracy_report(
        df_accuracy_metrics=res.df_accuracy_metrics,
        output_path=f"{results_dir}/accuracy_results.txt",
    )

    cv_results = create_cv_report(
        res.df_cv_results,
        output_path=f"{results_dir}/cv_results.txt",
        prediction_type=prediction_type,
    )

    feature_results = create_combined_feature_importance(
        df=res.df_feature_importance,
        output_path=f"{results_dir}/feature_importance.txt",
    )

    for partition in partitions:
        gran_results = []
        granular_results = create_granular_report(
            results_dir=results_dir,
            partition=partition,
            input_path=input_path,
            compression=compression,
            metric="perc_err_under_10min",
            output_path=f"{results_dir}/{partition}_granular_report.txt",
            bins=[0, 30, 60, 120, 240],
        )
        gran_results.append(granular_results)

    return (res, accuracy_results, cv_results, feature_results, gran_results)


def create_granular_report(
    results_dir,
    partition,
    input_path,
    compression: str,
    metric: str,
    output_path: str,
    bins: list = [0, 30, 60, 120, 240],
):

    df_recollect = recreate_dataset(
        results_dir=results_dir,
        partition=partition,
        input_path=input_path,
        compression=compression,
    )

    cols = [c for c in df_recollect.columns if c.startswith("y_pred")]

    dfs = []

    for res in cols:
        df_ev = evaluate_by_wait_time_bins(df_recollect, y_pred_col=res, bins=bins)
        df_ev_metric = df_ev.filter(pl.col("metric") == metric)
        df_ev_metric[0, 0] = res
        dfs.append(df_ev_metric)

    df_concat = pl.concat(dfs)
    # df_concat = df_concat.with_columns(pl.lit(partition).alias("partition"))

    df_concat = explode_res_name(df_concat)

    # partitions = df_concat.select(pl.col("partition")).unique().to_series()
    models = df_concat.select(pl.col("model")).unique().to_series()

    res_list = []
    # for partition in partitions:
    for model in models:
        df_filter = df_concat.filter(pl.col("model") == model)

        df_pd = pd.DataFrame(df_filter, columns=df_filter.columns)
        df_pd = df_pd.drop(columns="model")

        res = print_table(df_pd, partition=partition, model=model)
        res_list.append(res)

    with open(output_path, "w") as f:
        f.write("\n".join(res_list))

    return res


def create_accuracy_report(
    df_accuracy_metrics: pl.DataFrame,
    output_path: str = "accuracy_metrics.txt",
):
    """Creates accuracy report."""
    df_accuracy_metrics = format_accuracy_metrics(df_accuracy_metrics)
    df_accuracy_metrics = explode_result_name(df_accuracy_metrics)

    dimension_cols = ["partition", "model", "feature_set"]

    unique_values = {
        col: df_accuracy_metrics.select(pl.col(col)).unique().to_series().to_list()
        for col in dimension_cols
    }

    doc_list = []

    for pivot_col in dimension_cols:
        filter_col_names = [c for c in dimension_cols if c != pivot_col]

        combos = [
            dict(zip(filter_col_names, vals))
            for vals in _product(*(unique_values[c] for c in filter_col_names))
        ]

        for filter_cols in combos:
            filtered = df_accuracy_metrics.filter(
                pl.all_horizontal(pl.col(c) == v for c, v in filter_cols.items())
            )
            if filtered.is_empty():
                continue

            _, res = get_comparison_results(
                df_accuracy_metrics,
                filter_cols=filter_cols,
                pivot_col=pivot_col,
            )
            doc_list.append(res)

    with open(output_path, "w") as f:
        f.write("\n".join(doc_list))

    return doc_list


def create_cv_report(
    df: pl.DataFrame,
    prediction_type: str,
    output_path: str = "cv_results.txt",
):
    """Creates cross-validation report."""
    df = explode_result_name(df)

    rm_cols = [
        c
        for c in df.columns
        if c.startswith("split")
        or c.startswith("std")
        or c.startswith("mean_score")
        or c.startswith("param")
        # regression
        or c.startswith("rank_test_mape")
        or c.startswith("rank_test_rmse")
        or c.startswith("rank_test_r2")
        # classification
        or c.startswith("rank_test_balanced_accuracy")
        or c.startswith("rank_test_roc_auc")
    ]

    df = df.drop(rm_cols)

    partitions = df.select(pl.col("partition")).unique().to_series().to_list()
    models = df.select(pl.col("model")).unique().to_series().to_list()

    res_list = []

    for partition in partitions:
        for model in models:
            df_filter = df.filter(
                (pl.col("partition") == partition) & (pl.col("model") == model)
            )

            df_filter = df_filter.drop(["partition", "model"])

            cols_without_nulls = [
                c for c in df_filter.columns if df_filter[c].null_count() == 0
            ]

            df_filter = df_filter.select(cols_without_nulls)

            if prediction_type == "regression":
                first_cols = [
                    "feature_set",
                    "mean_test_mae",
                    "mean_test_rmse",
                    "mean_test_r2",
                    "mean_test_mape",
                    "rank_test_mae",
                ]

                sort_cols = ["feature_set", "rank_test_mae"]

            elif prediction_type == "classification":
                first_cols = [
                    "feature_set",
                    "mean_test_f1_macro",
                    "mean_test_balanced_accuracy",
                    "mean_test_roc_auc",
                    "rank_test_f1_macro",
                ]

                sort_cols = ["feature_set", "rank_test_f1_macro"]

            else:
                raise ValueError("prediction type must be in ")

            df_filter = df_filter.select(
                first_cols + [c for c in df_filter.columns if c not in first_cols]
            )

            df_filter = df_filter.sort(sort_cols, descending=False)

            df_pd = pd.DataFrame(df_filter, columns=df_filter.columns)

            if df_pd.shape[0] > 0:
                res = print_table(df_pd, partition=partition, model=model)

                res_list.append(res)

    with open(output_path, "w") as f:
        f.write("\n".join(res_list))

    return res_list


def get_comparison_results(
    df: pl.DataFrame,
    filter_cols: Mapping[str, str],
    pivot_col: str,
):
    """Auxillary function for `create_accuracy_report`."""
    filter_col_names = list(filter_cols.keys())

    df = df.filter(
        pl.all_horizontal(pl.col(c) == v for c, v in filter_cols.items())
    ).drop(filter_col_names)

    df = df.pivot(values="value", on=pivot_col)

    label = (
        " | ".join(f"{c}={v}" for c, v in filter_cols.items())
        + f" | comparing: {pivot_col}"
    )
    res = print_table(df, partition=label)

    return (df, res)


def create_combined_feature_importance(
    df: pl.DataFrame, output_path: str
) -> pl.DataFrame:
    df = explode_result_name(df)
    # Iterate only existing partition-model combinations
    pairs = df.select(["partition", "model"]).unique().iter_rows(named=True)
    # Least valuable -> most valuable
    set_priority = ["minimal", "naive", "baseline", "full"]
    res_list = []
    for pair in pairs:
        partition = pair["partition"]
        model = pair["model"]
        df_filter = df.filter(
            (pl.col("partition") == partition) & (pl.col("model") == model)
        )
        if df_filter.is_empty():
            continue
        df_res = (
            df_filter.with_columns(
                (
                    pl.col("value")
                    / pl.col("value").sum().over(["feature_set", "model"])
                ).alias("pct_contribution")
            )
            .select(["feature", "feature_set", "pct_contribution"])
            .pivot(
                values="pct_contribution",
                index="feature",
                on="feature_set",
                aggregate_function="first",
            )
            .fill_null(0.0)
        )
        # Keep only sets that exist for this partition/model
        present_sets = [c for c in set_priority if c in df_res.columns]
        # Sort rows by least valuable set first (fallbacks naturally if missing)
        if present_sets:
            df_res = df_res.sort(by=present_sets, descending=[True] * len(present_sets))
        # Keep output columns in weak->strong order
        df_res = df_res.select(["feature"] + present_sets)
        df_pd = pd.DataFrame(df_res, columns=df_res.columns)
        if df_pd.shape[0] > 0:
            res = print_table(df_pd, partition=partition, model=model)
            res_list.append(res)
    with open(output_path, "w") as f:
        f.write("\n".join(res_list))
    return res_list


# def create_combined_feature_importance(
#     df: pl.DataFrame, output_path: str
# ) -> pl.DataFrame:
#
#     df = explode_result_name(df)
#
#     partitions = df.select(pl.col("partition")).unique().to_series().to_list()
#     models = df.select(pl.col("model")).unique().to_series().to_list()
#
#     res_list = []
#
#     for partition in partitions:
#         for model in models:
#             df_filter = df.filter(
#                 (pl.col("partition") == partition) & (pl.col("model") == model)
#             )
#             df_res = (
#                 df_filter.with_columns(
#                     (
#                         pl.col("value")
#                         / pl.col("value").sum().over(["feature_set", "model"])
#                     ).alias("pct_contribution")
#                 )
#                 .select(["feature", "feature_set", "pct_contribution"])
#                 .pivot(
#                     values="pct_contribution",
#                     index="feature",
#                     on="feature_set",
#                     aggregate_function="first",
#                 )
#                 .fill_null(0.0)
#                 .sort(["minimal", "naive", "baseline", "full"], descending=True)
#             )
#
#             # df_res = df_res.drop(["model"])
#
#             df_res = df_res.select(
#                 pl.col(
#                     [
#                         "feature",
#                         # "model",
#                         # "feature_set",
#                         "full",
#                         "baseline",
#                         "naive",
#                         "minimal",
#                     ]
#                 )
#             )
#             df_pd = pd.DataFrame(df_res, columns=df_res.columns)
#
#             if df_pd.shape[0] > 0:
#                 res = print_table(df_pd, partition=partition, model=model)
#
#                 res_list.append(res)
#
#     with open(output_path, "w") as f:
#         f.write("\n".join(res_list))
#
#     return res_list


def _product(*iterables):
    result = [()]
    for pool in iterables:
        result = [x + (y,) for x in result for y in pool]
    return result
