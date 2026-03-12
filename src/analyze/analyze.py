from collections.abc import Mapping

import polars as pl

from analyze.utils import combine_results, explode_result_name, format_accuracy_metrics


def create_reports(results_dir: str):
    """Creates combined model-training results.

    Args:
        results_dir: Directory where with the training-results.
    """
    res = combine_results(results_dir=results_dir)

    accuracy_results = create_accuracy_report(
        df_accuracy_metrics=res.df_accuracy_metrics,
        output_path=f"{results_dir}/accuracy_results.txt",
    )

    cv_results = create_cv_report(
        res.df_cv_results, output_path=f"{results_dir}/cv_results.txt"
    )

    return res, accuracy_results, cv_results


def create_accuracy_report(
    df_accuracy_metrics: pl.DataFrame,
    output_path: str = "tables.txt",
):
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


def _product(*iterables):
    result = [()]
    for pool in iterables:
        result = [x + (y,) for x in result for y in pool]
    return result


def create_cv_report(df: pl.DataFrame, output_path: str = "cv_results.txt"):

    df = explode_result_name(df)

    rm_cols = [
        c
        for c in df.columns
        if c.startswith("split")
        or c.startswith("std")
        or c.startswith("mean_score")
        or c.startswith("param")
        or c.startswith("mean_fit_time")
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

            first_cols = ["feature_set", "mean_test_score", "rank_test_score"]
            df_filter = df_filter.select(
                first_cols + [c for c in df_filter.columns if c not in first_cols]
            )
            df_filter = df_filter.sort(
                ["feature_set", "rank_test_score"], descending=False
            )

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
