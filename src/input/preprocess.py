import polars as pl

from input.params import (
    NON_NA_COLUMNS,
    USED_PARTITIONS,
    column_dict,
)


def preprocess(df: pl.LazyFrame):
    """Main function of `preprocess`.

    This function collectively runs all required preprocessing operations for the data.

    Args:
        df: Raw Slurm output of the `sacct` utility.

    Returns: Preprocessed polars dataframe.
    """
    df_preprocess = (
        df.pipe(parse_input)
        .pipe(handle_cancellations)
        .pipe(add_cols)
        .pipe(add_target)
        .pipe(explode_col)
        .collect()
    )

    df_preprocess = fill_empty_allocs(df_preprocess)

    return df_preprocess


def parse_input(
    df: pl.LazyFrame,
    column_names: dict = column_dict,
    non_na_cols: list = NON_NA_COLUMNS,
    used_partitions: list = USED_PARTITIONS,
) -> pl.LazyFrame:
    """Removes problematic or redundant rows and renames colums."""
    df = (
        df.select(list(column_names.keys()))
        .rename(column_names)
        .unique()
        .filter(pl.col("partition").is_in(used_partitions))
        .drop_nulls(subset=[column_names.get(c, c) for c in non_na_cols])
        .sort("submit_ts")
    )

    return df


def handle_cancellations(df: pl.LazyFrame) -> pl.LazyFrame:
    """Removes the user-id from the cancel state."""
    df = df.with_columns(
        state=(
            pl.when(pl.col("state").str.starts_with("CANC"))
            .then(pl.lit("CANCELLED"))
            .otherwise(pl.col("state"))
        )
    )

    return df


def add_cols(df: pl.LazyFrame) -> pl.LazyFrame:
    """Adds auxillary rows to the data."""
    return df.with_columns(
        year_month=pl.col("start_ts").dt.strftime("%Y-%m").cast(pl.Utf8),
        year_month_id=pl.col("start_ts").dt.strftime("%Y%m").cast(pl.Int32),
        year_id=pl.col("start_ts").dt.strftime("%Y").cast(pl.Int32),
        day_id=pl.col("start_ts").dt.strftime("%Y%m%d").cast(pl.Int32),
        allocated_timelimit_seconds=pl.col("timelimit_minutes") * 60,
    )


def add_target(df: pl.LazyFrame) -> pl.LazyFrame:
    """Adds the target variable to the data."""
    return df.with_columns(
        wait_time_seconds=(
            pl.col("start_ts") - pl.col("eligible_start_ts")
        ).dt.total_seconds()
    ).filter(pl.col("wait_time_seconds") >= 0)


def fill_empty_allocs(df: pl.DataFrame):
    """Fills empty columns with prefix `allocated_`.

    The function `parse_input` already drops rows where the original `AllocTRES` field
    is empty, but for each of the partitions there are a few jobs (very very small
    partion of the total) jobs, that are nulls. After quick diagnostics, the missing
    values are the nodes. For all of these jobs (<10), replacing the null with 1 is
    sufficient, but this could be made more robust by creating a specific dictionary,
    where one could derive the actual amount of memory / nodes used from the CPU usage
    if null in the data.
    """
    cols = [pl.col(c) for c in df.columns if c.startswith("allocated_")]
    return df.with_columns([c.fill_null(1).fill_nan(1) for c in cols])


def explode_col(
    df: pl.LazyFrame, col_name: str = "allocated_resources"
) -> pl.LazyFrame:
    """Explodes the `AllocTRES` column into individual columns.

    Furthermore, this function offers functionality to normalize both the memory and
    GPU usage. In particular, the memory can be with either M, G, or T. Currently,
    this function normalizes the values into Gigabytes.

    The GPU counts could also be altered from here, in case for example the a40 GPU
    would equal 0.5 a100 GPUs. At this stage, all GPUs are left equal.

    Args:
        df: Raw slurm output with renamed columns.
        col_name: The column to be exploded. Can be either `allocated_resources` or
            `required_resources`.

    Returns:
        Dataframe with exploded resources.
    """
    df_with_index = df.with_row_index("row_idx")

    long = (
        df_with_index.select("row_idx", pl.col(col_name).str.split(",").alias(col_name))
        .explode(col_name)
        .with_columns(kv=pl.col(col_name).str.splitn("=", 2))
        .with_columns(
            key=pl.col("kv").struct.field("field_0"),
            value=pl.col("kv").struct.field("field_1"),
        )
        .select("row_idx", "key", "value")
        .with_columns(num=(pl.col("value").str.strip_chars_end("GMT").cast(pl.Float64)))
        .with_columns(
            val=(
                # Terabytes to Gigabytes
                pl.when(pl.col("value").str.ends_with("T"))
                .then(pl.col("num") * 1024.0)
                # Megabytes to Gigabytes
                .when(pl.col("value").str.ends_with("M"))
                .then(pl.col("num") / 1024.0)
                # If older GPU, resources smaller
                .when(pl.col("key").str.ends_with("a40"))
                .then(pl.col("num") * 1)
                .when(pl.col("key").str.ends_with("a100"))
                .then(pl.col("num") * 1)
                .when(pl.col("key").str.ends_with("nvme"))
                .then(pl.col("num") * 1)
                .otherwise(pl.col("num"))
            )
        )
        .with_columns(
            pl.when(pl.col("key").str.starts_with("gres"))
            .then(pl.lit("gpu"))
            .otherwise(pl.col("key"))
            .alias("key")
        )
        .with_columns(
            key=pl.concat_str(
                [
                    pl.when(pl.col("key").is_in(["energy", "billing"]))
                    .then(pl.lit("consumed_"))
                    .otherwise(pl.lit("allocated_")),
                    pl.col("key"),
                ]
            )
        )
    )

    idx = pl.col("row_idx")
    columns = pl.col("key")
    values = pl.col("val")
    unique_keys = long.select(pl.col("key").unique()).collect().to_series().to_list()
    unique_keys = [k for k in unique_keys if k is not None]
    agg_func = lambda col: col.first()

    wide = long.group_by(idx).agg(
        [
            agg_func(values.filter(columns.eq_missing(value))).alias(value)
            for value in unique_keys
        ]
    )

    df_with_index = df_with_index.join(wide, on="row_idx")
    df_with_index = df_with_index.drop(col_name, "row_idx")

    return df_with_index
