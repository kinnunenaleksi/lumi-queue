import polars as pl

from src.input.params import (
    INVALID_STATES,
    NON_NA_COLUMNS,
    USED_PARTITIONS,
    column_dict,
)


def preprocess(df: pl.LazyFrame):

    df_preprocess = (
        df.pipe(parse_input)
        .pipe(handle_cancellations)
        .pipe(add_dates)
        .pipe(add_target)
        .pipe(explode_col)
        .collect()
    )

    df_preprocess = fill_empty_allocs(df_preprocess)
    df_final = clip_allocated_columns(df_preprocess)

    return df_final


def fill_empty_allocs(df: pl.DataFrame):
    cols = [pl.col(c) for c in df.columns if c.startswith("allocated_")]
    return df.with_columns([c.fill_null(0).fill_nan(0) for c in cols])


def clip_allocated_columns(
    df: pl.DataFrame,
    prefix: str = "allocated_",
    percentile: float = 0.9999999,  # prev 0.9995
) -> pl.DataFrame:

    cols = [c for c in df.columns if c.startswith(prefix)]

    pvals = df.select(
        [pl.col(c).quantile(percentile).alias(c) for c in cols]
    ).to_dicts()[0]

    return df.with_columns(
        [pl.col(c).clip(lower_bound=0, upper_bound=pvals[c]).alias(c) for c in cols]
    )


def parse_input(
    df: pl.LazyFrame,
    column_names: dict = column_dict,
    invalid_states: list = INVALID_STATES,
    non_na_cols: list = NON_NA_COLUMNS,
    used_partitions: list = USED_PARTITIONS,
) -> pl.LazyFrame:

    df = (
        df.select(list(column_names.keys()))
        .rename(column_names)
        .unique()
        .filter(~pl.col("state").is_in(invalid_states))
        .filter(pl.col("partition").is_in(used_partitions))
        .drop_nulls(subset=[column_names.get(c, c) for c in non_na_cols])
        .sort("submit_ts")
    )

    return df


def handle_cancellations(df: pl.LazyFrame) -> pl.LazyFrame:
    df = df.with_columns(
        state=(
            pl.when(pl.col("state").str.starts_with("CANC"))
            .then(pl.lit("CANCELLED"))
            .otherwise(pl.col("state"))
        )
    )

    return df


def add_dates(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        # year_month_id=pl.col("start_ts").dt.strftime("%Y%m").cast(pl.Int32),
        year_month_id=pl.col("start_ts").dt.strftime("%Y-%m").cast(pl.Utf8),
        year=pl.col("start_ts").dt.strftime("%Y").cast(pl.Int32),
    )


def add_target(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        wait_time_seconds=(
            pl.col("start_ts") - pl.col("eligible_start_ts")
        ).dt.total_seconds()
    ).filter(pl.col("wait_time_seconds") >= 0)


def explode_col(df, col_name: str = "allocated_resources"):

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
                .then(pl.col("num") / 1024.0)
                # Megabytes to Gigabytes
                .when(pl.col("value").str.ends_with("M"))
                .then(pl.col("num") * 1024.0)
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
        .with_columns(key=pl.concat_str([pl.lit("allocated_"), pl.col("key")]))
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
