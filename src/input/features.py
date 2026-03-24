import numpy as np
import polars as pl


def add_features(df: pl.DataFrame, partition: str) -> pl.DataFrame:
    """Main function of `features`.

    This function adds all the required features into the preprocessed dataframe. When
    using this function, the dataframe has to be filtered into a specific partition,
    because the `add_helper_allocated_flags` function dynamically calculates job sizes,
    which are then used by `add_active_allocs` and `add_queued_allocs` functions.

    The dynamic job size categorization is done with the number of nodes for the
    partitions `standard` and `standard-g`, as they are node- allocatable partitions.
    Similarly, the cpu count is used for all other partitions, as they are resource-
    allocatable partitions. they are resource- allocatable partitions. they are resource-
    allocatable partitions.

    Args:
        df: Preprocessed dataframe filtered into a single partition. See `input.preproces`
            for further details.
        partition (str): The partition the data is filtered into.

    Returns:
        Preprocessed dataframe with added features.
    """
    if partition in ["standard", "standard-g"]:
        denoting_column = "allocated_node"
    else:
        denoting_column = "allocated_cpu"

    helper_columns, df = add_helper_allocated_flags(df, denoting_column=denoting_column)
    resource_cols = [col for col in df.columns if col.startswith("allocated_")]

    # df1 = add_active_allocs(df, resource_cols=resource_cols)
    df1 = add_active_allocs_with_remaining_timelimit(df, resource_cols=resource_cols)
    df2 = add_queued_allocs(df1, resource_cols=resource_cols)
    df2 = add_reservation_flag(df2)
    df2 = add_chained_flag(df2)
    df2 = add_temporal_features(df2)
    df2 = df2.drop(helper_columns)
    return df2


def add_reservation_flag(df: pl.DataFrame):
    """Adds binary flag whether job has reservation."""
    return df.with_columns(
        pl.when(pl.col("reservation_id") > 0)
        .then(1)
        .otherwise(0)
        .alias("reservation_flag")
    )


def add_chained_flag(df: pl.DataFrame):
    """Adds binary flag whether job was chained."""
    return df.with_columns(
        pl.when(
            (pl.col("eligible_start_ts") - pl.col("submit_ts")).dt.total_seconds() > 5
        )
        .then(1)
        .otherwise(0)
        .alias("chained_flag")
    )


def add_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    ts = pl.col("eligible_start_ts")
    hour = ts.dt.hour()
    return df.with_columns(
        month=ts.dt.month().cast(pl.Int8),
        year=ts.dt.year().cast(pl.Int16),
        hour=hour.cast(pl.Int8),
        day=ts.dt.day().cast(pl.Int8),
        day_flag=hour.is_between(6, 17, closed="both").cast(pl.Int8),
        night_flag=((hour < 6) | (hour >= 18)).cast(pl.Int8),
        day_of_week=ts.dt.weekday(),
        weekend_flag=ts.dt.weekday().is_in([6, 7]).cast(pl.Int8),
    )


def add_helper_allocated_flags(
    df: pl.DataFrame,
    denoting_column: str = "allocated_node",
):
    """Auxillary function to calculate queue- count."""
    added_columns = [
        "allocated_count_small_jobs",
        "allocated_count_medium_jobs",
        "allocated_count_large_jobs",
    ]

    q50, q80 = df.select(
        pl.col(denoting_column).quantile(0.5).alias("q50"),
        pl.col(denoting_column).quantile(0.8).alias("q80"),
    ).row(0)

    out = df.with_columns(
        (pl.col(denoting_column) < pl.lit(q50)).cast(pl.Int8).alias(added_columns[0]),
        (
            pl.col(denoting_column).is_between(
                pl.lit(q50),
                pl.lit(q80),
                closed="both",
            )
        )
        .cast(pl.Int8)
        .alias(added_columns[1]),
        (pl.col(denoting_column) > pl.lit(q80)).cast(pl.Int8).alias(added_columns[2]),
    )

    return added_columns, out


def add_active_allocs(
    df: pl.DataFrame,
    resource_cols: list = [
        "allocated_mem",
        "allocated_cpu",
    ],
) -> pl.DataFrame:
    """Adds active resource columns.

    This function adds a new column for all specified `resource_cols`, i.e. calculates how
    much of the given resource were active when the given job was eligible to start. For
    example, if `resorce_cols` would denote only `allocated_mem`, then this function would
    add a column `active_mem`, that would denote the sum of the total memory occupied by
    the jobs that are running when the job is eligible to start.

    Args:
        df: Preprocessed dataframe. See `input.preprocess.preprocess` for details.
        resource_cols: List of allocated columns that are added as active.
    """
    df = df.with_row_index("row_idx")

    work = df.sort(
        [
            "eligible_start_ts",
            "submit_ts",
            "start_ts",
            "row_idx",
        ]
    )

    starts = work.select(
        pl.col("start_ts").alias("ts"),
        *[pl.col(c) for c in resource_cols],
        pl.lit(1).alias("etype"),
    )

    ends = work.select(
        pl.col("end_ts").alias("ts"),
        *[(-pl.col(c)).alias(c) for c in resource_cols],
        pl.lit(0).alias("etype"),
    )

    events = (
        pl.concat([starts, ends])
        .sort(["ts", "etype"])
        .with_columns(
            [
                pl.col(c).cum_sum().alias(f"active_{c.split('allocated_')[1]}")
                for c in resource_cols
            ]
        )
    )

    active = (
        work.join_asof(
            events,
            left_on="eligible_start_ts",
            right_on="ts",
            strategy="backward",
        )
        .with_columns(
            [
                pl.coalesce(
                    [
                        pl.col(f"active_{c.split('allocated_')[1]}"),
                        pl.lit(0),
                    ]
                ).alias(f"active_{c.split('allocated_')[1]}")
                for c in resource_cols
            ]
        )
        # subtract own allocation if already running at eligible_start_ts
        .with_columns(
            [
                pl.when(
                    (pl.col("eligible_start_ts") >= pl.col("start_ts"))
                    & (pl.col("eligible_start_ts") < pl.col("end_ts"))
                )
                .then(pl.col(f"active_{c.split('allocated_')[1]}") - pl.col(c))
                .otherwise(pl.col(f"active_{c.split('allocated_')[1]}"))
                .alias(f"active_{c.split('allocated_')[1]}")
                for c in resource_cols
            ]
        )
        .select(
            "row_idx",
            *[f"active_{c.split('allocated_')[1]}" for c in resource_cols],
        )
    )

    out = df.join(active, on="row_idx", how="left").sort("row_idx").drop("row_idx")
    return out


def add_queued_allocs(
    df: pl.DataFrame,
    resource_cols: list = [
        "allocated_mem",
        "allocated_cpu",
    ],
) -> pl.DataFrame:
    """Adds queued resource columns.

    This function adds the queued resources as columns to the data, similar to the
    `add_active_allocs` function. However, adding the queued resources is programatically
    exponentially harder compared to the active columns. This is because while the active
    jobs could just be simply filtered, here also the priority has to be taken into
    consideration, i.e. jobs submitted later will take priority even if they were
    submitted later on. Therefore, only a specific subset of jobs are summed, and in case
    of ties in terms of priority, the job that was submitted earlier take the higher
    priority.

    This function is by far the most difficult and time-consuming task of the entire
    preprocessing procedure. One would be tempted to use something like `polars.iter_rows`
    for this, but due to the large data-sets and the exponential time complexity, it would
    take this task well over tens of hours with a normal computer.

    As a result of many sleepless and deep-discussions with many different agents, this is
    currently the best working solution. This algorithm uses the Fenwick tree to add the
    queued resources, is relatively fast, but unfortunately not very interpretable by
    nature. Better solutions are more than welcome for this task.

    Args:
        df: Preprocessed dataframe filtered to one partition. See `input.preprocess` for
            details.
        resource_cols: Columns to add the queued resources for.
    """
    df = df.with_row_index("row_idx")
    n = df.height
    if n == 0:
        return df

    # Pull columns into numpy
    submit = df["submit_ts"].to_numpy()
    elig = df["eligible_start_ts"].to_numpy()
    start = df["start_ts"].to_numpy()
    priority = df["priority"].to_numpy()

    # resources: shape (R, n)
    res_arrays = [df[c].to_numpy().astype(np.float64) for c in resource_cols]
    R = len(resource_cols)
    resources = np.vstack(res_arrays)

    job_idx = np.arange(n)

    # Global queue rank: priority DESC, submit ASC, job_idx ASC
    order = np.lexsort((job_idx, submit, -priority))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)

    # Events for waiting intervals [eligible_start, start)
    # add at eligible_start, remove at start
    evt_time = np.concatenate([elig, start])
    evt_job = np.concatenate([job_idx, job_idx])
    evt_is_add = np.concatenate(
        [
            np.ones(n, dtype=bool),
            np.zeros(n, dtype=bool),
        ]
    )

    order_events = np.argsort(evt_time)
    evt_time = evt_time[order_events]
    evt_job = evt_job[order_events]
    evt_is_add = evt_is_add[order_events]

    # Jobs in order of evaluation time (eligible_start_ts)
    job_order = np.argsort(elig)

    # Fenwick tree over rank dimension, one tree per resource type
    bit = np.zeros((R, n + 1), dtype=np.float64)

    def bit_add(idx: int, delta: np.ndarray):
        i = idx
        while i <= n:
            bit[:, i] += delta
            i += i & -i

    def bit_prefix(idx: int) -> np.ndarray:
        out = np.zeros(R, dtype=np.float64)
        i = idx
        while i > 0:
            out += bit[:, i]
            i -= i & -i
        return out

    queued = np.zeros((R, n), dtype=np.float64)
    e = 0
    m = evt_time.shape[0]

    for j in job_order:
        t = elig[j]

        # Process all interval events up to and including time t
        while e < m and evt_time[e] <= t:
            i = evt_job[e]
            idx = rank[i] + 1  # Fenwick is 1-based
            delta = resources[:, i] if evt_is_add[e] else -resources[:, i]
            bit_add(idx, delta)
            e += 1

        # Sum over all waiting jobs with better rank (rank < rank[j])
        queued[:, j] = bit_prefix(rank[j])

    # Attach queued_* columns
    queued_cols = {
        f"queued_{resource_cols[k].split('allocated_')[1]}": queued[k] for k in range(R)
    }

    out = df.with_columns(
        [pl.Series(name, queued_cols[name]) for name in queued_cols]
    ).drop("row_idx")

    return out


def add_active_allocs_with_remaining_timelimit(
    df: pl.DataFrame,
    resource_cols: list[str] = [
        "allocated_mem",
        "allocated_cpu",
        "allocated_timelimit_seconds",
    ],
) -> pl.DataFrame:
    df = df.with_row_index("row_idx")
    if df.height == 0:
        return df.drop("row_idx")
    timelimit_col = "allocated_timelimit_seconds"
    # base_cols = [c for c in resource_cols if c != timelimit_col]
    base_cols = list(resource_cols)
    work = df.sort(
        ["eligible_start_ts", "submit_ts", "start_ts", "row_idx"]
    ).with_columns(
        pl.col("start_ts").cast(pl.Datetime("ns")),
        pl.col("end_ts").cast(pl.Datetime("ns")),
        pl.col("eligible_start_ts").cast(pl.Datetime("ns")),
    )
    # 1) Standard active resources (constant while running): [start_ts, end_ts)
    if base_cols:
        starts = work.select(
            pl.col("start_ts").alias("ts"),
            *[pl.col(c) for c in base_cols],
            pl.lit(1).alias("etype"),
        ).with_columns(pl.col("ts").cast(pl.Datetime("ns")))
        ends = work.select(
            pl.col("end_ts").alias("ts"),
            *[(-pl.col(c)).alias(c) for c in base_cols],
            pl.lit(0).alias("etype"),
        ).with_columns(pl.col("ts").cast(pl.Datetime("ns")))
        events = (
            pl.concat([starts, ends])
            .sort(["ts", "etype"])
            .with_columns(
                [
                    pl.col(c).cum_sum().alias(f"active_{c.split('allocated_')[1]}")
                    for c in base_cols
                ]
            )
        )
        active_base = (
            work.join_asof(
                events,
                left_on="eligible_start_ts",
                right_on="ts",
                strategy="backward",
            )
            .with_columns(
                [
                    pl.coalesce(
                        [pl.col(f"active_{c.split('allocated_')[1]}"), pl.lit(0)]
                    ).alias(f"active_{c.split('allocated_')[1]}")
                    for c in base_cols
                ]
            )
            .with_columns(
                [
                    pl.when(
                        (pl.col("eligible_start_ts") >= pl.col("start_ts"))
                        & (pl.col("eligible_start_ts") < pl.col("end_ts"))
                    )
                    .then(pl.col(f"active_{c.split('allocated_')[1]}") - pl.col(c))
                    .otherwise(pl.col(f"active_{c.split('allocated_')[1]}"))
                    .alias(f"active_{c.split('allocated_')[1]}")
                    for c in base_cols
                ]
            )
            .select(
                "row_idx", *[f"active_{c.split('allocated_')[1]}" for c in base_cols]
            )
        )
    else:
        active_base = work.select("row_idx")
    # 2) Timelimit as remaining seconds, capped by actual end time
    if timelimit_col in resource_cols:
        work_tl = work.with_columns(
            (
                pl.col("start_ts")
                + pl.col(timelimit_col).cast(pl.Int64) * pl.duration(seconds=1)
            ).alias("_requested_deadline")
        ).with_columns(
            pl.min_horizontal("end_ts", "_requested_deadline")
            .cast(pl.Datetime("ns"))
            .alias("_effective_deadline")
        )
        starts_tl = work_tl.select(
            pl.col("start_ts").alias("ts"),
            pl.col("_effective_deadline").dt.epoch("s").alias("deadline_s"),
            pl.lit(1).alias("cnt"),
            pl.lit(1).alias("etype"),
        ).with_columns(pl.col("ts").cast(pl.Datetime("ns")))
        ends_tl = work_tl.select(
            pl.col("_effective_deadline").alias("ts"),
            (-pl.col("_effective_deadline").dt.epoch("s")).alias("deadline_s"),
            pl.lit(-1).alias("cnt"),
            pl.lit(0).alias("etype"),
        ).with_columns(pl.col("ts").cast(pl.Datetime("ns")))
        events_tl = (
            pl.concat([starts_tl, ends_tl])
            .sort(["ts", "etype"])
            .with_columns(
                pl.col("cnt").cum_sum().alias("_cum_cnt"),
                pl.col("deadline_s").cum_sum().alias("_cum_deadline_s"),
            )
        )
        active_tl = (
            work_tl.join_asof(
                events_tl,
                left_on="eligible_start_ts",
                right_on="ts",
                strategy="backward",
            )
            .with_columns(
                pl.coalesce([pl.col("_cum_cnt"), pl.lit(0)]).alias("_cum_cnt"),
                pl.coalesce([pl.col("_cum_deadline_s"), pl.lit(0)]).alias(
                    "_cum_deadline_s"
                ),
                pl.col("eligible_start_ts").dt.epoch("s").alias("_elig_s"),
            )
            .with_columns(
                (pl.col("_cum_deadline_s") - pl.col("_elig_s") * pl.col("_cum_cnt"))
                .clip(lower_bound=0)
                .alias("active_timelimit_seconds_remaining")
            )
            .with_columns(
                pl.when(
                    (pl.col("eligible_start_ts") >= pl.col("start_ts"))
                    & (pl.col("eligible_start_ts") < pl.col("_effective_deadline"))
                )
                .then(
                    pl.col("active_timelimit_seconds_remaining")
                    - (
                        pl.col("_effective_deadline").dt.epoch("s")
                        - pl.col("eligible_start_ts").dt.epoch("s")
                    ).clip(lower_bound=0)
                )
                .otherwise(pl.col("active_timelimit_seconds_remaining"))
                .clip(lower_bound=0)
                .alias("active_timelimit_seconds_remaining")
            )
            .select("row_idx", "active_timelimit_seconds_remaining")
        )
    else:
        active_tl = work.select("row_idx")
    active = active_base.join(active_tl, on="row_idx", how="left").sort("row_idx")
    return df.join(active, on="row_idx", how="left").sort("row_idx").drop("row_idx")
