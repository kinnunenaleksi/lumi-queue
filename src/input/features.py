import numpy as np
import polars as pl


def add_features(df: pl.DataFrame):
    resource_cols = [col for col in df.columns if col.startswith("allocated_")]
    df1 = add_active_allocs(df, resource_cols=resource_cols)
    df2 = add_queued_allocs(df1, resource_cols=resource_cols)
    df2 = add_reservation_flag(df2)
    return df2


def add_reservation_flag(df: pl.DataFrame):
    return df.with_columns(
        pl.when(pl.col("reservation_id") > 0)
        .then(1)
        .otherwise(0)
        .alias("reservation_flag")
    )


def add_active_allocs(
    df: pl.DataFrame, resource_cols: list = ["allocated_mem", "allocated_cpu"]
) -> pl.DataFrame:
    """
    Add active_* columns: sum of allocations of all other jobs that are
    running at a job's eligible_start_ts.
    """
    df = df.with_row_index("row_idx")

    work = df.sort(["eligible_start_ts", "submit_ts", "start_ts", "row_idx"])

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
        .sort(["ts", "etype"])  # end (0) before start (1)
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
                    [pl.col(f"active_{c.split('allocated_')[1]}"), pl.lit(0)]
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
    df: pl.DataFrame, resource_cols: list = ["allocated_mem", "allocated_cpu"]
) -> pl.DataFrame:
    """
    Fast queued_* implementation using a Fenwick tree.

    For each job A at t = eligible_start_ts_A, queued_* is the sum of allocations
    of jobs B such that:

      - eligible_start_B <= t < start_B
      - B has *higher queue rank* than A, where rank is defined by:
          priority DESC, submit_ts ASC, job_index ASC

    Higher rank means:
      - higher priority wins;
      - for ties, earlier submit wins.

    This matches the semantics from the test:
      - a job with strictly higher priority blocks a job with lower priority;
      - between equal priority jobs, earlier submit blocks later.
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
    evt_is_add = np.concatenate([np.ones(n, dtype=bool), np.zeros(n, dtype=bool)])

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
