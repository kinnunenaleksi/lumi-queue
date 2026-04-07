import os

import polars as pl

from input.features import add_features
from input.preprocess import preprocess


def create_datasets(
    partitions: list,
    data_path: str,
    export_path: str,
    truncate_pct: float,
    type: str = "with_features",
    filter_geq_minutes: int = 0,
):
    """Main function of `input`. Creates preprocessed datasets for defined partitions.

    This function loops the `get_partition` function over determined partitions.
    Running this function on MacBook Pro 2021 M1 takes ~1min for partitions with around
    two million rows, and ~2.5 minutes for the largest `small` partition with five
    million rows.

    Args:
        partitions: List of partitions to create.
        data_path: Relative path to data.
        export_path: Relative path to the folder with new datasets.
        truncate_pct: See function `get_partition`.
        type: See function `get_partition`.

    Returns:
    """
    os.makedirs(f"{export_path}/", exist_ok=True)

    written_files = []

    for partition in partitions:
        dataset_name = f"{partition}.parquet"
        path = f"{export_path}/{dataset_name}"

        df_partition = get_partition(
            partition=partition,
            type=type,
            truncate_pct=truncate_pct,
            data_path=data_path,
        )

        if filter_geq_minutes > 0:
            df_partition = df_partition.filter(
                pl.col("wait_time_minutes") >= filter_geq_minutes
            )

        df_partition.write_parquet(path)
        written_files.append(path)

    return written_files


def get_partition(
    data_path: str,
    partition: str,
    type: str,
    truncate_pct: float = 1.0,
):
    """Runs preprocessing and adds features for a single partition.

    This function takes the relative path to the raw Slurm data, and returns the
    data in the specified format. In particular, this function can either filter
    the data into a single partition, preprocess the data as per the
    `preprocess` function, add the required features or all of the above.

    If features are added, then it is mandatory to filter the data into a specific
    partition. This is because some of the features are calculated with respect to
    the limits within the partition, and all partitions are unique in their distributions.

    Args:
        data_path: Relative path to the raw Slurm output of `sacct`.
        partition: A valid Slurm partition in Lumi, e.g. 'small' or 'standard-g'.
        type: The level of operations conducted for the data. Can be one of the following:
            - `raw`: Returns the data as is.
            - `preprocess`: Returns preprocessed dataset. See `input.preprocess` for further
                    details.
            - `with_features`: Returns preprocessed dataset with features. See `input.features`
                    for further details.
        truncate_pct: Determines the percentage of the data fetched. Used for testing
            and validation purposes.

    Returns:
        Requested polars dataframe.
    """
    if partition == "all":
        lf = pl.scan_parquet(data_path)

    elif partition in ["small", "small-g", "standard", "standard-g"]:
        lf = pl.scan_parquet(data_path).filter(pl.col("Partition") == partition)
    else:
        raise ValueError(
            "Partition has to be in [small, small-g, standard, standard-g]"
        )

    if truncate_pct < 1.0:
        lf = lf.select(pl.all().sample(fraction=truncate_pct, seed=49))

    if type == "raw":
        df = lf.collect()

    elif type == "preprocess":
        df = preprocess(lf)

    elif type == "with_features":
        df = preprocess(lf)
        df = add_features(df, partition=partition)
    else:
        raise ValueError("`type` has to be in [raw, preprocess, with_features]")

    return df
