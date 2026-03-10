import os

import polars as pl

from input.features import add_features
from input.preprocess import preprocess

INPUT_PATH = "../anonJobs.parquet"
EXPORT_PATH = "data/"


def get_partition(
    data_path: str = INPUT_PATH,
    partition: str = "largemem",
    type: str = "raw",
    truncate_pct: float = 1.0,
):
    """Main function of the `input` module.

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

    else:
        lf = pl.scan_parquet(data_path).filter(pl.col("Partition") == partition)

    if truncate_pct < 1.0:
        lf = lf.select(pl.all().sample(fraction=truncate_pct, seed=49))

    if type == "raw":
        df = lf.collect()

    if type == "preprocess":
        df = preprocess(lf)

    if type == "with_features":
        df = preprocess(lf)
        df = add_features(df, partition=partition)

    return df


def create_datasets(
    partitions: list,
    data_path: str = INPUT_PATH,
    export_path: str = EXPORT_PATH,
    type: str = "with_features",
    truncate_pct: float = 1,
):
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

        df_partition.write_parquet(path)
        written_files.append(path)

    return written_files
