import polars as pl

from input.features import add_features
from input.preprocess import preprocess

DATA_PATH = "../anonJobs.parquet"


def get_partition(partition: str = "largemem", type: str = "raw"):

    if partition == "all":
        lf = pl.scan_parquet(DATA_PATH)

    else:
        lf = pl.scan_parquet(DATA_PATH).filter(pl.col("Partition") == partition)

    if type == "raw":
        df = lf.collect()

    if type == "preprocess":
        df = preprocess(lf)

    if type == "with_features":
        df = preprocess(lf)
        df = add_features(df)

    return df
