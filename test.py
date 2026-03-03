import polars as pl
from src.input.input import get_partition

df = get_partition(partition="small-g", type="with_features")

print(df.shape[0])


# from src.input.preprocess import preprocess
#
# lf = pl.scan_parquet("../anonJobs.parquet").filter(pl.col("Partition") == "small-g")
#
# df = preprocess(lf)
#
# print(df.shape[0])
