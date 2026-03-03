import polars as pl
from src.input.preprocess import preprocess

lf = pl.scan_parquet("../anonJobs.parquet").filter(pl.col("Partition") == "largemem")

df = preprocess(lf)

print(df.shape[0])
