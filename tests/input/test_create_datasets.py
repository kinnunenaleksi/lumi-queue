import shutil

import polars as pl

from input.input import create_datasets

EXPORT_PATH = "test_data/"


def test_create_datasets(
    partitions=["small-g", "standard"], truncate_pct=0.01, export_path=EXPORT_PATH
):

    paths = create_datasets(
        partitions=partitions, truncate_pct=truncate_pct, export_path=export_path
    )

    df1 = pl.read_parquet(f"{EXPORT_PATH}/small-g.parquet")
    df2 = pl.read_parquet(f"{EXPORT_PATH}/standard.parquet")

    assert df1 is not None
    assert df2 is not None

    shutil.rmtree(f"{EXPORT_PATH}")

    # assert paths == ["small-g.parquet", "standard.parquet"]
