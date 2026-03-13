import polars as pl

from input.features import add_active_allocs, add_queued_allocs


def test_active_allocs_simple():
    df = pl.DataFrame(
        {
            "submit_ts": [0, 5, 15, 25, 40],
            "eligible_start_ts": [0, 5, 15, 25, 40],
            "start_ts": [10, 15, 15, 35, 45],
            "end_ts": [50, 40, 30, 60, 55],
            "priority": [1, 1, 1, 1, 1],  # not used for active
            "allocated_mem": [10, 20, 30, 40, 50],
            "allocated_cpu": [1, 2, 3, 4, 5],
            "allocated_gpu": [0, 0, 0, 0, 0],
        }
    )

    out = add_active_allocs(df)

    mem = out.get_column("active_mem").to_list()
    cpu = out.get_column("active_cpu").to_list()

    assert mem == [0, 0, 30, 60, 50]
    assert cpu == [0, 0, 3, 6, 5]


def test_queued_allocs_simple():
    df = pl.DataFrame(
        {
            "submit_ts": [0, 2, 4, 6, 8],
            "eligible_start_ts": [0, 2, 4, 6, 8],
            "start_ts": [20, 15, 15, 12, 30],
            "end_ts": [100, 50, 40, 25, 60],
            "priority": [5, 5, 7, 7, 4],
            "allocated_mem": [10, 20, 30, 40, 50],
            "allocated_cpu": [1, 2, 3, 4, 5],
            "allocated_gpu": [0, 0, 0, 0, 0],
        }
    )

    df2 = add_active_allocs(df)
    df3 = add_queued_allocs(df2)

    queued_mem = df3.get_column("queued_mem").to_list()
    queued_cpu = df3.get_column("queued_cpu").to_list()

    assert queued_mem == [0, 10, 0, 30, 100]
    assert queued_cpu == [0, 1, 0, 3, 10]
