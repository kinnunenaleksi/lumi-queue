from input.input import create_datasets

DATA_PATH = "../anonJobs.parquet"
EXPORT_PATH = "filtered_data"
PARTITIONS = ["small", "small-g", "standard", "standard-g"]
TRUNCATE_PCT = 1


def main():
    _ = create_datasets(
        partitions=PARTITIONS,
        export_path=EXPORT_PATH,
        data_path=DATA_PATH,
        truncate_pct=TRUNCATE_PCT,
        filter_geq_minutes=10,
    )


if __name__ == "__main__":
    main()
