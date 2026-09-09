import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from analyze.utils import explode_res_name, recreate_dataset


def create_visualizations(
    results_dir, partitions, input_path, output_path, compression
):

    for partition in partitions:
        df_partition = recreate_dataset(
            results_dir=results_dir,
            partition=partition,
            input_path=input_path,
            compression=compression,
        )


def plot_err_histogram(df, output_path: str):

    pred_cols = [c for c in df.columns if c.startswith("y_pred_")]

    models = [c.split("_")[3] for c in pred_cols]
    feature_sets = [c.split("_")[4] for c in pred_cols]

    for model in models:
        fig, ax = plt.subplots(ncols=2, nrows=1)
        for feature_set in feature_sets:
            fig = sns.histplot(
                df, x=f"y_pred_{model}_{feature_set}", ax=ax, log_scale=True, alpha=0.3
            )

            plt.savefig(f"{output_path}/histogram_err_{model}_{feature_set}")
