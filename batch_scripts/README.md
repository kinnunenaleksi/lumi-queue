# Batch Scipts 

This directory holds the batch-job scripts for Lumi required for the analysis. In particular, 

`create_datasets.sh`: Creates a preprocessed dataset for each partition with features under `data/`.
`train_baseline_models.sh`: Creates predictive models with RF, XGB, and MLP methods and stores results for each partition under `results/`
`train_ablation_sets.sh`: Retrains models created in `train_baseline_models` for different feature-sets and stores results under `results/`.
`full_run.sh`: Collectively runs all above.

