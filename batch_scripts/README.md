# Batch Scipts 

This directory holds the batch-job scripts for Lumi used in the thesis.  

## Contents

| File    | Description |
| -------- | ------- |
| [`scripts/`](scripts/)    | Python executables that call functions from [`src`](../src/README.md) |
| [`create_datasets.sh`](create_datasets.sh)    | Creates preprocessed datasets in Lumi. |
| [`train_baseline_models.sh`](train_baseline_models.sh) | Trains and tunes various models for `baseline` feature-sets. |
| [`train_ablation_sets.sh`](train_ablation_sets.sh) | Trains and tunes the best model from baseline for various feature-sets. |

Both `.out` and `.err` files are stored in a new directory `runs/`.
