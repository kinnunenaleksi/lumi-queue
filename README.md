# Lumi Queue Time Predictions

Configurable framework to analyze queue performance in Lumi Supercomputer.

## Structure 

| Directory    | Description |
| -------- | ------- |
| [`src/`](src/)  | Holds underlying code and parameters for both preprocessing and model training.    |
| [`tests/`](tests/)    | Unit-tests for functionality in `src`.  |
| [`batch_scripts/`](batch_scripts/) | Makes batch-jobs for preprocessing and model-training in Lumi.     |

## Usage on Lumi Supercomputer

1. Pull this repository into your project-storage `projappl/<project_id>` with 

```bash
git clone git@github.com:kinnunenaleksi/lumi-queue.git
```

2. Move the dataset (see [data specifications](src/input/README.md)) into the same folder with 

```bash
mv <PATH-TO-DATA> projappl/<PROJECT_ID>/lumi-queue
```

3. Submit a batch-job to create preprocessed datasets with

```bash
sbatch batch_scripts/create_datasets.sh
```
This script creates a preprocessed dataset with added features for each of the designated partitions in a new directory `data/`

4. Submit a batch-job to train models with 

```bash
sbatch batch_scripts/train_models.sh
```
This script trains and tunes models for configured partitions and features as per the
[configurations](src/train/README.md), and saves results into a new directory `model_results/`.
