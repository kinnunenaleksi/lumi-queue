# Lumi Queue Time Predictions

The scope of this repository is to offer configurable framework for: 

- Pre-processing raw Slurm data from the `sacct` utility
- Feature engineering new variables
- Training and tuning predictive models

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

3. Create datasets and run model-training by submitting batch-jobs into queue with

```bash
sbatch batch_scripts/full_run.sh
```
This script simultaneously:

1. Creates a preprocessed dataset with added features for each of the designated partitions in a new
   directory `data/`
2. Trains and tunes models for various feature-sets as per the [configurations](src/train/params),
   and saves results into a new directory `results/`
