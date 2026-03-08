# train

This module trains predictive models for the proprocessed data, i.e. the output of the module 
`input`. 

## Structure 

### Main Functionality

`regress.py`: Trains models for one partition at a time.
`train.py`: Loops the `regress.py` functionality over partitions and feature-sets.
`utils.py`: Auxillary functions for both `regress.py` and `train.py`.

### Parameters 

The dedicated `params/` folder holds the following:

`params_models.py`: Holds the model configurations and hyperparameter grids for training.
`params_training.py`: Holds the training specifications, i.e. what models is trained for each
partition and feature-set.
`params_features.py`: Denotes the different columns for feature-sets.

## Feature Sets 

## Models 

## Usage
The following demonstrates the logic and usage of the `train` module. Functions take 

The snippet below trains and tunes RF for the partition `small-g` for two feature-sets, and both
RF and XGB algorithms for the partition `standard` for a single feature-set.

```python
from train.train import train_models

TEST_MODELS = {
    "small-g": {
        "models": ["rf"],
        "feature_sets": ["perfect", "baseline"],
    },
    "standard": {
        "models": ["rf", "xgb"],
        "feature_sets": ["perfect"],
    },
}

FEATURE_SETS = {
    'small-g' : {
        'baseline': ['priority', 'queued_count_large_jobs', 'allocated_timelimit_seconds'],
        'perfect' : ['priority', 'queued_count_large_jobs', 'allocated_elapsed_seconds']
    },

    'standard' : {
        'perfect' : ['priority', 'queued_count_large_jobs', 'allocated_elapsed_seconds']
    }
}

TEST_MODEL_CONFIGS = {
    "rf": ModelConfig(
        estimator=RandomForestRegressor,
        param_grid={
            "n_estimators": [10, 20],
        },
        cv_folds=3,
        use_permutation_importance=False,
        estimator_kwargs={"n_jobs": -1},
        grid_search_kwargs={"scoring": "neg_mean_squared_error", "refit": True},
    ),
    "xgb": ModelConfig(
        estimator=xgb.XGBRegressor,
        param_grid={
            "max_depth": [1, 3],
        },
        cv_folds=3,
        use_permutation_importance=True,
        estimator_kwargs={"n_jobs": -1},
        grid_search_kwargs={"scoring": "neg_mean_squared_error", "refit": True},
    ),
}

training_results = train_models(
    train_dict=TEST_MODELS,
    feature_sets=FEATURE_SETS,
    model_configs=TEST_MODEL_CONFIGS,
    y_col="wait_time_seconds",
    test_size=0.4,
    save_results=True,
    truncate_pct=0.02,
    scaling_policy="none",
    log_transform_policy="none",
    split_method="timeseries",
    search_method="random",
)
```
In particular, the `train_models` function trains all of the required models for each partition and 

Models and results are then stored in a new folder `results/`, where they can be later
gathered from.

In total, three models are considered: Random Forest (RF), XGBoost (XGB) Neural Networks (MLP). All
models go through hyperparameter tuning, and currently only the best model is saved in to the
results.



