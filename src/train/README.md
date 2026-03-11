# Train

Functionality to train models from the preprocessed data.

## Structure 

### Main Functionality

| File    | Description |
| -------- | ------- |
| [`regress.py`](regress.py) | Trains models for one partition at a time.  |
| [`train.py`](train.py) | Loops the functionality above over partitions and feature-sets.     |
| [`utils.py`](utils.py) | Auxillary functions for both above.     |

### Parameters 

The dedicated `params/` folder holds the following:

| File    | Description |
| -------- | ------- |
| [`params_models.py`](params/params_models.py) | Holds model configurations and hyperparameter grids for training.  |
| [`params_training.py`](params/params_training.py) | Holds the training specifications, i.e. what models is trained for each partition and feature-set.     |
| [`params_features.py`](params/params_features.py) | Holds the features for each feature-set.     |

## Usage 

The main function of this module is the `train.train_models`, that creates and tunes models
predictive models and saves results in a pickle file. This function expectes having datasets 
as derived in the [input-module](../input/README.md). Thus, the following script...

```python
from train.train import train_models
from train.params.params_models import MODEL_CONFIGS
from train.params.params_features import FEATURE_SETS

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

result_dir = train_models(
    train_dict=TEST_MODELS,
    feature_sets=FEATURE_SETS,
    model_configs=MODEL_CONFIGS,
    y_col="wait_time_seconds",
    test_size=0.4,
    split_method="timeseries",
    search_method="grid",
)

result_dir = train_models()
```
Yields the following folder-structure:

```bash
.
├── model_results/
│   ├── rf-xgb__small-g-standard__20260311T1552
│   │   ├── features.json
│   │   ├── model_parameters.json
│   │   ├── res_small_g_rf_baseline.pkl
│   │   ├── res_small-g_rf_perfect.pkl
│   │   ├── res_standard_rf_perfect.pkl
│   │   └── res_standard_xgb_perfect.pkl
```
From above, each of the `.pkl` files store the various training metrics in a dataclass. In particular,
the following are saved:

| Object    | Type | Description |
| -------- | ------ |------- |
| `df_feature_selection`| DataFrame | Feature selection scores.  |
| `df_cv_results`| DataFrame | Hyperparameter tuning results for each CV.  |
| `df_feature_importance`| DataFrame |  Feature importances calculated as per `params.params_models` for the best model.|
| `df_accuracy_metrics`| DataFrame |  Accuracy metrics for the best model.  |
| `y_pred`| Array |  Target predictions.  |
| `validation_indices`| Array |  Validation row indexes.  |
| `best_model`| sklearn.Model |  Best model of hyperparameter tuning.  |

## Configurations
### Feature Sets 

Feature sets denote the collection of explanatory variables used for training. To granularly understand how
specific features affect the training results, a variety of different combinations should be
considered and compared to each other. The following have been pre-configured.

`baseline`: Use all derived features as per [preprocessing](../input/README.md).

`perfect`: Replace the user-set time limits to actual run-times of the jobs to assess
how much the error in estimates affect the model performance.

`without_alloc`: Remove the `allocTRES` derived features from the `baseline` set, to see model performance when
predicting only by the system load.

`without_load` Remove the features with prefix `queued_` and `active_`, to see model performance
when predicting only by the resources requested for a job.

Intuitively, 


### Models 

For training, Random Forest (RF), XGBoost (XGB), Gradient Boosting (GB) and Multi-Layered Perceptron
(MLP) models have been pre-configured.

## Usage

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

```
In particular, the `train_models` function trains all of the required models for each partition and 

Models and results are then stored in a new folder `results/`, where they can be later
gathered from.

In total, three models are considered: Random Forest (RF), XGBoost (XGB) Neural Networks (MLP). All
models go through hyperparameter tuning, and currently only the best model is saved in to the
results.



