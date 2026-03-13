# Source

Source code for the training framework. 

## Structure 

| Module    | Description |
| -------- | ------- |
| [`input/`](input/) | Preprocessess data and saves datasets in `data/` |
| [`train/`](train/) | Trains and tunes predictive models and saves training results in `model_results/` |
| [`analyze/`](analyze/) | Functionality for model performance analysis and comparison. |

## Usage 

The intended training pipeline is as follows:

1. Create preprocessed datasets for each partition of interest 
2. Train and tune models based on the new datasets 
3. Fetch and combine results from the `results/` folder 

Thus, the following goes through the entire pipeline.

```python
from input.input import create_datasets
from train.train import train_models
from analyze.analyze import create_reports

from train.params.params_features import FEATURE_SETS
from train.params.params_models import MODEL_CONFIGS

TRAINING_MODELS = {
    "small-g": {
        "models": ["rf"],
        "feature_sets": ["perfect", "baseline"],
    },
    "standard": {
        "models": ["rf", "xgb"],
        "feature_sets": ["perfect"],
    },
}

PARTITIONS = list(TRAINING_MODELS.keys())
DATA_PATH = "<PATH-TO-DATA>"
INPUT_PATH = "data/"
RESULT_PATH = "model_results/"

written_files = create_datasets(
    data_path = DATA_PATH,
    export_path = INPUT_PATH,
    partitions = PARTITIONS,
)

results_dir = train_models(
    train_dict = TRAINING_MODELS,
    feature_sets = FEATURE_SETS,
    model_configs = MODEL_CONFIGS,
    input_path = INPUT_PATH,
    export_path = RESULT_PATH,
    test_size = 0.2,
    y_col = "wait_time_seconds",
    split_method = "timeseries",
)

_ = create_reports(
    results_dir = results_dir
)
```
