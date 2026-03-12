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

```python
from input.input import create_datasets
from train.train import train_models
from analyze.analyze import fetch_results

written_datasets = create_datasets()
written_models = train_models()
analysis = fetch_results()
```
