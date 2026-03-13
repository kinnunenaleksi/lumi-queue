# Analyze

Functionality to combine and analyze the model training results.

## Structure 

| File    | Description |
| -------- | ------- |
| [`analyze.py`](analyze.py)    | Creates reports.  |
| [`utils.py`](utils.py) | Auxillary functions for analysis.     |
| [`visualization.py`](visualization.py) | Helper functions for visualizations.     |

## Usage 

The main function of this module is the `analyze.create_reports`, that combines the training
results from [`train`](../train/README.md), and creates text-reports with both Markdown 
and Typst output. Thus, the following...

```python
from analyze.analyze import create_reports

result_dir = 'model_results/rf-xgb__small-g-standard__20260311T1552'
res, _, _ = create_reports(results_dir=result_dir)
```
Adds two files into the model result directory: `accuracy_results.txt` and `cv_results.txt`. In
particular, `accuracy_results.txt` holds all variations of accuracy-results, comparing feature-sets
and different models. The output is in the following format:

```bash
----- partition=standard | feature_set=perfect | comparing: model -----

| metric               | xgb                    | rf                     |
|:---------------------|:-----------------------|:-----------------------|
| rmse                 | 13129.56               | 10835.2                |
| r2                   | -0.65                  | -0.13                  |
| med_seconds          | 23.82                  | 67.9                   |
| perc_err_under_1min  | 62.41%                 | 47.44%                 |
| perc_err_under_3min  | 79.07%                 | 69.02%                 |
| perc_err_under_5min  | 83.72%                 | 76.3%                  |
| perc_err_under_10min | 87.92%                 | 84.86%                 |
| perc_err_under_30min | 92.97%                 | 92.0%                  |

#table(
  columns: 4,
  [metric], [xgb], [rf],

  [rmse], [13129.56], [10835.2],
  [r2], [-0.65], [-0.13],
  [med_seconds], [23.82], [67.9],
  [perc_err_under_1min], [62.41%], [47.44%],
  [perc_err_under_3min], [79.07%], [69.02%],
  [perc_err_under_5min], [83.72%], [76.3%],
  [perc_err_under_10min], [87.92%], [84.86%],
  [perc_err_under_30min], [92.97%], [92.0%],
)
```

The `cv_results.txt` instead compares the hyperparameter tuning and cross-validation results. The
output is in the following format: 

```bash
----- small-g, rf -----

| feature_set   | mean_test_score   | rank_test_score   | n_estimators   |
|:--------------|:------------------|:------------------|:---------------|
| baseline      | -8.87734          | 1                 | 20             |
| baseline      | -9.15372          | 2                 | 10             |
| perfect       | -8.52598          | 1                 | 20             |
| perfect       | -8.97922          | 2                 | 10             |

#table(
  columns: 5,
  [feature_set], [mean_test_score], [rank_test_score], [n_estimators],

  [baseline], [-8.877340], [baseline], [-9.153723], [perfect], [-8.525980], [perfect], [-8.979223], )
```
