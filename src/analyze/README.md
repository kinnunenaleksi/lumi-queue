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
Adds two files into the model result directory: `accuracy_results.txt` and `cv_results.txt`. 
