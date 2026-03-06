# train

This module trains predictive models for the proprocessed data, i.e. the output of the module 
`input`. Models and results are then stored in a new folder `results/`, where they can be later
gathered from.

In total, three models are considered: Random Forest (RF), XGBoost (XGB) Neural Networks (MLP). All
models go through hyperparameter tuning, and currently only the best model is saved in to the
results.

## Structure 

`model_params.py`: Holds the model configurations and hyperparameter grids for training.
`regress.py`: Holds utilities to train multiple models into one dataset.
`train_params.py`: Denotes the feature-sets for training.
`train.py`: Combines all above. Enables training multiple models for multiple partitions, storing
results into a single folder,  and holds utilies to combine results in hindsight.

## Usage 


