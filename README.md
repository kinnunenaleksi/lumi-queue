# Lumi Queue Time Predictions

Repository to predict the queue times in the Lumi supercomputer.

## Structure 

`src`: Holds the underlying Python functions used to preprocess data and the predictive algorithms.
`batch_scripts`: Holds the Slurm batch-job scripts to train models in Lumi.

| Directory    | Description |
| -------- | ------- |
| [`src`](src/)  | Holds underlying code for both preprocessing and model training.    |
| [`batch_scripts`](batch_scripts/) | Makes batch-jobs for preprocessing and model-training in Lumi.     |
| [`tests`](tests/)    | Unit-tests for functionality in `src`.  |

## Methdology

The framework to create predictions and consequent inference is two-fold. First, three distinct 
predictive algorithms are fitted for the original data, and the best tuned algorithm is chosen for 
each partition. This model is then analyzed via various performance metrics, 

Second, using the same algorithm for each partition as in the first part, the model is re-trained 
for multiple different feature-sets (ablation-sets) to analyze the predictive power of distinct 
features.
