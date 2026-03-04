from input.input import get_partition
from train.regress import predict

# import pickle
from joblib import dump

print("Getting data...")
df = get_partition(partition="largemem", type="with_features").sample(n=2000)

predictors = [
    "priority",
    "allocated_node",
    "allocated_cpu",
    "allocated_mem",
    "queued_node",
    "queued_cpu",
    "queued_mem",
    "active_node",
    "active_cpu",
    "active_mem",
    "timelimit_minutes",
    "reservation_flag",
]

print("Getting predictions...")
res = predict(
    df,
    y_col="wait_time_seconds",
    x_cols=predictors,
    no_features=len(predictors),
    search_method="grid",
    test_size=0.4,
    # split_method="timeseries",
    split_method="random",
)

print("Accuracy Metrics")
print(res.accuracy_metrics)

dump(res, "ml_results.pkl")
