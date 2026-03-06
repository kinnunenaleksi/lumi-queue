from input.input import get_partition
from train.regress import predict

print("Getting data...")
df = get_partition(partition="small", type="with_features").sample(n=2000000)

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
)

print("Accuracy Metrics")
print(res.accuracy_metrics)
