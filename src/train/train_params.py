PARTITION_LIST_FULL = ["small", "small-g", "standard", "standard-g"]
PARTITION_LIST_TEST = ["largemem", "lumid"]
MODELS_LIST_TEST = ["rf", "gb"]

# These features are used for all baseline models
BASELINE_FEATURES = [
    "priority",
    "reservation_flag",
    "chained_flag",
    "allocated_timelimit_seconds",
    "queued_timelimit_seconds",
    "active_timelimit_seconds",
    "queued_count_small_jobs",
    "queued_count_medium_jobs",
    "queued_count_large_jobs",
    "active_count_small_jobs",
    "active_count_medium_jobs",
    "active_count_large_jobs",
]

# For the perfect-feature-set, the timelimits are changed to actual run-times
PERFECT_FEATURE_MAPPING = {
    "allocated_timelimit_seconds": "allocated_elapsed_seconds",
    "queued_timelimit_seconds": "queued_elapsed_seconds",
    "active_timelimit_seconds": "active_elapsed_seconds",
}

PERFECT_FEATURES = [PERFECT_FEATURE_MAPPING.get(f, f) for f in BASELINE_FEATURES]

# Features for resource-allocatable partitions
CPU_FEATURES = ["allocated_cpu", "queued_cpu", "active_cpu"]
MEM_FEATURES = ["allocated_mem", "queued_mem", "active_mem"]
GPU_FEATURES = ["allocated_gpu", "queued_gpu", "active_gpu"]
NODE_FEATURES = ["allocated_node", "queued_node", "active_node"]

LOAD_FEATURES = CPU_FEATURES + MEM_FEATURES + GPU_FEATURES + NODE_FEATURES


FEATURE_SETS = {
    "standard": {
        "baseline": BASELINE_FEATURES + NODE_FEATURES,
        "perfect": PERFECT_FEATURES + NODE_FEATURES,
    },
    "standard-g": {
        "baseline": BASELINE_FEATURES + NODE_FEATURES,
        "perfect": PERFECT_FEATURES + NODE_FEATURES,
    },
    "small": {
        "baseline": BASELINE_FEATURES + CPU_FEATURES + MEM_FEATURES + NODE_FEATURES,
        "perfect": PERFECT_FEATURES + CPU_FEATURES + MEM_FEATURES + NODE_FEATURES,
    },
    "small-g": {
        "baseline": BASELINE_FEATURES + LOAD_FEATURES,
        "perfect": PERFECT_FEATURES + LOAD_FEATURES,
    },
    # For testing purposes
    "largemem": {"baseline": BASELINE_FEATURES, "perfect": PERFECT_FEATURES},
    "lumid": {"baseline": BASELINE_FEATURES, "perfect": PERFECT_FEATURES},
}
