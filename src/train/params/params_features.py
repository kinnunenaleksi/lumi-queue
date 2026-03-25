# These features are used for all baseline models
BASELINE_FEATURES = [
    "priority",
    "reservation_flag",
    "chained_flag",
    "allocated_timelimit_seconds",
    "queued_timelimit_seconds",
    "active_timelimit_seconds",
    "active_timelimit_seconds_remaining",
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
    "active_timelimit_seconds_remaining": "active_elapsed_seconds_remaining",
}

PERFECT_FEATURES = [PERFECT_FEATURE_MAPPING.get(f, f) for f in BASELINE_FEATURES]

TEMPORAL_FEATURES = [
    "month",
    "year",
    "hour",
    "day",
    "day_of_week",
    "day_flag",
    "night_flag",
    "weekend_flag",
]

USAGE_FEATURES = [
    "count_user_submitted_jobs_7d",
    "count_account_submitted_jobs_7d",
]


# Features for resource-allocatable partitions
CPU_FEATURES = ["allocated_cpu", "queued_cpu", "active_cpu"]
MEM_FEATURES = ["allocated_mem", "queued_mem", "active_mem"]
GPU_FEATURES = ["allocated_gpu", "queued_gpu", "active_gpu"]
NODE_FEATURES = ["allocated_node", "queued_node", "active_node"]

LOAD_FEATURES = CPU_FEATURES + MEM_FEATURES + GPU_FEATURES + NODE_FEATURES
"""1.

PERFECT
2. PERFECT WITHOUT TEMPORAL
3. BASELINE
4. BASELINE WITHOUT USAGE
5. BASELINE WITHOUT NODE_FEATURES
"""

FEATURE_SETS = {
    "standard": {
        "perfect": BASELINE_FEATURES
        + NODE_FEATURES
        + USAGE_FEATURES
        + TEMPORAL_FEATURES,
        "baseline": BASELINE_FEATURES + NODE_FEATURES + USAGE_FEATURES,
        "system": BASELINE_FEATURES + NODE_FEATURES,
        "naive": BASELINE_FEATURES,
    },
    "standard-g": {
        "perfect": BASELINE_FEATURES
        + NODE_FEATURES
        + USAGE_FEATURES
        + TEMPORAL_FEATURES,
        "baseline": BASELINE_FEATURES + NODE_FEATURES + USAGE_FEATURES,
        "system": BASELINE_FEATURES + NODE_FEATURES,
        "naive": BASELINE_FEATURES,
    },
    "small": {
        "baseline": BASELINE_FEATURES + CPU_FEATURES + MEM_FEATURES + NODE_FEATURES,
        "perfect": PERFECT_FEATURES + CPU_FEATURES + MEM_FEATURES + NODE_FEATURES,
    },
    "small-g": {
        "perfect": BASELINE_FEATURES
        + LOAD_FEATURES
        + USAGE_FEATURES
        + TEMPORAL_FEATURES,
        "baseline": BASELINE_FEATURES + LOAD_FEATURES + USAGE_FEATURES,
        "system": BASELINE_FEATURES + LOAD_FEATURES,
        "naive": BASELINE_FEATURES,
    },
}
# FEATURE_SETS = {
#     "standard": {
#         "perfect": PERFECT_FEATURES
#         + NODE_FEATURES
#         + TEMPORAL_FEATURES
#         + USAGE_FEATURES,
#         "without_temporal": BASELINE_FEATURES + NODE_FEATURES + USAGE_FEATURES,
#         "baseline": BASELINE_FEATURES + NODE_FEATURES,
#         "naive": BASELINE_FEATURES,
#     },
#     "standard-g": {
#         "baseline": BASELINE_FEATURES + NODE_FEATURES,
#         "perfect": PERFECT_FEATURES + NODE_FEATURES,
#     },
#     "small": {
#         "baseline": BASELINE_FEATURES + CPU_FEATURES + MEM_FEATURES + NODE_FEATURES,
#         "perfect": PERFECT_FEATURES + CPU_FEATURES + MEM_FEATURES + NODE_FEATURES,
#     },
#     "small-g": {
#         "baseline": BASELINE_FEATURES + LOAD_FEATURES,
#         "perfect": PERFECT_FEATURES + LOAD_FEATURES,
#     },
# }
