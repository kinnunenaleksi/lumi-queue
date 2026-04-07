# These features are used for all baseline models
BASELINE_FEATURES = [
    "reservation_flag",
    "chained_flag",
    "allocated_timelimit_seconds",
]

SYSTEM_LOAD_FEATURES = [
    "queued_timelimit_seconds",
    "active_timelimit_seconds",
    "queued_count_small_jobs",
    "queued_count_medium_jobs",
    "queued_count_large_jobs",
    "active_count_small_jobs",
    "active_count_medium_jobs",
    "active_count_large_jobs",
    "active_timelimit_seconds_remaining",
]

NODE_ALLOCATABLE_ALLOC_FEATURES = ["allocated_node"]
NODE_ALLOCATABLE_SYSTEM_LOAD_FEATURES = ["queued_node", "active_node"]

RESOURCE_ALLOCATABLE_ALLOC_FEATURES = [
    "allocated_cpu",
    "allocated_mem",
    "allocated_node",
]

RESOURCE_ALLOCATABLE_SYSTEM_FEATURES = [
    "queued_cpu",
    "active_cpu",
    "queued_mem",
    "active_mem",
    "queued_node",
    "active_node",
]


GPU_ALLOC = ["allocated_gpu"]
GPU_SYSTEM_LOAD = ["queued_gpu", "active_gpu"]
GPU_TOTAL = GPU_ALLOC + GPU_SYSTEM_LOAD

TEMPORAL_FEATURES = [
    "month",
    "year",
    "hour",
    "day",
    "day_of_week",
    # "day_flag",
    # "night_flag",
    # "weekend_flag",
]

USAGE_FEATURES = [
    "count_user_submitted_jobs_7d",
    "count_account_submitted_jobs_7d",
    "priority",
]

STANDARD_FULL = (
    BASELINE_FEATURES
    + SYSTEM_LOAD_FEATURES
    + NODE_ALLOCATABLE_ALLOC_FEATURES
    + NODE_ALLOCATABLE_SYSTEM_LOAD_FEATURES
    + TEMPORAL_FEATURES
    + USAGE_FEATURES
)

STANDARD_BASELINE = (
    BASELINE_FEATURES
    + SYSTEM_LOAD_FEATURES
    + NODE_ALLOCATABLE_ALLOC_FEATURES
    + NODE_ALLOCATABLE_SYSTEM_LOAD_FEATURES
    + USAGE_FEATURES
)

STANDARD_NAIVE = (
    BASELINE_FEATURES
    + SYSTEM_LOAD_FEATURES
    + NODE_ALLOCATABLE_ALLOC_FEATURES
    + NODE_ALLOCATABLE_SYSTEM_LOAD_FEATURES
)

STANDARD_MINIMAL = BASELINE_FEATURES + NODE_ALLOCATABLE_ALLOC_FEATURES

SMALL_FULL = (
    BASELINE_FEATURES
    + SYSTEM_LOAD_FEATURES
    + RESOURCE_ALLOCATABLE_ALLOC_FEATURES
    + RESOURCE_ALLOCATABLE_SYSTEM_FEATURES
    + TEMPORAL_FEATURES
    + USAGE_FEATURES
)

SMALL_BASELINE = (
    BASELINE_FEATURES
    + SYSTEM_LOAD_FEATURES
    + RESOURCE_ALLOCATABLE_ALLOC_FEATURES
    + RESOURCE_ALLOCATABLE_SYSTEM_FEATURES
    + USAGE_FEATURES
)

SMALL_NAIVE = (
    BASELINE_FEATURES
    + SYSTEM_LOAD_FEATURES
    + RESOURCE_ALLOCATABLE_ALLOC_FEATURES
    + RESOURCE_ALLOCATABLE_SYSTEM_FEATURES
)

SMALL_MINIMAL = BASELINE_FEATURES + RESOURCE_ALLOCATABLE_ALLOC_FEATURES


FEATURE_SETS = {
    "standard": {
        "full": STANDARD_FULL,
        "baseline": STANDARD_BASELINE,
        "naive": STANDARD_NAIVE,
        "minimal": STANDARD_MINIMAL,
    },
    "standard-g": {
        "full": STANDARD_FULL,
        "baseline": STANDARD_BASELINE,
        "naive": STANDARD_NAIVE,
        "minimal": STANDARD_MINIMAL,
    },
    "small": {
        "full": SMALL_FULL,
        "baseline": SMALL_BASELINE,
        "naive": SMALL_NAIVE,
        "minimal": SMALL_MINIMAL,
    },
    "small-g": {
        "full": SMALL_FULL + GPU_TOTAL,
        "baseline": SMALL_BASELINE + GPU_TOTAL,
        "naive": SMALL_NAIVE + GPU_TOTAL,
        "minimal": SMALL_MINIMAL + GPU_ALLOC,
    },
}
