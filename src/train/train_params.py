PARTITION_LIST_FULL = ["small", "small-g", "standard", "standard-g"]
PARTITION_LIST_TEST = ["largemem", "lumid"]
MODELS_LIST_TEST = ["rf", "gb"]

ABLATION_SETS = {
    "largemem": {
        "baseline": [
            "priority",
            "timelimit_minutes",
            "allocated_cpu",
            "allocated_mem",
            "allocated_node",
        ]
    },
    "lumid": {
        "baseline": [
            "priority",
            "timelimit_minutes",
            "allocated_cpu",
            "allocated_mem",
            "allocated_node",
        ]
    },
}
