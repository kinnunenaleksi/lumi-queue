# First round traning: see what algorithm is the best for the baseline features-sets
BASELINE_MODELS = {
    "small": {"models": ["rf", "xgb", "mlp"], "feature_sets": ["baseline"]},
    "small-g": {
        "models": ["rf", "xgb", "mlp"],
        "feature_sets": ["baseline"],
    },
    "standard": {
        "models": ["rf", "xgb", "mlp"],
        "feature_sets": ["baseline"],
    },
    "standard-g": {
        "models": ["rf", "xgb", "mlp"],
        "feature_sets": ["baseline"],
    },
}

# Second round training: using the best algorithm from the first round, train further
# features-sets
ABLATION_MODELS = {
    "small": {"models": ["rf"], "feature_sets": ["perfect"]},
    "small-g": {
        "models": ["rf"],
        "feature_sets": ["perfect"],
    },
    "standard": {
        "models": ["rf"],
        "feature_sets": ["perfect"],
    },
    "standard-g": {
        "models": ["rf"],
        "feature_sets": ["perfect"],
    },
}
