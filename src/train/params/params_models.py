from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

SEED = 49


@dataclass
class ModelConfig:
    estimator: type
    param_grid: dict[str, Any]
    search_method: str
    use_permutation_importance: bool
    scaling_policy: str
    log_transform_policy: str
    sample_weight_method: str = "none"
    cv_strategy: str = "kfold"
    cv_folds: int = 5
    lower_bound: int = 0
    upper_bound: float = np.inf
    grid_search_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "scoring": {
                "mae": "neg_mean_absolute_error",
                "mape": "neg_mean_absolute_percentage_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            "refit": "mae",
            "n_jobs": -1,
        }
    )
    estimator_kwargs: dict[str, Any] = field(default_factory=lambda: {})
    extra_grid_search_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_estimator_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.grid_search_kwargs = {
            **self.grid_search_kwargs,
            **self.extra_grid_search_kwargs,
        }
        self.estimator_kwargs = {**self.estimator_kwargs, **self.extra_estimator_kwargs}


# First round training with RandomizedSearchCV
BASELINE_CONFIGS = {
    "rf": ModelConfig(
        estimator=RandomForestRegressor,
        param_grid={
            "n_estimators": randint(80, 200),  # default = 100
            "max_depth": [None] + list(range(5, 30)),  # default = None
            "min_samples_split": randint(2, 10),  # default = 2
            "min_samples_leaf": randint(1, 10),  # default = 1
            "criterion": ["squared_error"],  # default ='squared_error'
            "max_features": ["sqrt", "log2", None, 0.2, 0.5, 1.0],  # default=1.0
        },
        lower_bound=1,
        upper_bound=300,
        search_method="random",
        scaling_policy="none",
        log_transform_policy="only_target",
        sample_weight_method="aggressive",
        cv_strategy="timeseries",
        use_permutation_importance=False,
        extra_grid_search_kwargs={"n_iter": 50, "random_state": SEED},
        extra_estimator_kwargs={"verbose": 1, "n_jobs": 4},
    ),
    "xgb": ModelConfig(
        estimator=xgb.XGBRegressor,
        param_grid={
            "n_estimators": randint(80, 500),
            "learning_rate": loguniform(1e-3, 3e-1),  # default=0.3
            "subsample": uniform(0.5, 0.5),  # default=1
            "max_depth": randint(3, 30),  # default=6
            "gamma": loguniform(1e-5, 1.0),  # default=0
            "min_child_weight": randint(1, 10),  # default=1
            "colsample_bytree": uniform(0.5, 0.5),
        },
        lower_bound=1,
        upper_bound=1440,
        search_method="random",
        scaling_policy="none",
        sample_weight_method="aggressive",
        log_transform_policy="none",
        cv_strategy="timeseries",
        use_permutation_importance=True,
        extra_grid_search_kwargs={"n_iter": 50, "random_state": SEED},
        extra_estimator_kwargs={"verbosity": 1},
    ),
    "mlp": ModelConfig(
        estimator=MLPRegressor,
        param_grid={
            "hidden_layer_sizes": [
                (50,),
                (100,),
                (50, 50),
                (100, 100),
                (200,),
                (200, 100),
                (200, 200),
                (250,),
                (250, 150),
                (250, 200),
                (250, 250),
            ],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            # "learning_rate": ["constant", "adaptive"],
            "max_iter": [2000],
            "learning_rate_init": [0.0003, 0.001, 0.01],
        },
        search_method="random",
        scaling_policy="all_variables",
        log_transform_policy="all_variables",
        sample_weight_method="aggressive",
        cv_strategy="timeseries",
        use_permutation_importance=True,
        extra_grid_search_kwargs={"n_iter": 30, "random_state": SEED},
        extra_estimator_kwargs={"verbose": 1},
    ),
}

CLASSIFIER_CONFIGS = {
    "rf": ModelConfig(
        estimator=RandomForestClassifier,
        param_grid={
            "n_estimators": randint(80, 200),  # default = 100
            "max_depth": [None] + list(range(5, 30)),  # default = None
            "min_samples_split": randint(2, 10),  # default = 2
            "min_samples_leaf": randint(1, 10),  # default = 1
            "criterion": ["gini", "entropy", "log_loss"],  # default ='squared_error'
            "max_features": ["sqrt", "log2", None, 0.2, 0.5, 1.0],  # default=1.0
        },
        search_method="random",
        scaling_policy="none",
        log_transform_policy="all_variables",
        sample_weight_method="aggressive",
        cv_strategy="timeseries",
        use_permutation_importance=False,
        extra_grid_search_kwargs={"n_iter": 50, "random_state": SEED},
        extra_estimator_kwargs={"verbose": 1, "n_jobs": 4},
    ),
    "xgb": ModelConfig(
        estimator=xgb.XGBClassifier,
        param_grid={
            "n_estimators": randint(80, 500),
            "learning_rate": loguniform(1e-3, 3e-1),  # default=0.3
            "subsample": uniform(0.5, 0.5),  # default=1
            "max_depth": randint(3, 30),  # default=6
            "gamma": loguniform(1e-5, 1.0),  # default=0
            "min_child_weight": randint(1, 10),  # default=1
            "colsample_bytree": uniform(0.5, 0.5),
        },
        search_method="random",
        scaling_policy="none",
        sample_weight_method="aggressive",
        log_transform_policy="all_variables",
        cv_strategy="timeseries",
        use_permutation_importance=True,
        extra_grid_search_kwargs={"n_iter": 50, "random_state": SEED},
        extra_estimator_kwargs={"verbosity": 1},
    ),
    "mlp": ModelConfig(
        estimator=MLPClassifier,
        param_grid={
            "hidden_layer_sizes": [
                (50,),
                (100,),
                (50, 50),
                (100, 100),
                (200,),
                (200, 100),
                (200, 200),
                (250,),
                (250, 150),
                (250, 200),
                (250, 250),
            ],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            # "learning_rate": ["constant", "adaptive"],
            "max_iter": [2000],
            "learning_rate_init": [0.0003, 0.001, 0.01],
        },
        search_method="random",
        scaling_policy="all_variables",
        log_transform_policy="all_variables",
        sample_weight_method="aggressive",
        cv_strategy="timeseries",
        use_permutation_importance=True,
        extra_grid_search_kwargs={"n_iter": 30, "random_state": SEED},
        extra_estimator_kwargs={"verbose": 1},
    ),
}


# Second round training: Halving search near parameters found in the first round
# ABLATION_CONFIG = {
#     "rf": ModelConfig(
#         estimator=RandomForestRegressor,
#         param_grid={
#             "n_estimators": [100, 200],  # default = 100
#             "min_samples_split": [1, 2, 3, 4],  # default = 2
#             "min_samples_leaf": [1, 2, 3, 4],  # default = 1
#             "min_sampls_criterion": ["squared_error"],
#             "max_depth": [None],
#         },
#         use_permutation_importance=False,
#     ),
#     "xgb": ModelConfig(
#         estimator=xgb.XGBRegressor,
#         param_grid={
#             # "n_estimators": [50, 100],
#             # "subsample": [0.8, 1.0],
#             "learning_rate": (0.05, 0.10, 0.15),
#             "max_depth": [3, 5],
#             # "min_child_weigth": [1, 3],
#             "gamma": [0.0, 0.01],
#         },
#         use_permutation_importance=True,
#     ),
#     "gb": ModelConfig(
#         estimator=HistGradientBoostingRegressor,
#         param_grid={
#             "learning_rate": [0.05, 0.1],  # default = 0.1
#             "max_iter": [100, 200],  # default=100
#             "max_leaf_nodes": [None, 31],  # default=31
#             "max_depth": [3, 5, None],
#             "min_samples_leaf": [20],
#         },
#         use_permutation_importance=True,
#     ),
#     "mlp": ModelConfig(
#         estimator=MLPRegressor,
#         param_grid={
#             "hidden_layer_sizes": [(50,), (100,), (50, 50)],
#             "activation": ["relu", "tanh"],
#             "alpha": [0.0001, 0.001, 0.01],
#             "learning_rate": ["constant", "adaptive"],
#             "max_iter": [1000],
#         },
#         use_permutation_importance=True,
#     ),
# }

# Used for unit-tests

# LUMI_TEST_CONFIG = {
#     "rf": ModelConfig(
#         estimator=RandomForestRegressor,
#         # param_grid={
#         #     "n_estimators": [100, 200, 500],
#         # },
#         param_grid={
#             "n_estimators": randint(50, 1000),  # default = 100
#             "max_depth": [None] + list(range(1, 10)),  # default = None
#             "min_samples_split": randint(2, 10),  # default = 2
#             "min_samples_leaf": randint(1, 10),  # default = 1
#             "criterion": ["squared_error"],  # default ='squared_error'
#             "max_features": ["sqrt", "log2", None],  # default='sqrt'
#         },
#         cv_folds=2,
#         search_method="random",
#         scaling_policy="none",
#         log_transform_policy="only_target",
#         use_permutation_importance=False,
#         extra_grid_search_kwargs={"random_state": SEED, "n_iter": 100},
#         extra_estimator_kwargs={"verbose": 3, "random_state": SEED},
#     ),
#     "xgb": ModelConfig(
#         estimator=xgb.XGBRegressor,
#         param_grid={
#             "max_depth": [1, 5, 10],
#             "n_estimators": [100, 300, 500, 1000],
#         },
#         cv_folds=2,
#         search_method="random",
#         scaling_policy="none",
#         log_transform_policy="only_target",
#         use_permutation_importance=True,
#         extra_grid_search_kwargs={"random_state": SEED},
#         extra_estimator_kwargs={"verbosity": 3, "seed": SEED},
#     ),
# }
TEST_MODEL_CONFIGS = {
    "rf": ModelConfig(
        estimator=RandomForestRegressor,
        param_grid={
            "n_estimators": [10, 20],
            "max_features": ["sqrt", "log2", None, 0.2, 0.5, 1.0],  # default=1.0
        },
        cv_folds=2,
        search_method="random",
        scaling_policy="none",
        lower_bound=1,
        upper_bound=1000,
        sample_weight_method="aggressive",
        log_transform_policy="all_variables",
        use_permutation_importance=False,
        extra_grid_search_kwargs={"n_iter": 2, "random_state": SEED},
        extra_estimator_kwargs={"n_jobs": 2, "verbose": 3, "random_state": SEED},
    ),
    "xgb": ModelConfig(
        estimator=xgb.XGBRegressor,
        param_grid={
            "max_depth": [1, 3],
            "n_estimators": [10, 20],
        },
        cv_folds=2,
        search_method="random",
        scaling_policy="none",
        lower_bound=1,
        upper_bound=1000,
        log_transform_policy="only_target",
        sample_weight_method="aggressive",
        use_permutation_importance=True,
        extra_grid_search_kwargs={"n_iter": 2, "random_state": SEED},
        extra_estimator_kwargs={"n_jobs": 2, "verbosity": 3, "seed": SEED},
    ),
}
