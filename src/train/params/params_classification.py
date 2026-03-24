from dataclasses import dataclass, field
from typing import Any

import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

SEED = 49


@dataclass
class ClassifierConfig:
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
    upper_bound: float = 11
    grid_search_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "scoring": {
                # "mae": "neg_mean_absolute_error",
                # "mape": "neg_mean_absolute_percentage_error",
                # "rmse": "neg_root_mean_squared_error",
                # "r2": "r2",
                "f1_macro": "f1_macro",
                "balanced_accuracy": "balanced_accuracy",
                "roc_auc": "roc_auc_ovr_weighted",
                # "roc_auc_ovo": "roc_auc_ovo",
            },
            "refit": "f1_macro",
            # "refit": "mae",
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


TEST_CLASSIFIER_CONFIG = {
    "rf": ClassifierConfig(
        estimator=RandomForestClassifier,
        param_grid={
            "n_estimators": randint(10, 20),  # default = 100
        },
        search_method="random",
        scaling_policy="none",
        log_transform_policy="none",
        sample_weight_method="none",
        cv_strategy="kfold",
        use_permutation_importance=False,
        extra_grid_search_kwargs={"n_iter": 3, "random_state": SEED},
        extra_estimator_kwargs={"verbose": 1, "n_jobs": 4},
    ),
    "xgb": ClassifierConfig(
        estimator=xgb.XGBClassifier,
        param_grid={
            "n_estimators": randint(10, 20),
        },
        search_method="random",
        scaling_policy="none",
        sample_weight_method="none",
        log_transform_policy="none",
        cv_strategy="kfold",
        use_permutation_importance=True,
        extra_grid_search_kwargs={"n_iter": 3, "random_state": SEED},
        extra_estimator_kwargs={"verbosity": 1},
    ),
    "mlp": ClassifierConfig(
        estimator=MLPClassifier,
        param_grid={
            "hidden_layer_sizes": [
                (10,),
                (15,),
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
    "rf": ClassifierConfig(
        estimator=RandomForestClassifier,
        param_grid={
            "n_estimators": randint(80, 200),  # default = 100
            "max_depth": [None] + list(range(5, 20)),  # default = None
            "min_samples_split": randint(2, 10),  # default = 2
            "min_samples_leaf": randint(1, 10),  # default = 1
            "criterion": ["gini", "entropy", "log_loss"],  # default ='squared_error'
            "max_features": ["sqrt", "log2", None, 0.2, 0.5, 1.0],  # default=1.0
        },
        search_method="random",
        scaling_policy="none",
        log_transform_policy="none",
        sample_weight_method="none",
        cv_strategy="kfold",
        use_permutation_importance=False,
        extra_grid_search_kwargs={"n_iter": 50, "random_state": SEED},
        extra_estimator_kwargs={"verbose": 1, "n_jobs": 4},
    ),
    "xgb": ClassifierConfig(
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
        sample_weight_method="none",
        log_transform_policy="none",
        cv_strategy="kfold",
        use_permutation_importance=True,
        extra_grid_search_kwargs={"n_iter": 50, "random_state": SEED},
        extra_estimator_kwargs={"verbosity": 1},
    ),
    "mlp": ClassifierConfig(
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
