from dataclasses import dataclass, field
from typing import Any

import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor

SEED = 49


@dataclass
class ModelConfig:
    estimator: type
    param_grid: dict[str, Any]
    cv_folds: int = 5
    use_permutation_importance: bool = False
    scaling_policy: str = "none"
    log_transform_policy: str = "all_variables"
    search_method: str = "grid"
    grid_search_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"scoring": "neg_mean_squared_error", "refit": True}
    )
    estimator_kwargs: dict[str, Any] = field(default_factory=lambda: {"n_jobs": -1})


MODEL_CONFIGS = {
    "rf": ModelConfig(
        estimator=RandomForestRegressor,
        estimator_kwargs={"n_jobs": -1},
        param_grid={
            "n_estimators": [100, 200],
            "criterion": ["squared_error"],
            "max_depth": [None],
        },
        use_permutation_importance=False,
    ),
    "xgb": ModelConfig(
        estimator=xgb.XGBRegressor,
        param_grid={
            # "n_estimators": [50, 100],
            # "subsample": [0.8, 1.0],
            "learning_rate": (0.05, 0.10, 0.15),
            "max_depth": [3, 5],
            # "min_child_weigth": [1, 3],
            "gamma": [0.0, 0.01],
        },
        use_permutation_importance=True,
    ),
    "gb": ModelConfig(
        estimator=HistGradientBoostingRegressor,
        param_grid={
            "max_iter": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5, None],
            "min_samples_leaf": [20],
        },
        use_permutation_importance=True,
    ),
    "mlp": ModelConfig(
        estimator=MLPRegressor,
        param_grid={
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "adaptive"],
            "max_iter": [1000],
        },
        use_permutation_importance=True,
    ),
}

TEST_MODEL_CONFIGS = {
    "rf": ModelConfig(
        estimator=RandomForestRegressor,
        estimator_kwargs={"n_jobs": -1},
        param_grid={
            "n_estimators": [10, 20],
        },
        cv_folds=2,
        use_permutation_importance=False,
    ),
    "xgb": ModelConfig(
        estimator=xgb.XGBRegressor,
        param_grid={
            "max_depth": [1, 3],
        },
        cv_folds=2,
        use_permutation_importance=True,
    ),
}
