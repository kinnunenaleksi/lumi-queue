from dataclasses import dataclass, field
from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb


@dataclass
class ModelConfig:
    estimator: type
    param_grid: dict[str, Any]
    cv_folds: int = 5
    use_permutation_importance: bool = False
    grid_search_kwargs: dict[str, Any] = field(default_factory=dict)
    estimator_kwargs: dict[str, Any] = field(default_factory=dict)


MODEL_CONFIGS = {
    "rf": ModelConfig(
        estimator=RandomForestRegressor,
        estimator_kwargs={"n_jobs": -1},
        param_grid={
            "n_estimators": [100, 200],
            "criterion": ["squared_error"],
            "max_depth": [None],
        },
        cv_folds=5,
        use_permutation_importance=False,
        grid_search_kwargs={"scoring": "neg_mean_squared_error", "refit": True},
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
        cv_folds=5,
        use_permutation_importance=True,
        estimator_kwargs={"n_jobs": -1},
        grid_search_kwargs={"scoring": "neg_mean_squared_error", "refit": True},
    ),
    "gb": ModelConfig(
        estimator=HistGradientBoostingRegressor,
        param_grid={
            "max_iter": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5, None],
            "min_samples_leaf": [20],
        },
        cv_folds=5,
        use_permutation_importance=True,
        grid_search_kwargs={},
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
        cv_folds=5,
        use_permutation_importance=True,
        grid_search_kwargs={"scoring": "neg_mean_squared_error", "refit": True},
    ),
}

TEST_MODEL_CONFIGS = {
    "rf": ModelConfig(
        estimator=RandomForestRegressor,
        estimator_kwargs={"n_jobs": -1},
        param_grid={
            "n_estimators": [10, 20],
            "criterion": ["squared_error"],
            "max_depth": [None],
        },
        cv_folds=3,
        use_permutation_importance=False,
        grid_search_kwargs={"scoring": "neg_mean_squared_error", "refit": True},
    ),
    "xgb": ModelConfig(
        estimator=xgb.XGBRegressor,
        param_grid={
            "max_depth": [1, 3],
        },
        cv_folds=3,
        use_permutation_importance=True,
        estimator_kwargs={"n_jobs": -1},
        grid_search_kwargs={"scoring": "neg_mean_squared_error", "refit": True},
    ),
    "gb": ModelConfig(
        estimator=HistGradientBoostingRegressor,
        param_grid={
            "learning_rate": [0.05, 0.1],
        },
        cv_folds=5,
        use_permutation_importance=True,
        grid_search_kwargs={},
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
        cv_folds=5,
        use_permutation_importance=True,
        grid_search_kwargs={"scoring": "neg_mean_squared_error", "refit": True},
    ),
}
