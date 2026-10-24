"""Сборка моделей: обработка признаков и оценщик в одном пайплайне"""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.preprocessing import build_preprocessor, get_feature_columns
from src.features import RecipeFeatures, SimpleFeatures


def _log_positive_prices(prices):
    """Логарифм определён только для конечных строго положительных цен."""
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Log target requires finite positive prices")
    return np.log(prices)


def build_model_pipeline(config, X):
    """Вернуть новый необученный Pipeline по конфигу и схеме X.

    X нужен только для проверки названий столбцов: статистики не вычисляются.
    На каждом фолде вызываем эту функцию заново, затем fit на train-части.
    При target_transform=log predict сам возвращает исходные цены через exp.
    """
    data_config = config["data"]
    numeric_columns, categorical_columns = get_feature_columns(
        X,
        config["features"],
        forbidden_columns=[data_config["id_column"], data_config["target"]],
    )
    task = config["task"]
    model_config = config["model"]
    model_name = model_config["name"]
    params = model_config.get("params", {})
    target_transform = model_config.get("target_transform", "none")
    if not isinstance(params, dict):
        raise ValueError("model.params must be a dictionary")
    if target_transform not in ("none", "log"):
        raise ValueError(f"Unknown target transform: {target_transform}")
    if model_name == "stacking":
        from src.stacking import build_stacking_pipeline
        return build_stacking_pipeline(config, X, **params)

    if task == "classification":
        if target_transform != "none":
            raise ValueError("Classification target must not be log-transformed")
        if model_name == "dummy_classifier":
            estimator = DummyClassifier(**params)
        elif model_name == "logistic_regression":
            estimator = LogisticRegression(**params)
        elif model_name == "random_forest":
            estimator = RandomForestClassifier(**params)
        elif model_name == "knn":
            estimator = KNeighborsClassifier(**params)
        elif model_name == "decision_tree":
            estimator = DecisionTreeClassifier(**params)
        elif model_name == "gradient_boosting":
            estimator = GradientBoostingClassifier(**params)
        elif model_name == "hist_gradient_boosting":
            estimator = HistGradientBoostingClassifier(**params)
        elif model_name == "catboost":
            from catboost import CatBoostClassifier
            estimator = CatBoostClassifier(**params)
        elif model_name == "lightgbm":
            from lightgbm import LGBMClassifier
            estimator = LGBMClassifier(**params)
        elif model_name == "xgboost":
            from xgboost import XGBClassifier
            estimator = XGBClassifier(**params)
        elif model_name == "torch_mlp":
            from src.neural import TorchClassifier
            estimator = TorchClassifier(**params)
        else:
            raise ValueError(f"Unknown classification model: {model_name}")
    elif task == "regression":
        if model_name == "dummy_regressor":
            estimator = DummyRegressor(**params)
        elif model_name == "linear_regression":
            estimator = LinearRegression(**params)
        elif model_name == "ridge":
            estimator = Ridge(**params)
        elif model_name == "lasso":
            estimator = Lasso(**params)
        elif model_name == "elastic_net":
            estimator = ElasticNet(**params)
        elif model_name == "knn":
            estimator = KNeighborsRegressor(**params)
        elif model_name == "decision_tree":
            estimator = DecisionTreeRegressor(**params)
        elif model_name == "random_forest":
            estimator = RandomForestRegressor(**params)
        elif model_name == "gradient_boosting":
            estimator = GradientBoostingRegressor(**params)
        elif model_name == "hist_gradient_boosting":
            estimator = HistGradientBoostingRegressor(**params)
        elif model_name == "catboost":
            from catboost import CatBoostRegressor
            estimator = CatBoostRegressor(**params)
        elif model_name == "lightgbm":
            from lightgbm import LGBMRegressor
            estimator = LGBMRegressor(**params)
        elif model_name == "xgboost":
            from xgboost import XGBRegressor
            estimator = XGBRegressor(**params)
        elif model_name == "torch_mlp":
            from src.neural import TorchRegressor
            if target_transform != "log":
                raise ValueError("Torch regression requires target_transform=log")
            estimator = TorchRegressor(**params)
        else:
            raise ValueError(f"Unknown regression model: {model_name}")
        if target_transform == "log" and model_name != "torch_mlp":
            # Меняем шкалу только y, не X. Обратный exp выполняется в predict.
            estimator = TransformedTargetRegressor(
                regressor=estimator,
                func=_log_positive_prices,
                inverse_func=np.exp,
            )
    else:
        raise ValueError(f"Unknown task: {task}")

    simple_features = config.get("feature_engineering", {}).get("simple_features", False)
    if not isinstance(simple_features, bool):
        raise ValueError("feature_engineering.simple_features must be boolean")
    steps = []
    if simple_features:
        transformer = SimpleFeatures(task)
        transformer._validate_input(X)
        numeric_columns.append(transformer.output_column)
        steps.append(("features", transformer))
    recipe = config.get("feature_engineering", {}).get("recipe")
    if recipe:
        transformer = RecipeFeatures(recipe)
        transformer._validate_input(X)
        numeric_columns.extend(transformer.numeric_outputs)
        categorical_columns.extend(transformer.categorical_outputs)
        steps.append(("recipe", transformer))
    encoding = config.get("preprocessing", {}).get("categorical_encoding", "onehot")
    if model_name == "torch_mlp" and params.get("embedding", False):
        if config.get("preprocessing", {}).get("numeric_missing_indicators", False):
            raise ValueError("Embedding recipe uses fixed numeric width, without missing indicators")
        encoding = "ordinal"
        estimator.set_params(numeric_features=len(numeric_columns))
    steps.extend([
        ("preprocessing", build_preprocessor(
            numeric_columns, categorical_columns,
            numeric_missing_indicators=config.get("preprocessing", {}).get("numeric_missing_indicators", False),
            numeric_imputation=config.get("preprocessing", {}).get("numeric_imputation", "median"),
            categorical_encoding=encoding,
        )),
        ("model", estimator),
    ])
    return Pipeline(steps)
