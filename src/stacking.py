"""Минимальный nested stacking: OOF метапризнаки только внутри outer-train."""

from copy import deepcopy

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline


def build_stacking_pipeline(config, X, *, inner_splits=3, random_state=42,
                            meta_regularization=1.0, meta_model="ridge"):
    """Новый необученный stack из двух полных базовых Pipeline."""
    from src.models import build_model_pipeline, _log_positive_prices

    if type(inner_splits) is not int or inner_splits < 2:
        raise ValueError("Stacking inner_splits must be an integer >=2")
    if type(random_state) is not int or not np.isfinite(meta_regularization) or meta_regularization <= 0:
        raise ValueError("Stacking needs an integer seed and positive meta regularization")
    task = config["task"]
    transform = config["model"].get("target_transform", "none")
    if (task, transform) not in (("classification", "none"), ("regression", "log")):
        raise ValueError("Stacking requires classification/none or regression/log")

    classification = task == "classification"
    if meta_model not in ("ridge", "linear") or (classification and meta_model != "ridge"):
        raise ValueError("meta_model=linear is only available for regression")
    linear_name = "logistic_regression" if classification else "ridge"
    linear_params = ({"C": 1.0, "solver": "lbfgs", "max_iter": 2000}
                     if classification else {"alpha": 1.0})
    base_configs = []
    for name, params in ((linear_name, linear_params),
                         ("hist_gradient_boosting", config["baseline_models"]["hist_gradient_boosting"])):
        base = deepcopy(config)
        # В regression логарифмируем y один раз снаружи всего stack.
        base["model"] = {"name": name, "params": deepcopy(params), "target_transform": "none"}
        base_configs.append(base)
    estimators = [("linear", build_model_pipeline(base_configs[0], X)),
                  ("hgb", build_model_pipeline(base_configs[1], X))]
    splitter = (StratifiedKFold if classification else KFold)(
        n_splits=inner_splits, shuffle=True, random_state=random_state)
    if classification:
        stack = StackingClassifier(
            estimators=estimators, final_estimator=LogisticRegression(
                C=meta_regularization, solver="lbfgs", max_iter=2000),
            cv=splitter, stack_method="predict_proba", passthrough=False, n_jobs=1)
    else:
        stack = StackingRegressor(
            estimators=estimators, final_estimator=(LinearRegression() if meta_model == "linear"
                                                   else Ridge(alpha=meta_regularization)),
            cv=splitter, passthrough=False, n_jobs=1)
        stack = TransformedTargetRegressor(regressor=stack, func=_log_positive_prices,
                                           inverse_func=np.exp)
    # без общего preprocessing перед stack: он находится внутри каждой базы.
    return Pipeline([("model", stack)])
