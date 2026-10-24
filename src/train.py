"""Обучение на сохранённых фолдах. Здеьс файлы не записываются"""

from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.models import build_model_pipeline
from src.validation import (
    initialize_cv_predictions,
    record_fold_predictions,
    summarize_cv_results,
    validate_folds,
)


def select_debug_positions(folds, rows_per_fold=20, random_state=42):
    """Выбрать немного строк из каждого исходного фолда, не создавая новый split."""
    if type(rows_per_fold) is not int or rows_per_fold < 2:
        raise ValueError("Debug requires at least two rows per fold")
    folds = np.asarray(folds)
    if folds.ndim != 1 or len(folds) == 0 or len(np.unique(folds)) < 2:
        raise ValueError("Debug folds must be a non-empty 1D array with >=2 folds")
    validate_folds(folds, len(folds), len(np.unique(folds)))
    rng = np.random.default_rng(random_state)
    selected = []
    for number in np.unique(folds):
        positions = np.flatnonzero(folds == number)
        count = min(rows_per_fold, len(positions))
        selected.extend(rng.choice(positions, size=count, replace=False).tolist())
    # Исходный порядок сохраняется для X, y, ID и номеров фолдов.
    return np.sort(np.asarray(selected, dtype=int))


def train_cv(config, X, y, X_test, folds, prediction_threshold=0.5):
    """Обучает отдельный Pipeline на каждом фолде и возвращает полный результат CV.

    X/X_test — DataFrame без ID и target, y — Series в порядке X.
    Для классификации буферы содержат вероятность класса 1; для регрессии —
    исходную цену. Время включает fit preprocessing и модели, но не predict.
    Файлы не записываются, входные таблицы и конфиг не меняются.
    Для torch_mlp внутри outer-train выделяется stopping-часть, preprocessing
    обучается лишь на inner-fit. Внешний фолд не выбирает эпоху.
    """
    if not isinstance(X, pd.DataFrame) or not isinstance(X_test, pd.DataFrame) or not isinstance(y, pd.Series):
        raise ValueError("X/X_test must be DataFrames and y must be a Series")
    if X.empty or X_test.empty or len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("Training rows and target must be non-empty and aligned")
    if not X.columns.equals(X_test.columns):
        raise ValueError("Train/test feature columns must match in the same order")
    n_splits = config["validation"]["n_splits"]
    if type(n_splits) is not int or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2")
    folds = np.asarray(folds)
    validate_folds(folds, len(X), n_splits)
    target = np.asarray(y, dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("Target must be finite")
    task, metric = config["task"], config["metric"]["name"]
    if (task, metric) not in (("classification", "accuracy"), ("regression", "rmse_log")):
        raise ValueError("Task and metric are inconsistent")
    expected_direction = "maximize" if task == "classification" else "minimize"
    if config["metric"]["direction"] != expected_direction:
        raise ValueError("Metric direction is inconsistent")
    if task == "classification":
        if not np.array_equal(np.unique(target), [0, 1]):
            raise ValueError("Classification target must contain both classes 0 and 1")
        if not np.isfinite(prediction_threshold) or not 0 <= prediction_threshold <= 1:
            raise ValueError("Prediction threshold must be in [0, 1]")
    elif (target <= 0).any():
        raise ValueError("Regression prices must be positive")

    oof, test_by_fold = initialize_cv_predictions(len(X), len(X_test), n_splits)
    fitted_models, training_seconds = [], []
    for number in range(n_splits):
        train_positions = np.flatnonzero(folds != number)
        valid_positions = np.flatnonzero(folds == number)
        if task == "classification" and not np.array_equal(np.unique(target[train_positions]), [0, 1]):
            raise ValueError("Each training fold needs both classes; increase the debug sample")

        # Новый объект на каждом проходе: imputer/encoder/scaler не разделяют состояние.
        pipeline = build_model_pipeline(config, X)
        started = perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            if config["model"]["name"] == "torch_mlp":
                from src.neural import fit_torch_fold
                seed = config["model"]["params"].get("random_state", 42) + number
                fit_torch_fold(pipeline, X.iloc[train_positions], y.iloc[train_positions], seed)
            else:
                pipeline.fit(X.iloc[train_positions], y.iloc[train_positions])
        elapsed = perf_counter() - started

        if task == "classification":
            if not np.array_equal(pipeline.classes_, [0, 1]):
                raise ValueError("Unexpected classifier class order")
            valid_prediction = pipeline.predict_proba(X.iloc[valid_positions])[:, 1]
            test_prediction = pipeline.predict_proba(X_test)[:, 1]
        else:
            valid_prediction = pipeline.predict(X.iloc[valid_positions])
            test_prediction = pipeline.predict(X_test)

        record_fold_predictions(oof, test_by_fold, folds, number, valid_prediction, test_prediction)
        fitted_models.append(pipeline)
        training_seconds.append(elapsed)
        if config["model"]["name"] == "torch_mlp":
            network = pipeline.named_steps["model"]
            print(f"Fold {number}: best epoch {network.best_epoch_}, last epoch {network.last_epoch_}", flush=True)

    report = summarize_cv_results(
        y, folds, oof, test_by_fold, training_seconds, metric, prediction_threshold
    )
    return {
        "oof_predictions": oof,
        "test_predictions_by_fold": test_by_fold,
        "fold_training_seconds": training_seconds,
        "fitted_models": fitted_models,
        "report": report,
    }
