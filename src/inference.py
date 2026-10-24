"""Прогноз сохранёнными Pipeline и запись submission. Обучения здесь нет."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
import yaml

from src.data import read_housing_csv
from src.preprocessing import get_feature_columns


def read_run_metadata(run_dir):
    """Проверить полный v2 run и прочитать metadata, не загружая joblib."""
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("format_version") != 2:
        raise ValueError("A complete v2 run with saved models is required")
    checksums = manifest.get("files_sha256")
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError("Missing artifact checksums")
    for name, expected in checksums.items():
        # Не разрешаем manifest читать файл за пределами каталога запуска.
        if not isinstance(name, str) or "\\" in name or ":" in name:
            raise ValueError("Invalid artifact path")
        path = (run_dir / name).resolve()
        if not path.is_relative_to(run_dir) or not path.is_file():
            raise ValueError("Artifact is missing or points outside run_dir")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    if not {"config.yaml", "summary.json", "experiment.json"}.issubset(checksums):
        raise ValueError("Run metadata must be covered by checksums")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    details = json.loads((run_dir / "experiment.json").read_text(encoding="utf-8"))
    if config.get("run", {}).get("mode") != "train" or details.get("run_mode") != "train":
        raise ValueError("Submission requires a full train run, not debug")
    task = config["task"]
    if task not in ("classification", "regression") or details["model"] != config["model"]["name"]:
        raise ValueError("Inconsistent task/model metadata")
    count = config["validation"]["n_splits"]
    if type(count) is not int or count < 2 or summary["n_splits"] != count:
        raise ValueError("Inconsistent model/fold count")
    names = [f"models/fold_{number}.joblib" for number in range(count)]
    expected_files = set(names) | {"config.yaml", "summary.json", "experiment.json", "fold_scores.csv",
                                   "oof_predictions.csv", "test_predictions.csv", "test_predictions_by_fold.csv"}
    if config["model"]["name"] == "torch_mlp":
        expected_files.update(f"models/fold_{number}_{suffix}" for number in range(count)
                              for suffix in ("best.pt", "last.pt", "history.csv"))
    if manifest.get("model_files") != names or set(checksums) != expected_files:
        raise ValueError("Manifest must contain one distinct model per fold and all result files")
    current_versions = {"python": platform.python_version(),
                        "numpy": version("numpy"), "pandas": version("pandas"),
                        "scikit-learn": version("scikit-learn"), "joblib": version("joblib")}
    if (config["model"]["name"] == "torch_mlp"
            or (config["model"]["name"] == "equal_average" and "torch" in details.get("versions", {}))):
        current_versions["torch"] = version("torch")
    for package in ("catboost", "lightgbm", "xgboost"):
        if (config["model"]["name"] == package
                or (config["model"]["name"] == "equal_average"
                    and package in details.get("versions", {}))):
            current_versions[package] = version(package)
    if details.get("versions") != current_versions:
        raise ValueError("Recorded package versions differ; restore the training environment before loading")
    if task == "classification":
        threshold = config.get("training", {}).get("prediction_threshold", 0.5)
        if summary["prediction_threshold"] != threshold:
            raise ValueError("Prediction threshold differs from the saved config")

    return config, summary, details, manifest


def load_fitted_run(run_dir, *, trusted=False):
    """Возвращает metadata и полные Pipeline."""
    if trusted is not True:
        raise ValueError("Only load trusted own models; explicitly pass trusted=True")
    config, summary, _, manifest = read_run_metadata(run_dir)
    run_dir = Path(run_dir).resolve()
    models = []
    for name in manifest["model_files"]:
        model = joblib.load(run_dir / name)
        if not isinstance(model, Pipeline):
            raise ValueError("Expected a full sklearn Pipeline, not only an estimator")
        check_is_fitted(model)
        models.append(model)
    return config, summary, models


def predict_folds(models, X, task, prediction_threshold=0.5, *, allow_single=False):
    """Возвращает метки 0/1 или цены, усреднив прогнозы независимых fold-моделей.
    X - DataFrame в исходном порядке столбцов без ID/target.
    Pipeline сам выполняет transform; fit/fit_transform здесь не нужны.
    """
    if not isinstance(X, pd.DataFrame) or X.empty or not X.columns.is_unique:
        raise ValueError("X must be a non-empty DataFrame with unique columns")
    if not isinstance(models, list) or len(models) < (1 if allow_single else 2):
        raise ValueError("Expected at least two fold models (or explicitly enabled single-fold inference)")
    if task not in ("classification", "regression"):
        raise ValueError("Unknown prediction task")
    if task == "classification" and (not np.isfinite(prediction_threshold) or not 0 <= prediction_threshold <= 1):
        raise ValueError("Prediction threshold must be in [0, 1]")

    predictions = []
    for model in models:
        check_is_fitted(model)
        if list(model.feature_names_in_) != X.columns.tolist():
            raise ValueError("Test feature names/order differ from the fitted Pipeline")
        if task == "classification":
            if not np.array_equal(model.classes_, [0, 1]):
                raise ValueError("Expected classifier classes [0, 1]")
            prediction = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
        else:
            # При log-target внешний predict уже применяет exp: второй exp не нужен.
            prediction = np.asarray(model.predict(X), dtype=float)
        if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
            raise ValueError("Model predictions must be finite and match test rows")
        if task == "classification" and ((prediction < 0).any() or (prediction > 1).any()):
            raise ValueError("Probabilities must be in [0, 1]")
        if task == "regression" and (prediction <= 0).any():
            raise ValueError("Predicted prices must be positive")
        predictions.append(prediction)
    # Матрица: строки — test-объекты, столбцы — модели фолдов.
    with np.errstate(over="ignore", invalid="ignore"):
        averaged = np.column_stack(predictions).mean(axis=1)
    if not np.isfinite(averaged).all():
        raise ValueError("Prediction averaging overflowed")
    if task == "classification":
        return (averaged >= prediction_threshold).astype(int)
    return averaged


def make_submission(run_dir, *, trusted=False, output_path=None, test_path=None, fold_index=None):
    """Загружает полный run, предсказывает competition test и записать новый CSV.
    Существующий файл не перезаписывается. Возвращается абсолютный Path.
    """
    config, summary, models = load_fitted_run(run_dir, trusted=trusted)
    if fold_index is not None:
        if type(fold_index) is not int or not 0 <= fold_index < len(models):
            raise ValueError("fold_index must identify an existing saved model")
        models = [models[fold_index]]
    data = config["data"]
    output = Path(output_path if output_path is not None else config["output"]["submission_path"]).resolve()
    if output.exists():
        raise FileExistsError(f"Submission already exists; choose a new path: {output}")
    if output.is_relative_to(Path(run_dir).resolve()):
        raise ValueError("Submission must be outside the immutable training run")
    mode = data.get("parsing_mode", "default")
    if mode not in ("default", "housing"):
        raise ValueError("Unknown CSV parsing mode")
    reader = read_housing_csv if mode == "housing" else pd.read_csv
    test = reader(test_path if test_path is not None else data["test_path"])
    id_column, target_column = data["id_column"], data["target"]
    if test.empty or not test.columns.is_unique or id_column not in test or target_column in test:
        raise ValueError("Test requires unique columns, ID and no target")
    ids = test[id_column]
    if ids.isna().any() or not ids.is_unique:
        raise ValueError("Test IDs must be unique and non-missing")
    X = test.drop(columns=[id_column])
    get_feature_columns(X, config["features"], forbidden_columns=[id_column, target_column])
    template = pd.read_csv(data["submission_template_path"])
    if template.columns.tolist() != [id_column, target_column]:
        raise ValueError("Template columns must be exactly ID and target")
    if template[id_column].isna().any() or not template[id_column].is_unique:
        raise ValueError("Template IDs must be unique and non-missing")
    if not np.array_equal(ids.to_numpy(), template[id_column].to_numpy()):
        raise ValueError("Template and test IDs must match in the same order")
    threshold = summary["prediction_threshold"] if config["task"] == "classification" else 0.5
    predictions = predict_folds(models, X, config["task"], threshold, allow_single=fold_index is not None)
    submission = pd.DataFrame({id_column: ids.to_numpy(), target_column: predictions})
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False, mode="x")
    return output
