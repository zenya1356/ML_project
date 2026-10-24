"""Сохранение одного завершённого CV-эксперимента"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
import yaml

from src.validation import summarize_cv_results


def validate_run_name(run_name):
    """Проверить безопасное имя нового экспериментна."""
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10))
    if not isinstance(run_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_name):
        raise ValueError("run_name must contain 1-64 letters, digits, underscores or hyphens")
    if run_name.upper() in reserved:
        raise ValueError("run_name is a reserved Windows name")


def _check_save_request(run_name, config, experiment_details, dataset_type):
    """Проверяет имя запуска, описание эксперимента и пару задача/метрика."""
    validate_run_name(run_name)
    if dataset_type not in ("real", "synthetic"):
        raise ValueError("dataset_type must be real or synthetic")
    required = {"model", "model_params", "feature_set", "preprocessing", "target_transform"}
    if not isinstance(experiment_details, dict) or not required.issubset(experiment_details):
        raise ValueError("Experiment details must describe model, params, features, preprocessing and target transform")

    metric = config["metric"]["name"]
    task = config["task"]
    if (task, metric) not in (("classification", "accuracy"), ("regression", "rmse_log")):
        raise ValueError("Task and metric are inconsistent")


def _prediction_tables(config, report, train_ids, test_ids, y_true, folds,
                       oof_predictions, test_predictions_by_fold):
    """Проверяет ID и собирает таблицы результатов без записи на диск."""
    summary = report["summary"]
    for ids, size in ((train_ids, summary["n_train"]), (test_ids, summary["n_test"])):
        values = pd.Series(np.asarray(ids))
        if values.ndim != 1 or len(values) != size or values.isna().any() or not values.is_unique:
            raise ValueError("IDs must be unique, non-missing and match prediction rows")
        if not values.astype(str).is_unique:
            raise ValueError("IDs collide after conversion to strings")
    if isinstance(train_ids, pd.Series) and isinstance(y_true, pd.Series):
        if not train_ids.index.equals(y_true.index):
            raise ValueError("Train IDs and target indices must match")

    id_column = config["data"]["id_column"]
    target_column = config["data"]["target"]
    if id_column == target_column or {id_column, target_column} & {"fold", "prediction", "probability"}:
        raise ValueError("ID/target names collide with result columns")
    oof = pd.DataFrame({
        id_column: np.asarray(train_ids),
        "fold": np.asarray(folds),
        target_column: np.asarray(y_true),
        "prediction": report["oof_labels"],
    })
    test = pd.DataFrame({id_column: np.asarray(test_ids), "prediction": report["test_labels"]})
    if config["task"] == "classification":
        oof["probability"] = np.asarray(oof_predictions)
        test["probability"] = report["test_predictions"]
    per_fold = pd.DataFrame(np.asarray(test_predictions_by_fold),
                            columns=[f"fold_{number}" for number in range(summary["n_splits"])])
    per_fold.insert(0, id_column, np.asarray(test_ids))
    return {
        "fold_scores.csv": report["fold_scores"],
        "oof_predictions.csv": oof,
        "test_predictions.csv": test,
        "test_predictions_by_fold.csv": per_fold,
    }


def _write_experiment(run_name, config, report, tables, experiment_details, dataset_type, fitted_models):
    """Проверяет пути и журнал, затем сохраняет новый каталог результатов."""
    summary = report["summary"]
    base = Path(config["output"]["artifacts_dir"]).resolve()
    run_dir = base / run_name
    if run_dir.resolve().parent != base or run_dir.exists():
        raise FileExistsError("Run directory already exists or points outside artifacts_dir")
    journal = Path(config["output"]["experiments_path"])
    if journal.resolve() == run_dir.resolve() or run_dir in journal.resolve().parents:
        raise ValueError("Experiment journal must be outside the run directory")
    created_at = datetime.now(timezone.utc).isoformat()
    entry = {
        "run_name": run_name,
        "task": config["task"],
        "dataset_type": dataset_type,
        "model": experiment_details["model"],
        "metric": config["metric"]["name"],
        "direction": summary["direction"],
        "cv_mean": summary["cv_mean"],
        "cv_std": summary["cv_std"],
        "oof_score": summary["oof_score"],
        "total_training_seconds": summary["total_training_seconds"],
        "artifact_path": str(run_dir),
        "created_at_utc": created_at,
    }
    entry_frame = pd.DataFrame([entry])
    if journal.exists():
        previous = pd.read_csv(journal, dtype={"run_name": str}, keep_default_na=False)
        if previous.columns.tolist() != entry_frame.columns.tolist():
            raise ValueError("Existing experiment journal has a different schema")
        if run_name in previous["run_name"].values:
            raise FileExistsError("run_name is already present in the experiment journal")

    # Сериализацию проверяем ДО создания каталога, чтобы не оставить полупустой run.
    text_files = {
        "summary.json": json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        "experiment.json": json.dumps(experiment_details, ensure_ascii=False, indent=2, allow_nan=False),
        "config.yaml": yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(run_dir / name, index=False, mode="x")
    for name, content in text_files.items():
        with (run_dir / name).open("x", encoding="utf-8") as file:
            file.write(content)
    model_files = []
    if fitted_models is not None:
        models_dir = run_dir / "models"
        models_dir.mkdir()
        for number, pipeline in enumerate(fitted_models):
            model_path = models_dir / f"fold_{number}.joblib"
            joblib.dump(pipeline, model_path, compress=3)
            model_files.append(model_path.relative_to(run_dir).as_posix())
            if config["model"]["name"] == "torch_mlp":
                import torch
                network = pipeline.named_steps["model"]
                for label in ("best", "last"):
                    checkpoint = dict(getattr(network, f"{label}_checkpoint_"))
                    checkpoint["split"] = network.split_
                    torch.save(checkpoint, models_dir / f"fold_{number}_{label}.pt")
                pd.DataFrame(network.history_).to_csv(
                    models_dir / f"fold_{number}_history.csv", index=False, mode="x")
    files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    checksums = {}
    for file in files:
        checksums[file.relative_to(run_dir).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest = {
        "format_version": 2 if fitted_models is not None else 1,
        "status": "complete",
        "run_name": run_name,
        "dataset_type": dataset_type,
        "created_at_utc": created_at,
        "files_sha256": checksums,
    }
    if fitted_models is not None:
        manifest["model_files"] = model_files
    # Журнал пополняется только после успешной записи всех результатов.
    journal.parent.mkdir(parents=True, exist_ok=True)
    entry_frame.to_csv(journal, index=False, mode="a", header=not journal.exists())
    with (run_dir / "manifest.json").open("x", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, allow_nan=False)
    return run_dir


def save_cv_results(run_name, config, train_ids, test_ids, y_true, folds,
                    oof_predictions, test_predictions_by_fold, fold_training_seconds,
                    experiment_details, prediction_threshold=0.5, dataset_type="real", *, fitted_models=None):
    """Проверяет полный CV, готовит таблицы, сохраняет файлы и обновляет журнал.

    experiment_details описывает model, model_params, feature_set, preprocessing,
    target_transform. Для искусственных проверок обязательно dataset_type='synthetic'.
    Имена запусков уникальны: готовый/частичный каталог никогда не перезаписывается.
    ID, y и predictions должны быть выровнены по порядку. Если запись прервалась,
    отсутствие manifest.json означает незавершённый запуск.
    При передаче fitted_models сохраняются полные Pipeline всех фолдов.
    Загружать joblib допустимо только из доверенных собственных запусков.
    """
    _check_save_request(run_name, config, experiment_details, dataset_type)

    report = summarize_cv_results(
        y_true, folds, oof_predictions, test_predictions_by_fold,
        fold_training_seconds, config["metric"]["name"], prediction_threshold
    )
    summary = report["summary"]
    if config["metric"]["direction"] != summary["direction"]:
        raise ValueError("Metric direction is inconsistent with its definition")
    if config["validation"]["n_splits"] != summary["n_splits"]:
        raise ValueError("Prediction fold count does not match the config")
    if fitted_models is not None:
        if not isinstance(fitted_models, list) or len(fitted_models) != summary["n_splits"]:
            raise ValueError("One fitted model is required per fold")
        if len({id(model) for model in fitted_models}) != len(fitted_models):
            raise ValueError("Each fold requires an independent fitted model")
        for model in fitted_models:
            check_is_fitted(model)

    tables = _prediction_tables(
        config, report, train_ids, test_ids, y_true, folds,
        oof_predictions, test_predictions_by_fold
    )
    return _write_experiment(
        run_name, config, report, tables, experiment_details, dataset_type, fitted_models
    )
