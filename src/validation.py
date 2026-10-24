"""CV-фолды, метрики и сбор предсказаний."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def _split_parameters(validation_config):
    """Отделяем параметры split и проверяем их."""
    params = {
        name: validation_config[name]
        for name in ("strategy", "n_splits", "shuffle", "random_state")
    }
    if params["strategy"] not in ("stratified_kfold", "kfold"):
        raise ValueError(f"Unknown validation strategy: {params['strategy']}")
    if type(params["n_splits"]) is not int or params["n_splits"] < 2:
        raise ValueError("n_splits must be an integer >= 2")
    if type(params["shuffle"]) is not bool:
        raise ValueError("shuffle must be a boolean")
        
    # Для воспроизводимости используем рандомное зерно.
    seed = params["random_state"]
    if params["shuffle"] and (
        type(seed) is not int or not 0 <= seed < 2**32
    ):
        raise ValueError("Shuffled folds require an integer random_state in [0, 2**32)")
    if not params["shuffle"]:
        params["random_state"] = None
    return params


def validate_folds(folds, n_rows, n_splits):
    """Каждой позиции назначен один допустимый номер validation-фолда."""
    folds = np.asarray(folds)
    if folds.ndim != 1 or len(folds) != n_rows:
        raise ValueError("folds must be a one-dimensional array with one entry per row")
    if not np.issubdtype(folds.dtype, np.integer):
        raise ValueError("Fold numbers must be integers")
    if not np.array_equal(np.unique(folds), np.arange(n_splits)):
        raise ValueError("Every fold from 0 to n_splits - 1 must be present")
    sizes = np.bincount(folds, minlength=n_splits)
    if sizes.max() - sizes.min() > 1:
        raise ValueError("KFold/StratifiedKFold validation sizes must differ by at most 1")


def make_folds(X, y, validation_config):
    """Возвращает номер проверочного фолда для каждой позиции строки X."""
    params = _split_parameters(validation_config)
    target = np.asarray(y)
    if target.ndim != 1 or len(X) != len(target):
        raise ValueError("X and one-dimensional y must have the same number of rows")
    if pd.isna(target).any():
        raise ValueError("Target contains missing values")
    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        if not X.index.equals(y.index):
            raise ValueError("X and y indices must match in the same order")
    if params["n_splits"] > len(X):
        raise ValueError("n_splits cannot exceed the number of rows")

    if params["strategy"] == "stratified_kfold":
        # Нужно достаточно объектов каждого класса для всех фолдов.
        counts = pd.Series(target).value_counts()
        if len(counts) < 2 or counts.min() < params["n_splits"]:
            raise ValueError("Stratification requires >= 2 classes and >= n_splits rows per class")
        splitter_class = StratifiedKFold
    else:
        splitter_class = KFold

    splitter = splitter_class(
        n_splits=params["n_splits"],
        shuffle=params["shuffle"],
        random_state=params["random_state"],
    )
    # -1 показывает, что строке пока не назначен фолд
    folds = np.full(len(X), -1, dtype=int)
    for fold_number, (_, valid_indices) in enumerate(splitter.split(X, y)):
        folds[valid_indices] = fold_number

    validate_folds(folds, len(X), params["n_splits"])
    return folds


def _prepare_fold_metadata(X, y, ids, params):
    """Проверяет ID/таргет и возвращает строковые ID и описание split."""
    if not isinstance(ids, pd.Series) or not ids.name or ids.name == "fold":
        raise ValueError("IDs must be a named Series with a name other than 'fold'")
    if len(ids) != len(X) or len(y) != len(X):
        raise ValueError("IDs, X and y must have the same length")
    if ids.isna().any() or not ids.is_unique:
        raise ValueError("IDs must be unique and non-missing")
    if isinstance(X, pd.DataFrame) and not X.index.equals(ids.index):
        raise ValueError("IDs and X indices must match in the same order")
    if isinstance(y, pd.Series) and not y.index.equals(ids.index):
        raise ValueError("IDs and y indices must match in the same order")
    target = np.asarray(y)
    if target.ndim != 1 or pd.isna(target).any():
        raise ValueError("Target must be one-dimensional and non-missing")

    # Текстовое представление ID
    keys = ids.astype(str).reset_index(drop=True)
    if not keys.is_unique:
        raise ValueError("IDs collide after conversion to strings")
    identity = pd.DataFrame({"id": keys, "target": target}).sort_values("id")
    json_text = identity.to_json(orient="records", double_precision=15)
    data_bytes = json_text.encode("utf-8")
    fingerprint = hashlib.sha256(data_bytes).hexdigest()
    metadata = {
        "format_version": 1,
        "id_column": ids.name,
        "validation": params,
        "n_rows": len(X),
        "id_target_sha256": fingerprint,
    }
    return keys, metadata


def _load_folds(path, keys, expected):
    """Проверяет сохранённый split и выстраивает фолды в текущем порядке ID."""
    metadata_path = path.with_suffix(".json")
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError("Saved folds do not match IDs, target or split config; use a new folds_path")
    csv_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if metadata.get("folds_csv_sha256") != csv_hash:
        raise ValueError("Saved folds CSV checksum does not match its metadata")

    id_column = expected["id_column"]
    saved = pd.read_csv(path, dtype={id_column: str}, keep_default_na=False)
    if saved.columns.tolist() != [id_column, "fold"]:
        raise ValueError("Saved folds must contain exactly the ID column and fold")
    if len(saved) != len(keys) or not saved[id_column].is_unique:
        raise ValueError("Saved folds contain missing or duplicate IDs")
    if set(saved[id_column]) != set(keys):
        raise ValueError("Saved fold IDs differ from current training IDs")

    # map, а не позиционное копирование: порядок текущих строк может измениться.
    fold_by_id = saved.set_index(id_column)["fold"]
    folds = keys.map(fold_by_id).to_numpy()
    validate_folds(folds, len(keys), expected["validation"]["n_splits"])
    return folds


def _save_folds(path, keys, folds, metadata):
    """Создаёт CSV и JSON без перезаписи существующих файлов."""
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = pd.DataFrame({metadata["id_column"]: keys, "fold": folds})
    saved.to_csv(path, index=False, mode="x")
    # Не изменяем переданный словарь: контрольную сумму добавляем в копию.
    metadata = metadata.copy()
    metadata["folds_csv_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path = path.with_suffix(".json")
    with metadata_path.open("x", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def get_or_create_folds(X, y, ids, validation_config, folds_path):
    """Загружает совместимый split или создаёт новый; возвращает (folds, created).

    CSV содержит ID и fold. JSON рядом хранит параметры, отпечаток ID/таргета
    и контрольную сумму CSV. Несовместимые/повреждённые файлы не перезаписываем.
    """
    params = _split_parameters(validation_config)
    keys, metadata = _prepare_fold_metadata(X, y, ids, params)

    path = Path(folds_path)
    metadata_path = path.with_suffix(".json")
    if path.suffix.lower() != ".csv":
        raise ValueError("folds_path must end in .csv")
    if path.exists() != metadata_path.exists():
        raise ValueError("Incomplete saved split: both folds CSV and JSON are required")

    if path.exists():
        folds = _load_folds(path, keys, metadata)
        return folds, False

    folds = make_folds(X, y, validation_config)
    _save_folds(path, keys, folds, metadata)
    return folds, True


def summarize_folds(folds, y, strategy):
    """Размеры частей и баланс классов для вывода в терминал."""
    target = np.asarray(y)
    rows = []
    for fold_number in np.unique(folds):
        valid_target = target[folds == fold_number]
        row = {
            "fold": int(fold_number),
            "train_rows": len(target) - len(valid_target),
            "valid_rows": len(valid_target),
        }
        if strategy == "stratified_kfold":
            for label in np.unique(target):
                count = int((valid_target == label).sum())
                row[f"class_{label}_count"] = count
                row[f"class_{label}_share"] = count / len(valid_target)
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_metric(y_true, y_pred, metric_name):
    """Считает метрику по готовым предсказаниям."""
    # Series сохраняет индекс
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        if not y_true.index.equals(y_pred.index):
            raise ValueError("True and predicted Series indices must match in the same order")

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Targets and predictions must be one-dimensional")
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Targets and predictions must have equal non-zero length")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Targets and predictions must be finite")

    if metric_name == "accuracy":
        # Нельзя молча принять вероятности за классы или подобрать здесь порог.
        if not np.isin(y_true, [0, 1]).all():
            raise ValueError("True labels must be 0 or 1")
        if not np.isin(y_pred, [0, 1]).all():
            raise ValueError("Predicted labels must be 0 or 1")
        # True считается как 1, False как 0; среднее равно доле верных ответов.
        return float(np.mean(y_true == y_pred))

    if metric_name == "rmse_log":
        if (y_true <= 0).any() or (y_pred <= 0).any():
            raise ValueError("Prices must be positive for logarithms")
        # Не исправляем плохие предсказания clipping-ом внутри метрики.
        errors = np.log(y_true) - np.log(y_pred)
        return float(np.sqrt(np.mean(errors ** 2)))

    raise ValueError(f"Unknown metric: {metric_name}")


def initialize_cv_predictions(n_train, n_test, n_splits):
    """NaN отмечает места, куда ещё не записаны прогнозы."""
    if n_train < 1 or n_test < 1 or not 2 <= n_splits <= n_train:
        raise ValueError("Invalid train/test sizes or fold count")
    return (np.full(n_train, np.nan), np.full((n_test, n_splits), np.nan))


def record_fold_predictions(oof_predictions, test_predictions_by_fold, folds,
                            fold_number, valid_predictions, test_predictions):
    """Записать validation-прогнозы на их места, test — в столбец фолда."""
    n_splits = test_predictions_by_fold.shape[1]
    if not 0 <= fold_number < n_splits:
        raise ValueError("fold_number is outside the split")
    positions = np.flatnonzero(np.asarray(folds) == fold_number)
    valid = np.asarray(valid_predictions, dtype=float)
    test = np.asarray(test_predictions, dtype=float)
    if valid.shape != (len(positions),) or test.shape != (len(test_predictions_by_fold),):
        raise ValueError("Prediction count does not match validation/test rows")
    if not np.isfinite(valid).all() or not np.isfinite(test).all():
        raise ValueError("Fold predictions must be finite")
    if not np.isnan(oof_predictions[positions]).all() or not np.isnan(test_predictions_by_fold[:, fold_number]).all():
        raise ValueError("This fold was already recorded")
    oof_predictions[positions] = valid
    test_predictions_by_fold[:, fold_number] = test


def summarize_cv_results(y_true, folds, oof_predictions, test_predictions_by_fold,
                         fold_training_seconds, metric_name, prediction_threshold=0.5):
    """Проверить полноту CV, посчитать fold/OOF scores и усреднить test."""
    if isinstance(y_true, pd.Series) and isinstance(oof_predictions, pd.Series):
        if not y_true.index.equals(oof_predictions.index):
            raise ValueError("Target and OOF indices must match")
    target = np.asarray(y_true, dtype=float)
    oof = np.asarray(oof_predictions, dtype=float)
    test = np.asarray(test_predictions_by_fold, dtype=float)
    times = np.asarray(fold_training_seconds, dtype=float)
    if target.ndim != 1 or oof.shape != target.shape or len(target) == 0:
        raise ValueError("Target and OOF must have the same one-dimensional shape")
    if test.ndim != 2 or len(test) == 0:
        raise ValueError("Test predictions must have shape (n_test, n_splits)")
    n_splits = test.shape[1]
    validate_folds(folds, len(target), n_splits)
    if times.shape != (n_splits,) or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("One finite non-negative training time is required per fold")
    if not np.isfinite(oof).all() or not np.isfinite(test).all():
        raise ValueError("CV is incomplete or contains non-finite predictions")

    # Сначала среднее вероятностей/цен, только потом порог для классификации.
    averaged_test = test.mean(axis=1)
    if not np.isfinite(averaged_test).all():
        raise ValueError("Prediction averaging overflowed")
    if metric_name == "accuracy":
        if not 0 <= prediction_threshold <= 1:
            raise ValueError("Prediction threshold must be in [0, 1]")
        if ((oof < 0) | (oof > 1)).any() or ((test < 0) | (test > 1)).any():
            raise ValueError("Classification probabilities must be in [0, 1]")
        oof_labels = (oof >= prediction_threshold).astype(int)
        test_labels = (averaged_test >= prediction_threshold).astype(int)
        threshold = float(prediction_threshold)
    elif metric_name == "rmse_log":
        if (oof <= 0).any() or (test <= 0).any():
            raise ValueError("Regression predictions must be positive prices")
        oof_labels, test_labels, threshold = oof, averaged_test, None
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    rows = []
    folds = np.asarray(folds)
    for number in range(n_splits):
        mask = folds == number
        rows.append({
            "fold": number,
            "valid_rows": int(mask.sum()),
            "score": calculate_metric(target[mask], oof_labels[mask], metric_name),
            "training_seconds": float(times[number]),
        })
    scores = np.array([row["score"] for row in rows])
    return {
        "fold_scores": pd.DataFrame(rows),
        "summary": {
            "metric": metric_name,
            "direction": "maximize" if metric_name == "accuracy" else "minimize",
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std(ddof=0)),
            "cv_std_ddof": 0,
            "oof_score": calculate_metric(target, oof_labels, metric_name),
            "n_train": len(target),
            "n_test": len(test),
            "n_splits": n_splits,
            "total_training_seconds": float(times.sum()),
            "prediction_threshold": threshold,
            "prediction_scale": "probability_class_1" if metric_name == "accuracy" else "original_price",
        },
        "oof_labels": oof_labels,
        "test_predictions": averaged_test,
        "test_labels": test_labels,
    }
