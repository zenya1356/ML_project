"""Равные веса сохранённых моделей: OOF-сравнение и Pipeline без нового fit."""

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.inference import load_fitted_run, read_run_metadata
from src.utils import save_cv_results, validate_run_name
from src.validation import summarize_cv_results


class FrozenAverage(BaseEstimator):
    """Хранит уже обученные Pipeline. Повторное обучение намеренно запрещено."""

    def __init__(self, models):
        self.models = models

    def fit(self, X, y=None):
        raise RuntimeError("Frozen average uses saved models; do not fit it again")

    def _mean(self, X):
        check_is_fitted(self, "feature_names_in_")
        if not isinstance(X, pd.DataFrame) or X.columns.tolist() != self.feature_names_in_.tolist():
            raise ValueError("Feature names/order differ from the saved average")
        predictions = []
        for model in self.models:
            prediction = (model.predict_proba(X)[:, 1] if self.task == "classification"
                          else model.predict(X))
            prediction = np.asarray(prediction, dtype=float)
            if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
                raise ValueError("Invalid member prediction")
            if self.task == "classification" and ((prediction < 0).any() or (prediction > 1).any()):
                raise ValueError("Member probabilities must be in [0, 1]")
            if self.task == "regression" and (prediction <= 0).any():
                raise ValueError("Member prices must be positive")
            predictions.append(prediction)
        result = np.asarray(predictions).mean(axis=0)
        if not np.isfinite(result).all():
            raise ValueError("Average prediction overflowed")
        return result


class AverageClassifier(ClassifierMixin, FrozenAverage):
    """Soft voting: сначала средняя вероятность, затем порог."""
    task = "classification"

    def predict_proba(self, X):
        probability = self._mean(X)
        return np.column_stack([1 - probability, probability])

    def predict(self, X):
        return (self._mean(X) >= .5).astype(int)


class AverageRegressor(RegressorMixin, FrozenAverage):
    """Среднее исходных цен, не среднее логарифмов."""
    task = "regression"

    def predict(self, X):
        return self._mean(X)


def make_average_pipeline(models, task):
    """Упаковать одинаковый fold разных моделей в один предсказывающий Pipeline."""
    if task not in ("classification", "regression") or len(models) < 2:
        raise ValueError("Average needs a supported task and at least two models")
    for model in models:
        check_is_fitted(model)
        if list(model.feature_names_in_) != list(models[0].feature_names_in_):
            raise ValueError("Member input schemas must match")
        if task == "classification" and not np.array_equal(model.classes_, [0, 1]):
            raise ValueError("Expected member classes [0, 1]")
    estimator = AverageClassifier(models) if task == "classification" else AverageRegressor(models)
    estimator.feature_names_in_ = np.asarray(models[0].feature_names_in_, dtype=object)
    estimator.n_features_in_ = len(estimator.feature_names_in_)
    if task == "classification":
        estimator.classes_ = np.array([0, 1])
    return Pipeline([("model", estimator)])


def save_equal_average(run_dirs, run_name, *, trusted=False):
    """Проверить совместимость OOF, усреднить и сохранить self-contained run.

    Веса фиксированы 1/N, поиск весов и обучение не выполняются. Для каждого
    outer-fold сохраняются копии полных Pipeline членов именно этого фолда.
    Старые файлы не меняются; новый run и отдельный журнал ансамблей создаются.
    """
    if trusted is not True:
        raise ValueError("Only use trusted own saved models; pass trusted=True")
    validate_run_name(run_name)
    paths = [Path(path).resolve() for path in run_dirs]
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("Choose at least two distinct runs")
    manifest_hashes = [hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() for path in paths]
    metadata = [read_run_metadata(path) for path in paths]
    reference = metadata[0][0]
    task, metric = reference["task"], reference["metric"]["name"]
    id_column, target = reference["data"]["id_column"], reference["data"]["target"]
    count = reference["validation"]["n_splits"]
    threshold = reference.get("training", {}).get("prediction_threshold", .5)
    identity, test_identity, oof_values, test_values, comparisons = None, None, [], [], []
    for path, (config, summary, details, manifest) in zip(paths, metadata):
        if (config["task"] != task or config["metric"] != reference["metric"]
                or config["data"]["id_column"] != id_column or config["data"]["target"] != target
                or config["data"].get("parsing_mode", "default") != reference["data"].get("parsing_mode", "default")
                or config["validation"]["n_splits"] != count
                or manifest["dataset_type"] != metadata[0][3]["dataset_type"]
                or config.get("training", {}).get("prediction_threshold", .5) != threshold):
            raise ValueError("Member tasks/metrics/schema/folds/threshold must match")
        hashes = details.get("input_files_sha256")
        if not hashes or hashes != metadata[0][2].get("input_files_sha256"):
            raise ValueError("Members must use the same source data and fixed folds")
        oof = pd.read_csv(path / "oof_predictions.csv")
        test = pd.read_csv(path / "test_predictions_by_fold.csv")
        current_identity = oof[[id_column, "fold", target]]
        if (not oof[id_column].is_unique or oof[id_column].isna().any()
                or not test[id_column].is_unique or test[id_column].isna().any()):
            raise ValueError("OOF/test IDs must be unique and non-missing")
        if identity is None:
            identity, test_identity = current_identity, test[[id_column]]
        if not identity.equals(current_identity) or not test_identity.equals(test[[id_column]]):
            raise ValueError("Member IDs/order/targets/folds must match exactly")
        if test.columns.tolist() != [id_column] + [f"fold_{n}" for n in range(count)]:
            raise ValueError("Unexpected per-fold test columns")
        value = oof["probability" if task == "classification" else "prediction"].to_numpy()
        per_fold = test.drop(columns=id_column).to_numpy()
        # Проверяем каждую модель до усреднения, а не только общий результат.
        report = summarize_cv_results(identity[target], identity["fold"], value, per_fold,
                                      np.zeros(count), metric, threshold)
        for key in ("cv_mean", "oof_score"):
            if not np.isclose(report["summary"][key], summary[key], rtol=1e-10, atol=1e-12):
                raise ValueError("Stored scores disagree with saved predictions")
        for key in ("n_train", "n_test", "n_splits"):
            if report["summary"][key] != summary[key]:
                raise ValueError("Stored sizes disagree with saved predictions")
        comparisons.append({"run_name": path.name, "model": config["model"]["name"],
                            "cv_mean": summary["cv_mean"], "oof_score": summary["oof_score"]})
        oof_values.append(value)
        test_values.append(per_fold)

    averaged_oof, averaged_test = np.mean(oof_values, axis=0), np.mean(test_values, axis=0)
    report = summarize_cv_results(identity[target], identity["fold"], averaged_oof, averaged_test,
                                  np.zeros(count), metric, threshold)
    comparisons.append({"run_name": run_name, "model": "equal_average",
                        "cv_mean": report["summary"]["cv_mean"], "oof_score": report["summary"]["oof_score"]})
    config = deepcopy(reference)
    config["model"] = {"name": "equal_average", "params": {"weights": [1 / len(paths)] * len(paths)},
                       "target_transform": "none"}
    config["run"] = {"mode": "train", "run_name": run_name, "operation": "average_saved_models_no_fit"}
    base = Path(reference["output"]["artifacts_dir"])
    config["output"]["artifacts_dir"] = str(base / "ensembles")
    config["output"]["experiments_path"] = str(Path(reference["output"]["experiments_path"]).with_name(f"{task}_ensembles.csv"))
    comparison_path = Path(config["output"]["experiments_path"]).with_name(f"{task}_comparison_{run_name}.csv")
    if (base / "ensembles" / run_name).exists() or comparison_path.exists():
        raise FileExistsError("Average run/comparison already exists; choose a new name")

    loaded = [load_fitted_run(path, trusted=True)[2] for path in paths]
    for path, expected in zip(paths, manifest_hashes):
        if hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() != expected:
            raise ValueError("Source run changed while assembling the average")
    frozen_models = [make_average_pipeline([models[n] for models in loaded], task) for n in range(count)]
    versions = dict(metadata[0][2]["versions"])
    for _, _, details, _ in metadata:
        for package in ("torch", "catboost", "lightgbm", "xgboost"):
            if package in details["versions"]:
                versions[package] = details["versions"][package]
    details = {"model": "equal_average", "model_params": config["model"]["params"],
               "feature_set": config["features"], "preprocessing": "each_member_uses_its_saved_pipeline",
               "target_transform": "none", "run_mode": "train", "versions": versions,
               "operation": "average_saved_models_no_fit", "input_files_sha256": metadata[0][2]["input_files_sha256"],
               "members": [{"path": str(path), "model": item[0]["model"]["name"],
                            "manifest_sha256": manifest_hash}
                           for path, item, manifest_hash in zip(paths, metadata, manifest_hashes)],
               "comparison": comparisons, "classification_method": "soft_voting" if task == "classification" else None,
               "regression_scale": "arithmetic_mean_of_prices" if task == "regression" else None}
    saved = save_cv_results(run_name, config, identity[id_column], test_identity[id_column],
                            identity[target], identity["fold"], averaged_oof, averaged_test,
                            np.zeros(count), details, dataset_type=metadata[0][3]["dataset_type"],
                            fitted_models=frozen_models)
    pd.DataFrame(comparisons).to_csv(comparison_path, index=False, mode="x")
    return saved, pd.DataFrame(comparisons)
