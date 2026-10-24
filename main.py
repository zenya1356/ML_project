import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform

import yaml
from src.data import load_data, split_features_target, validate_data
from src.ensemble import save_equal_average
from src.models import build_model_pipeline
from src.inference import make_submission
from src.train import select_debug_positions, train_cv
from src.utils import save_cv_results, validate_run_name
from src.validation import get_or_create_folds, summarize_folds

def parse_args():
    """Прочитать команду и проверить аргументы выбранного режима."""
    parser = argparse.ArgumentParser(
        description="Launch of the whole project"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to yaml config file; required for check/debug/train",
    )
    parser.add_argument("--mode", choices=("check", "debug", "train", "inference", "ensemble"), default="check",
                        help="check validates; debug/train fit; inference predicts; ensemble averages saved runs")
    parser.add_argument("--run-name", help="Unique name; required for debug/train/ensemble")
    parser.add_argument("--run-dirs", nargs="+", help="At least two completed runs for ensemble mode")
    parser.add_argument("--make-submission", action="store_true",
                        help="After full training, load the new own run and create a unique submission")
    parser.add_argument("--model", help="Default model or an alternative from baseline_models in YAML")
    parser.add_argument("--debug-rows-per-fold", type=int, default=20)
    parser.add_argument("--debug-test-rows", type=int, default=50)
    parser.add_argument("--numeric-missing-indicators", action="store_true",
                        help="Experiment: add numeric missingness indicators inside each fold")
    parser.add_argument("--numeric-imputation", choices=("median", "mean"),
                        help="Numeric imputation strategy for check/debug/train")
    parser.add_argument("--simple-features", action="store_true",
                        help="Add FamilySize (Titanic) or TotalSF (Housing) inside the pipeline")
    parser.add_argument("--run-dir", help="Completed training run for inference")
    parser.add_argument("--submission-path", help="New output CSV for inference")
    parser.add_argument("--test-path", help="Optional relocated competition test CSV for inference")
    parser.add_argument("--fold-index", type=int, help="Inference with one fixed saved fold (0-based); default averages all")
    parser.add_argument("--trust-models", action="store_true",
                        help="Only for own trusted joblib files; loading can execute code")

    args = parser.parse_args()
    if args.make_submission and args.mode != "train":
        parser.error("--make-submission is only available with --mode train")
    if args.run_dirs is not None and args.mode != "ensemble":
        parser.error("--run-dirs is only available with --mode ensemble")
    if args.mode == "ensemble":
        if not args.run_dirs or len(args.run_dirs) < 2 or not args.run_name or not args.trust_models:
            parser.error("Ensemble requires --run-dirs with at least two runs, --run-name and --trust-models")
        if (args.config is not None or args.model is not None or args.numeric_missing_indicators
                or args.numeric_imputation is not None or args.simple_features or args.run_dir is not None
                or args.submission_path is not None or args.test_path is not None or args.fold_index is not None):
            parser.error("Ensemble uses saved settings; config/model/preprocessing/inference overrides are not allowed")
        validate_run_name(args.run_name)
        return args
    if args.mode == "inference":
        if not args.run_dir or not args.trust_models:
            parser.error("Inference requires --run-dir and --trust-models for trusted own artifacts")
        if (args.config is not None or args.run_name is not None or args.model is not None
                or args.numeric_missing_indicators or args.numeric_imputation is not None
                or args.simple_features):
            parser.error("Inference reads the saved config; model/preprocessing overrides are not allowed")
        return args
    if args.config is None:
        parser.error("--config is required for check/debug/train")
    if (args.run_dir is not None or args.submission_path is not None or args.test_path is not None
            or args.trust_models or args.fold_index is not None):
        parser.error("--run-dir/--submission-path/--test-path/--trust-models are only used for inference")
    if args.mode == "check" and args.run_name is not None:
        parser.error("--run-name is only used with --mode debug/train")
    if args.mode != "check":
        if args.run_name is None:
            parser.error("--run-name is required for debug/train")
        validate_run_name(args.run_name)
    if args.debug_rows_per_fold < 2 or args.debug_test_rows < 1:
        parser.error("Debug requires >=2 rows per fold and >=1 test row")
    return args


def load_config(args):
    """Прочитать YAML и применить явно выбранные параметры эксперимента."""
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    # Снимок записываемых настроек соответствует фактически выбранной модели.
    config = deepcopy(config)
    if args.simple_features:
        config.setdefault("feature_engineering", {})["simple_features"] = True
    if args.numeric_missing_indicators:
        config.setdefault("preprocessing", {})["numeric_missing_indicators"] = True
    if args.numeric_imputation is not None:
        config.setdefault("preprocessing", {})["numeric_imputation"] = args.numeric_imputation
    if args.model is not None and args.model != config["model"]["name"]:
        alternatives = config.get("baseline_models", {})
        if args.model not in alternatives:
            raise ValueError("Model is not declared for this task in the YAML config")
        config["model"]["name"] = args.model
        config["model"]["params"] = deepcopy(alternatives[args.model])
    if args.mode == "debug" and config["model"]["name"] == "torch_mlp":
        config["model"]["params"]["max_epochs"] = 10
        config["model"]["params"]["patience"] = 3
    return config


def experiment_details(config, input_hashes):
    """Краткое описание запуска; точные настройки уже находятся в config.yaml."""
    model_name = config["model"]["name"]
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "joblib"):
        versions[package] = version(package)
    if model_name == "torch_mlp":
        versions["torch"] = version("torch")
    elif model_name in ("catboost", "lightgbm", "xgboost"):
        versions[model_name] = version(model_name)
    return {
        "model": model_name, "model_params": config["model"]["params"],
        "feature_set": config["features"],
        "feature_engineering": config.get("feature_engineering", {}),
        "preprocessing": config.get("preprocessing", {}),
        "target_transform": config["model"]["target_transform"],
        "run_mode": config["run"]["mode"], "versions": versions,
        "input_files_sha256": input_hashes,
    }


def main():
    """Проверить данные, обучить CV, применить модели или собрать ансамбль."""
    args = parse_args()
    if args.mode == "ensemble":
        saved, comparison = save_equal_average(args.run_dirs, args.run_name, trusted=args.trust_models)
        print(comparison.to_string(index=False))
        print("Ensemble saved without fit:", saved)
        return
    if args.mode == "inference":
        saved = make_submission(args.run_dir, trusted=args.trust_models,
                                output_path=args.submission_path, test_path=args.test_path,
                                fold_index=args.fold_index)
        print("Inference completed without fit. Submission saved:", saved)
        return

    # 1. Данные: признаки, ответ и идентификаторы храним отдельно.
    config = load_config(args)
    data_config = config["data"]

    train_df, test_df = load_data(data_config["train_path"], data_config["test_path"],
                                 parsing_mode=data_config.get("parsing_mode", "default"))
    validate_data(train_df, test_df, data_config["target"], data_config["id_column"])
    X, y = split_features_target(train_df, data_config["target"])
    id_column = data_config["id_column"]

    train_ids = train_df[id_column].copy()
    test_ids = test_df[id_column].copy()

    X = X.drop(columns=[id_column])
    X_test = test_df.drop(columns=[id_column])

    # Только собираем объект и проверяем выбор столбцов. fit здесь нет.
    build_model_pipeline(config, X)
    print("Task:", config["task"], "Model:", config["model"]["name"], "Mode:", args.mode)
    print("X:", X.shape, "y:", y.shape, "X_test:", X_test.shape)

    # 2. Прежние фолды используются для всех моделей. Test в split не участвует.
    validation_config = config["validation"]
    folds, created = get_or_create_folds(X, y, train_ids, validation_config,
                                        validation_config["folds_path"])
    print("Folds:", "created" if created else "loaded", validation_config["folds_path"])
    print(summarize_folds(folds, y, validation_config["strategy"]).to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    ))
    if args.mode == "check":
        print("Fold checks passed. Baseline pipeline assembled; no preprocessing or models were fitted.")
        return

    # 3. Debug — маленькая выборка и отдельный журнал, не полная оценка качества.
    config["run"] = {
        "mode": args.mode,
        "run_name": args.run_name,
        "source_train_rows": len(X),
        "source_test_rows": len(X_test),
    }
    if args.mode == "debug":
        seed = validation_config.get("random_state")
        if seed is None:
            seed = 42
        positions = select_debug_positions(folds, args.debug_rows_per_fold, seed)
        X, y = X.iloc[positions], y.iloc[positions]
        train_ids, folds = train_ids.iloc[positions], folds[positions]
        X_test, test_ids = X_test.head(args.debug_test_rows), test_ids.head(args.debug_test_rows)
        config["run"].update(rows_per_fold=args.debug_rows_per_fold, random_state=seed)
        # Debug — настоящие данные, но отдельные пути, не общий журнал full-CV.
        config["output"]["artifacts_dir"] = str(Path(config["output"]["artifacts_dir"]) / "debug")
        journal = Path(config["output"]["experiments_path"])
        config["output"]["experiments_path"] = str(journal.with_name(f"{journal.stem}_debug{journal.suffix}"))
    config["run"].update(n_train=len(X), n_test=len(X_test))
    run_dir = Path(config["output"]["artifacts_dir"]) / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"Run already exists; choose a new --run-name: {run_dir}")
    submission_output = Path(config["output"]["submission_path"]).with_name(f"{config['task']}_{args.run_name}.csv")
    if args.make_submission and submission_output.exists():
        raise FileExistsError(f"Submission already exists: {submission_output}")

    # 4. Обучение и сохранение — разные функции.
    print("Training rows:", len(X), "Test rows:", len(X_test))
    source_paths = [Path(data_config["train_path"]), Path(data_config["test_path"]),
                    Path(validation_config["folds_path"]), Path(validation_config["folds_path"]).with_suffix(".json")]
    input_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    threshold = config.get("training", {}).get("prediction_threshold", 0.5)
    result = train_cv(config, X, y, X_test, folds, prediction_threshold=threshold)
    for path in source_paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != input_hashes[str(path)]:
            raise ValueError("Input files changed during training; results will not be saved")
    details = experiment_details(config, input_hashes)
    saved = save_cv_results(
        args.run_name, config, train_ids, test_ids, y, folds,
        result["oof_predictions"], result["test_predictions_by_fold"],
        result["fold_training_seconds"], details,
        prediction_threshold=threshold, dataset_type="real", fitted_models=result["fitted_models"],
    )
    report = result["report"]
    print(report["fold_scores"].to_string(index=False))
    print("CV mean:", f"{report['summary']['cv_mean']:.6f}",
          "std:", f"{report['summary']['cv_std']:.6f}",
          "OOF:", f"{report['summary']['oof_score']:.6f}")
    print("Saved:", saved)
    if args.make_submission:
        submission = make_submission(saved, trusted=True, output_path=submission_output)
        print("Submission saved without additional fit:", submission)

if __name__ == "__main__":
    main()
