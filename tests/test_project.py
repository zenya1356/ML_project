"""Небольшой набор проверок главного пути проекта на искусственных данных."""

from copy import deepcopy
import hashlib
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
import yaml

import main as app
from src.data import load_data, read_housing_csv, split_features_target, validate_data
from src.ensemble import make_average_pipeline
from src.features import RecipeFeatures, SimpleFeatures
from src.inference import load_fitted_run, make_submission, read_run_metadata
from src.models import build_model_pipeline
from src.train import train_cv
from src.utils import save_cv_results
from src.validation import (
    calculate_metric, get_or_create_folds, initialize_cv_predictions,
    record_fold_predictions, summarize_cv_results,
)


class ProjectTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ml-project-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.X = pd.DataFrame({"value": [1., np.nan, 3., 5., 8., 13., 21., 34., 55., 89.],
                               "group": ["a", "b"] * 5})
        self.y = pd.Series([0, 1] * 5)
        self.X_test = pd.DataFrame({"value": [np.nan, 100.], "group": ["new", "a"]})
        self.folds = np.repeat(np.arange(5), 2)
        self.ids = pd.Series(np.arange(10), name="Id")
        limit = threadpool_limits(limits=2)
        self.addCleanup(limit.restore_original_limits)

    def config(self, task="classification"):
        classification = task == "classification"
        return {
            "task": task,
            "data": {"id_column": "Id", "target": "Target", "parsing_mode": "default",
                     "train_path": str(self.root / "train.csv"),
                     "test_path": str(self.root / "test.csv"),
                     "submission_template_path": str(self.root / "template.csv")},
            "features": {"numeric": ["value"], "categorical": ["group"], "drop": []},
            "model": {"name": "logistic_regression" if classification else "ridge",
                      "params": {"max_iter": 1000} if classification else {"alpha": 1.},
                      "target_transform": "none" if classification else "log"},
            "metric": {"name": "accuracy" if classification else "rmse_log",
                       "direction": "maximize" if classification else "minimize"},
            "validation": {"strategy": "stratified_kfold" if classification else "kfold",
                           "n_splits": 5, "shuffle": True, "random_state": 42,
                           "folds_path": str(self.root / "folds.csv")},
            "output": {"artifacts_dir": str(self.root / task),
                       "experiments_path": str(self.root / f"{task}.csv"),
                       "submission_path": str(self.root / f"{task}_submission.csv")},
            "run": {"mode": "train", "run_name": "toy"},
        }

    def saved_run(self, task="classification", run_name="toy"):
        config = self.config(task)
        config["run"]["run_name"] = run_name
        y = self.y if task == "classification" else pd.Series(np.arange(1, 11) * 100.)
        test = self.X_test.copy()
        test.insert(0, "Id", [100, 101])
        test.to_csv(config["data"]["test_path"], index=False)
        pd.DataFrame({"Id": [100, 101], "Target": [0, 0]}).to_csv(
            config["data"]["submission_template_path"], index=False)
        result = train_cv(config, self.X, y, self.X_test, self.folds)
        run = save_cv_results(
            run_name, config, self.ids, [100, 101], y, self.folds,
            result["oof_predictions"], result["test_predictions_by_fold"],
            result["fold_training_seconds"], app.experiment_details(config, {
                config["data"]["test_path"]: hashlib.sha256(Path(config["data"]["test_path"]).read_bytes()).hexdigest(),
            }),
            dataset_type="synthetic", fitted_models=result["fitted_models"],
        )
        return run, result

    def test_01_load_and_split_data(self):
        train = self.X.copy()
        train.insert(0, "Id", self.ids)
        train["Target"] = self.y
        test = train.drop(columns="Target")
        train.to_csv(self.root / "train.csv", index=False)
        test.to_csv(self.root / "test.csv", index=False)
        loaded, loaded_test = load_data(self.root / "train.csv", self.root / "test.csv")
        validate_data(loaded, loaded_test, "Target", "Id")
        X, y = split_features_target(loaded, "Target")
        self.assertNotIn("Target", X)
        self.assertIn("Target", loaded)
        np.testing.assert_array_equal(y, self.y)

    def test_02_housing_absence_is_not_unknown(self):
        path = self.root / "housing.csv"
        pd.DataFrame({"Alley": ["NA", ""], "MasVnrType": ["None", "NA"]}).to_csv(path, index=False)
        data = read_housing_csv(path)
        self.assertEqual(data.loc[0, "Alley"], "NA")
        self.assertEqual(data.loc[0, "MasVnrType"], "None")
        self.assertTrue(pd.isna(data.loc[1, "Alley"]))
        self.assertTrue(pd.isna(data.loc[1, "MasVnrType"]))

    def test_03_saved_folds_follow_ids_and_reject_changed_target(self):
        config = self.config()["validation"]
        path = config["folds_path"]
        folds, created = get_or_create_folds(self.X, self.y, self.ids, config, path)
        self.assertTrue(created)
        order = np.arange(9, -1, -1)
        reordered, created = get_or_create_folds(self.X.iloc[order], self.y.iloc[order],
                                                self.ids.iloc[order], config, path)
        self.assertFalse(created)
        np.testing.assert_array_equal(reordered, folds[order])
        changed_y = self.y.copy()
        changed_y.iloc[0] = 1
        with self.assertRaises(ValueError):
            get_or_create_folds(self.X, changed_y, self.ids, config, path)

    def test_04_metrics_use_labels_and_natural_log(self):
        self.assertEqual(calculate_metric([0, 1], [0, 1], "accuracy"), 1.)
        expected = abs(np.log(100.) - np.log(110.))
        self.assertAlmostEqual(calculate_metric([100.], [110.], "rmse_log"), expected)
        with self.assertRaises(ValueError):
            calculate_metric([100.], [0.], "rmse_log")
        with self.assertRaises(ValueError):
            calculate_metric([0, 1], [.2, .8], "accuracy")

    def test_05_preprocessing_is_train_only_and_handles_new_categories(self):
        model = build_model_pipeline(self.config(), self.X)
        model.fit(self.X.iloc[:8], self.y.iloc[:8])
        preprocessor = model.named_steps["preprocessing"]
        median = preprocessor.named_transformers_["numbers"].named_steps["fill_missing"].statistics_[0]
        self.assertEqual(median, self.X.iloc[:8]["value"].median())
        self.assertTrue(np.isfinite(model.predict_proba(self.X_test)).all())
        bad = self.config()
        bad["features"]["numeric"] = ["Id"]
        with self.assertRaises(ValueError):
            build_model_pipeline(bad, self.X.assign(Id=self.ids))

    def test_06_tree_and_external_models_for_both_tasks(self):
        for task in ("classification", "regression"):
            original = yaml.safe_load(Path(f"configs/{task}.yaml").read_text(encoding="utf-8"))
            y = self.y if task == "classification" else pd.Series(np.arange(1, 11) * 100.)
            for name in ("random_forest", "gradient_boosting", "hist_gradient_boosting",
                         "catboost", "lightgbm", "xgboost"):
                with self.subTest(task=task, model=name):
                    config = self.config(task)
                    params = deepcopy(original["baseline_models"][name])
                    rounds = "iterations" if name == "catboost" else "max_iter" if name == "hist_gradient_boosting" else "n_estimators"
                    params[rounds] = 3
                    config["model"].update(name=name, params=params)
                    model = build_model_pipeline(config, self.X).fit(self.X, y)
                    prediction = model.predict(self.X_test)
                    self.assertTrue(np.isfinite(prediction).all())
                    if task == "classification":
                        np.testing.assert_allclose(model.predict_proba(self.X_test).sum(axis=1), 1.)
                    else:
                        self.assertTrue((prediction > 0).all())

    def test_07_oof_matches_held_out_predictions(self):
        result = train_cv(self.config(), self.X, self.y, self.X_test, self.folds)
        self.assertEqual(result["test_predictions_by_fold"].shape, (2, 5))
        self.assertEqual(len({id(model) for model in result["fitted_models"]}), 5)
        for fold, model in enumerate(result["fitted_models"]):
            valid = self.folds == fold
            np.testing.assert_allclose(result["oof_predictions"][valid], model.predict_proba(self.X.loc[valid])[:, 1])
            median = model.named_steps["preprocessing"].named_transformers_["numbers"].named_steps["fill_missing"].statistics_[0]
            self.assertEqual(median, self.X.loc[~valid, "value"].median())

    def test_08_regression_predict_applies_exp_once(self):
        model = build_model_pipeline(self.config("regression"), self.X)
        model.fit(self.X, np.arange(1, 11) * 100.)
        transformed = model.named_steps["preprocessing"].transform(self.X_test)
        log_predictions = model.named_steps["model"].regressor_.predict(transformed)
        np.testing.assert_allclose(model.predict(self.X_test), np.exp(log_predictions))

    def test_09_incomplete_and_duplicate_oof_are_rejected(self):
        oof, test = initialize_cv_predictions(10, 2, 5)
        with self.assertRaises(ValueError):
            summarize_cv_results(self.y, self.folds, oof, test, np.zeros(5), "accuracy")
        record_fold_predictions(oof, test, self.folds, 0, [.2, .8], [.3, .7])
        with self.assertRaises(ValueError):
            record_fold_predictions(oof, test, self.folds, 0, [.2, .8], [.3, .7])

    def test_10_save_reload_and_submit_without_fit(self):
        for task in ("classification", "regression"):
            with self.subTest(task=task):
                run, result = self.saved_run(task)
                with patch.object(Pipeline, "fit", side_effect=AssertionError("No inference fit")):
                    output = make_submission(run, trusted=True)
                frame = pd.read_csv(output)
                self.assertEqual(frame.columns.tolist(), ["Id", "Target"])
                self.assertEqual(frame.Id.tolist(), [100, 101])
                expected = result["test_predictions_by_fold"].mean(axis=1)
                if task == "classification":
                    expected = (expected >= .5).astype(int)
                np.testing.assert_allclose(frame.Target, expected)
                with patch.object(Pipeline, "fit", side_effect=AssertionError("No inference fit")):
                    single = make_submission(run, trusted=True, output_path=self.root / f"{task}_fold0.csv", fold_index=0)
                expected_single = result["test_predictions_by_fold"][:, 0]
                if task == "classification":
                    expected_single = (expected_single >= .5).astype(int)
                np.testing.assert_allclose(pd.read_csv(single).Target, expected_single)

    def test_11_untrusted_or_damaged_run_is_rejected(self):
        run, _ = self.saved_run()
        with self.assertRaises(ValueError):
            load_fitted_run(run)
        with (run / "config.yaml").open("a", encoding="utf-8") as file:
            file.write("\n")
        with self.assertRaises(ValueError):
            read_run_metadata(run)

    def test_12_cli_selects_model_without_changing_yaml(self):
        config = self.config()
        config["baseline_models"] = {"catboost": {"iterations": 3, "verbose": False}}
        path = self.root / "config.yaml"
        original = yaml.safe_dump(config)
        path.write_text(original, encoding="utf-8")
        with patch("sys.argv", ["main.py", "--config", str(path), "--model", "catboost"]):
            args = app.parse_args()
        selected = app.load_config(args)
        self.assertEqual(selected["model"]["name"], "catboost")
        self.assertEqual(selected["model"]["params"]["iterations"], 3)
        self.assertEqual(path.read_text(encoding="utf-8"), original)
        ensemble_args = ["main.py", "--mode", "ensemble", "--run-dirs", "first", "second",
                         "--run-name", "average", "--trust-models"]
        with patch("sys.argv", ensemble_args):
            self.assertEqual(app.parse_args().run_dirs, ["first", "second"])
        invalid = [ensemble_args[:-1], ensemble_args[:5] + ensemble_args[6:],
                   ensemble_args + ["--config", str(path)], ensemble_args + ["--fold-index", "0"],
                   ["main.py", "--config", str(path), "--run-dirs", "first", "second"]]
        for command in invalid:
            with self.subTest(command=command), patch("sys.argv", command), patch("sys.stderr", StringIO()):
                with self.assertRaises(SystemExit):
                    app.parse_args()

    def test_13_simple_features_do_not_change_input(self):
        X = pd.DataFrame({"SibSp": [0, 1], "Parch": [0, 2]})
        result = SimpleFeatures("classification").fit_transform(X)
        self.assertEqual(result.FamilySize.tolist(), [1, 4])
        self.assertNotIn("FamilySize", X)

    def test_14_mlp_and_stacking_keep_working(self):
        X = pd.concat([self.X, self.X], ignore_index=True)
        folds = np.repeat(np.arange(2), 10)
        for task in ("classification", "regression"):
            y = pd.Series([0, 1] * 10 if task == "classification" else np.arange(1, 21) * 100.)
            for name in ("torch_mlp", "stacking"):
                with self.subTest(task=task, model=name):
                    config = self.config(task)
                    config["validation"]["n_splits"] = 2
                    config["baseline_models"] = {"hist_gradient_boosting": {
                        "max_iter": 3, "min_samples_leaf": 1, "early_stopping": False, "random_state": 42}}
                    params = {"hidden_sizes": [4], "max_epochs": 3, "patience": 2, "batch_size": 4,
                              "random_state": 42} if name == "torch_mlp" else {"inner_splits": 2}
                    config["model"].update(name=name, params=params)
                    result = train_cv(config, X, y, self.X_test, folds)
                    self.assertTrue(np.isfinite(result["oof_predictions"]).all())

    def test_15_average_uses_probabilities_and_original_prices(self):
        for task in ("classification", "regression"):
            config = self.config(task)
            y = self.y if task == "classification" else pd.Series(np.arange(1, 11) * 100.)
            models = [build_model_pipeline(config, self.X).fit(self.X, y) for _ in range(2)]
            average = make_average_pipeline(models, task)
            if task == "classification":
                expected = np.mean([model.predict_proba(self.X_test) for model in models], axis=0)
                np.testing.assert_allclose(average.predict_proba(self.X_test), expected)
            else:
                expected = np.mean([model.predict(self.X_test) for model in models], axis=0)
                np.testing.assert_allclose(average.predict(self.X_test), expected)
            first, result = self.saved_run(task)
            second, _ = self.saved_run(task, "toy_second")
            command = ["main.py", "--mode", "ensemble", "--run-dirs", str(first), str(second),
                       "--run-name", "cli_average", "--trust-models"]
            with patch("sys.argv", command), patch("sys.stdout", StringIO()), \
                    patch.object(Pipeline, "fit", side_effect=AssertionError("Ensemble must not fit")), \
                    patch.object(app, "load_data", side_effect=AssertionError("Ensemble reads saved results")):
                app.main()
                output = make_submission(self.root / task / "ensembles" / "cli_average", trusted=True,
                                         output_path=self.root / f"{task}_average.csv")
            expected = result["test_predictions_by_fold"].mean(axis=1)
            if task == "classification":
                expected = (expected >= .5).astype(int)
            np.testing.assert_allclose(pd.read_csv(output).Target, expected)

    def test_16_checklist_models_encoding_and_features(self):
        for task in ("classification", "regression"):
            y = self.y if task == "classification" else np.arange(1, 11) * 100.
            recipes = [("knn", {"n_neighbors": 3}), ("decision_tree", {"max_depth": 3})]
            if task == "regression":
                recipes += [("lasso", {"alpha": .01}), ("elastic_net", {"alpha": .01})]
            for name, params in recipes:
                with self.subTest(task=task, model=name):
                    config = self.config(task)
                    config["model"].update(name=name, params=params)
                    config["preprocessing"] = {"categorical_encoding": "ordinal"}
                    model = build_model_pipeline(config, self.X).fit(self.X, y)
                    self.assertTrue(np.isfinite(model.predict(self.X_test)).all())
                    categories = model.named_steps["preprocessing"].named_transformers_["categories"]
                    self.assertEqual(categories.transform(self.X_test[["group"]])[0, 0], -1)
        X = pd.DataFrame({"Name": ["A, Mr. B", None], "SibSp": [0, -1], "Parch": [0, 0],
                          "Fare": [10., -2.], "YrSold": [2010, 2000], "YearBuilt": [2000, 2010],
                          "MoSold": [1, 13], "OverallQual": [5, 6], "GrLivArea": [100., -2.]})
        for recipe in RecipeFeatures.RECIPES:
            with self.subTest(recipe=recipe):
                transformer = RecipeFeatures(recipe)
                result = transformer.fit_transform(X)
                self.assertEqual(len(result), len(X))
                self.assertEqual(set(result) - set(X), set(transformer.numeric_outputs + transformer.categorical_outputs))
                if recipe == "fare_per_person":
                    self.assertTrue(pd.isna(result.loc[1, "FarePerPerson"]))
                if recipe == "house_age":
                    self.assertTrue(pd.isna(result.loc[1, "HouseAge"]))
        with patch("sys.argv", ["main.py", "--config", "configs/classification.yaml", "--mode", "train",
                                "--run-name", "one_command", "--make-submission"]):
            self.assertTrue(app.parse_args().make_submission)

    def test_17_neural_options_and_embedding_roundtrip(self):
        import joblib
        import torch
        from src.neural import TorchClassifier, TorchRegressor, fit_torch_fold
        X = np.column_stack([np.arange(9) / 9., np.arange(9) % 2])
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
        variants = [{"batch_norm": True}, {"activation": "gelu"}, {"optimizer": "adam"},
                    {"scheduler": "cosine"}, {"loss": "weighted_bce"},
                    {"embedding": True, "numeric_features": 1}]
        for number, params in enumerate(variants):
            with self.subTest(params=params):
                model = TorchClassifier(hidden_sizes=[4], dropout=0., max_epochs=2,
                                        batch_size=4, **params)
                model.fit(X, y, validation_data=(X[:3], y[:3]))
                prediction_X = X[:3].copy()
                if params.get("embedding"):
                    prediction_X[0, 1] = -1  # UNK получен из train-only OrdinalEncoder.
                expected = model.predict_proba(prediction_X)
                path = self.root / f"neural_{number}.joblib"
                joblib.dump(model, path)
                np.testing.assert_allclose(joblib.load(path).predict_proba(prediction_X), expected)
                if params.get("scheduler"):
                    self.assertIsNotNone(model.last_checkpoint_["scheduler_state_dict"])
                if params.get("loss"):
                    self.assertAlmostEqual(model.positive_weight_, 5 / 4)
                self.assertTrue(torch.isfinite(torch.tensor(expected)).all())
        regressor = TorchRegressor(hidden_sizes=[4], max_epochs=2, loss="huber", batch_size=4)
        regressor.fit(X, np.arange(1, 10) * 100., validation_data=(X[:3], np.arange(1, 4) * 100.))
        self.assertTrue((regressor.predict(X[:3]) > 0).all())
        for task in ("classification", "regression"):
            config = self.config(task)
            config["model"].update(name="torch_mlp", params={"hidden_sizes": [4], "embedding": True,
                                                            "max_epochs": 2, "batch_size": 4})
            model = build_model_pipeline(config, self.X)
            y = self.y if task == "classification" else pd.Series(np.arange(1, 11) * 100.)
            fit_torch_fold(model, self.X, y, 42)
            path = self.root / f"embedding_pipeline_{task}.joblib"
            joblib.dump(model, path)
            np.testing.assert_allclose(joblib.load(path).predict(self.X_test), model.predict(self.X_test))


if __name__ == "__main__":
    unittest.main()
