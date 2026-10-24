"""Два простых построчных признака, без обучения статистик на данных."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class SimpleFeatures(TransformerMixin, BaseEstimator):
    """FamilySize или TotalSF; исходный DataFrame не изменяется."""

    def __init__(self, task):
        self.task = task

    @property
    def output_column(self):
        if self.task == "classification":
            return "FamilySize"
        if self.task == "regression":
            return "TotalSF"
        raise ValueError(f"Unknown feature task: {self.task}")

    def _validate_input(self, X):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("SimpleFeatures requires a DataFrame with unique columns")
        required = (["SibSp", "Parch"] if self.task == "classification"
                    else ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"])
        if self.output_column in X.columns:
            raise ValueError(f"Generated column already exists: {self.output_column}")
        if not set(required).issubset(X.columns):
            raise ValueError(f"SimpleFeatures requires columns: {required}")

    def fit(self, X, y=None):
        self._validate_input(X)
        # Сохраняем только схему входа, не средние/медианы и не target.
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        self._validate_input(X)
        if X.columns.tolist() != self.feature_names_in_.tolist():
            raise ValueError("Feature columns/order differ from fitted input")
        result = X.copy()
        if self.task == "classification":
            result["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        else:
            result["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
        # Арифметика сохраняет NaN: downstream imputer обучается внутри фолда.
        return result


class RecipeFeatures(TransformerMixin, BaseEstimator):
    """Один фиксированный FE-рецепт; статистики/target не используются."""

    RECIPES = {
        "title": (["Name"], [], ["Title"]),
        "is_alone": (["SibSp", "Parch"], ["IsAlone"], []),
        "fare_per_person": (["SibSp", "Parch", "Fare"], ["FarePerPerson"], []),
        "log_fare": (["Fare"], ["LogFare"], []),
        "house_age": (["YrSold", "YearBuilt"], ["HouseAge"], []),
        "time": (["YearBuilt", "MoSold"], ["BuiltDecade", "MonthSin", "MonthCos"], []),
        "quality_area": (["OverallQual", "GrLivArea"], ["QualityArea"], []),
        "log_area": (["GrLivArea"], ["LogGrLivArea"], []),
    }

    def __init__(self, recipe):
        self.recipe = recipe

    @property
    def numeric_outputs(self):
        return list(self.RECIPES[self.recipe][1])

    @property
    def categorical_outputs(self):
        return list(self.RECIPES[self.recipe][2])

    def _validate_input(self, X):
        if self.recipe not in self.RECIPES:
            raise ValueError(f"Unknown feature recipe: {self.recipe}")
        required = self.RECIPES[self.recipe][0]
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique or not set(required) <= set(X.columns):
            raise ValueError(f"Recipe requires unique DataFrame columns including {required}")
        if set(self.numeric_outputs + self.categorical_outputs) & set(X.columns):
            raise ValueError("Generated feature already exists")

    def fit(self, X, y=None):
        self._validate_input(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        self._validate_input(X)
        if X.columns.tolist() != self.feature_names_in_.tolist():
            raise ValueError("Feature columns/order differ from fitted input")
        result = X.copy()
        if self.recipe == "title":
            result["Title"] = X["Name"].str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
        elif self.recipe in ("is_alone", "fare_per_person"):
            size = X["SibSp"] + X["Parch"] + 1
            if self.recipe == "is_alone":
                result["IsAlone"] = (size == 1).astype(float).where(size.notna())
            else:
                result["FarePerPerson"] = X["Fare"] / size.where(size > 0)
        elif self.recipe in ("log_fare", "log_area"):
            source, output = (("Fare", "LogFare") if self.recipe == "log_fare"
                              else ("GrLivArea", "LogGrLivArea"))
            result[output] = np.log1p(X[source].where(X[source] >= 0))
        elif self.recipe == "house_age":
            age = X["YrSold"] - X["YearBuilt"]
            result["HouseAge"] = age.where(age >= 0)
        elif self.recipe == "time":
            result["BuiltDecade"] = (X["YearBuilt"] // 10) * 10
            month = X["MoSold"].where(X["MoSold"].between(1, 12))
            result["MonthSin"] = np.sin(2 * np.pi * month / 12)
            result["MonthCos"] = np.cos(2 * np.pi * month / 12)
        elif self.recipe == "quality_area":
            result["QualityArea"] = X["OverallQual"] * X["GrLivArea"]
        return result
