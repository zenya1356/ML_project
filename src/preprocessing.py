"""Сборка обработки признаков без обучения на данных."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def get_feature_columns(X, feature_config, forbidden_columns=()):
    """Проверяет явное распределение всех входных столбцов по трём группам.

    численные и категориальные используются моделью.
    Возвращает копии двух списков используемых признаков, не данные.
    """
    if not X.columns.is_unique:
        raise ValueError("Input feature names must be unique")

    groups = {}
    for name in ("numeric", "categorical", "drop"):
        columns = feature_config.get(name, [])
        if not isinstance(columns, list) or any(not isinstance(column, str) for column in columns):
            raise ValueError(f"features.{name} must be a list of column names")
        groups[name] = columns.copy()

    selected = groups["numeric"] + groups["categorical"]
    declared = selected + groups["drop"]
    if not selected:
        raise ValueError("At least one model feature is required")
    if len(declared) != len(set(declared)):
        raise ValueError("Feature groups must not contain repeated or overlapping columns")
    forbidden = set(selected) & set(forbidden_columns)
    if forbidden:
        raise ValueError(f"ID/target cannot be model features: {sorted(forbidden)}")

    unknown = set(declared) - set(X.columns)
    undeclared = set(X.columns) - set(declared)
    if unknown:
        raise ValueError(f"Configured feature columns are missing from X: {sorted(unknown)}")
    if undeclared:
        raise ValueError(f"Input columns must be selected or explicitly dropped: {sorted(undeclared)}")

    return groups["numeric"], groups["categorical"]


def build_preprocessor(numeric_columns, categorical_columns, *, numeric_missing_indicators=False,
                       numeric_imputation="median", categorical_encoding="onehot"):
    """Вернуть новый, ещё не обученный обработчик для указанных столбцов.

    аргументы - списки названий столбцов. Остальные столбцы исключаются.
    Внутри каждого фолда создается свой обработчик и вызывается fit только
    на обучающей части
    При numeric_missing_indicators=True импутер добавляет признаки пропусков
    только для числовых столбцов с пропусками на train-части. Scaler обрабатывает
    и заполненные числа, и индикаторы; категориальная ветка не меняется.
    numeric_imputation выбирает median или mean, статистика считается на fit.
    """
    if type(numeric_missing_indicators) is not bool:
        raise ValueError("numeric_missing_indicators must be a boolean")
    if numeric_imputation not in ("median", "mean"):
        raise ValueError("numeric_imputation must be median or mean")
    if categorical_encoding not in ("onehot", "ordinal"):
        raise ValueError("categorical_encoding must be onehot or ordinal")
    encoder = (OneHotEncoder(handle_unknown="ignore", sparse_output=False)
               if categorical_encoding == "onehot" else
               OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    numeric_pipeline = Pipeline([
        ("fill_missing", SimpleImputer(
            strategy=numeric_imputation,
            keep_empty_features=True,
            add_indicator=numeric_missing_indicators,
        )),
        ("scale", StandardScaler()),
    ])

    category_pipeline = Pipeline([
        ("fill_missing", SimpleImputer(
            strategy="most_frequent",
            keep_empty_features=True,
        )),
        ("encode", encoder),
    ])

    return ColumnTransformer([
        ("numbers", numeric_pipeline, numeric_columns),
        ("categories", category_pipeline, categorical_columns),
    ], remainder="drop")
