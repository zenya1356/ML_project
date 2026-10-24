from pathlib import Path

import pandas as pd


# По data/regression/data_description.txt: NA здесь обозначает отсутствие объекта.
HOUSING_ABSENCE_NA_COLUMNS = frozenset({
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature",
})


def read_housing_csv(path: str | Path) -> pd.DataFrame:
    """Читает House Prices как в EDA

    В 14 полях NA сохраняется строковой категорией отсутствия объекта.
    В остальных полях NA — неизвестное значение.
    
    keep_default_na=False сохраняет буквальную строку None в MasVnrType.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No CSV file: {path}")
    # Нужен только заголовок: не читаем весь CSV сначала как текстовые данные.
    columns = pd.read_csv(path, nrows=0, keep_default_na=False).columns
    missing_tokens = {
        column: [""] if column in HOUSING_ABSENCE_NA_COLUMNS else ["", "NA"]
        for column in columns
    }
    return pd.read_csv(path, keep_default_na=False, na_values=missing_tokens)


def load_data(
    train_path: str,
    test_path: str,
    *,
    parsing_mode: str = "default",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает два CSV; режим выбирается явно"""
    if parsing_mode not in ("default", "housing"):
        raise ValueError(f"Unknown CSV parsing mode: {parsing_mode}")
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.is_file():
        raise FileNotFoundError(f"No train file: {train_path}")

    if not test_path.is_file():
        raise FileNotFoundError(f"No test file: {test_path}")

    reader = read_housing_csv if parsing_mode == "housing" else pd.read_csv
    train_df = reader(train_path)
    test_df = reader(test_path)

    return train_df, test_df


def split_features_target(
    train_df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Отделить target в Series; исходный DataFrame не изменяется."""
    if target_column not in train_df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    return X, y

def validate_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> None:

    """Проверить наличие ID/target и одинаковый порядок признаков train/test."""

    if target_column not in train_df.columns:
        raise ValueError(f"Target column missing from train: {target_column}")

    if target_column in test_df.columns:
        raise ValueError(f"Target column unexpectedly in test: {target_column}")

    if id_column not in train_df.columns:
        raise ValueError(f"Id column missing from train: {id_column}")

    if id_column not in test_df.columns:
        raise ValueError(f"ID column missing from test: {id_column}")

    train_features = train_df.drop(columns=[target_column]).columns.tolist()
    test_features = test_df.columns.tolist()

    if train_features != test_features:
        raise ValueError("Train and test features or their orders differ")

     

