from pathlib import Path

import pandas as pd

def load_data(
    train_path: str,
    test_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.exists():
        raise FileNotFoundError(f"No train file: {train_path}")
  
    if not test_path.exists():
        raise FileNotFoundError(f"No test file: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df

def split_features_target(
    train_df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:

    if target_column not in train_df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    return X, y

def validate_data(
    train_df: pd.DataFrame,
    test_df:pd.DataFrame,
    target_column: str,
    id_column: str,
) -> None:

    "Validate data"

    if target_column not in train_df.columns:
        raise ValueError(f"Target column missing from train: {target_column}")
    
    if target_column in test_df.columns:
        raise ValueError(f"Target column unexpectedly in test: {target_column}")    
    
    if id_column not in train_df.columns:
        raise ValueError(f"Id column missing from train: {id_column}")
    
    if id_column not in test_df.columns:
        raise ValueError(f"Id column missin from test: {id_column}")

    train_features = train_df.drop(columns=[target_column]).columns.tolist()
    test_features = test_df.columns.tolist()

    if train_features != test_features:
        raise ValueError("Train and test features or their orders differ")

     

