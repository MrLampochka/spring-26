from __future__ import annotations

import os

import pandas as pd

DATASET_ID = "kukuroo3/body-performance-data"
TARGET = "class"


def load() -> pd.DataFrame:
    """Скачать датасет с Kaggle (в репозитории данные не хранятся)."""
    import kagglehub

    path = kagglehub.dataset_download(DATASET_ID)
    csv_files = [f for f in sorted(os.listdir(path)) if f.endswith(".csv")]
    return pd.read_csv(os.path.join(path, csv_files[0]))


def describe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "n_unique": df.nunique(),
        "min": df.select_dtypes("number").min(),
        "max": df.select_dtypes("number").max(),
    })


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Кодирование категорий и целевой переменной."""
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    # Единственный категориальный признак; бинарный, поэтому достаточно 0/1
    df["gender"] = (df["gender"].astype(str).str.strip() == "M").astype(int)

    # В данных встречаются физически невозможные нули давления — трактуем их
    # как ошибки измерения и заменяем медианой.
    for column in ["diastolic", "systolic"]:
        invalid = df[column] <= 0
        if invalid.any():
            df.loc[invalid, column] = df.loc[~invalid, column].median()

    classes = sorted(df[TARGET].astype(str).unique())
    y = df[TARGET].astype(str).map({c: i for i, c in enumerate(classes)}).astype(int)
    X = df.drop(columns=[TARGET]).astype(float)

    # Пробелы в названиях мешают читать графики
    X.columns = [c.strip().replace(" ", "_") for c in X.columns]
    return X, y, classes
