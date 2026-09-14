from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATASET_ID = "uciml/red-wine-quality-cortez-et-al-2009"
TARGET = "quality"


def load() -> pd.DataFrame:
    """Скачать датасет с Kaggle (в репозитории данные не хранятся)."""
    import kagglehub

    path = kagglehub.dataset_download(DATASET_ID)
    csv_files = [f for f in sorted(os.listdir(path)) if f.endswith(".csv")]
    return pd.read_csv(os.path.join(path, csv_files[0]))


def describe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "n_missing": df.isna().sum(),
        "mean": df.mean().round(3),
        "std": df.std().round(3),
        "min": df.min().round(3),
        "max": df.max().round(3),
    })


def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Стандартизация признаков; возвращает X, метки качества и их названия.

    Стандартизация обязательна: признаки различаются по масштабу на три порядка
    (плотность ~0.997 против общего диоксида серы ~46), и без неё ковариационные
    матрицы компонент получаются вырожденными.
    """
    df = df.copy().drop_duplicates().reset_index(drop=True)

    y = df[TARGET].astype(int).values
    X = df.drop(columns=[TARGET]).astype(float)
    feature_names = list(X.columns)

    X = X.values
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y, feature_names


def binarize_quality(y: np.ndarray, threshold: int = 6) -> np.ndarray:
    """Бинарная метка «хорошее вино» (quality >= threshold) для проверки GMM-байеса."""
    return (y >= threshold).astype(int)
