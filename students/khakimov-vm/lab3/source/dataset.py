from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATASET_ID = "camnugent/california-housing-prices"
TARGET = "median_house_value"


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
        "mean": df.select_dtypes("number").mean().round(2),
        "std": df.select_dtypes("number").std().round(2),
    })


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature engineering, one-hot для категорий, импутация медианой."""
    df = df.copy()

    # Абсолютные счётчики по округу малоинформативны — важны душевые показатели
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]

    # Единственный категориальный признак — one-hot (всего 5 категорий)
    dummies = pd.get_dummies(df["ocean_proximity"], prefix="ocean").astype(float)
    df = pd.concat([df.drop(columns=["ocean_proximity"]), dummies], axis=1)

    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET]).astype(float)

    # Целевая переменная обрезана сверху на 500001 — такие объекты портят метрику
    capped = y >= 500000
    X, y = X[~capped].reset_index(drop=True), y[~capped].reset_index(drop=True)

    # Деревья к пропускам не готовы, поэтому заполняем медианой
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    # Работаем в тысячах долларов — так метрики читаются легче
    return X, y / 1000.0
