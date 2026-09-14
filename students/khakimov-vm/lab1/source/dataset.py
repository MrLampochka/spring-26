from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

DATASET_ID = "yasserh/titanic-dataset"
TARGET = "Survived"

# Редкие титулы схлопываем в общие группы, иначе получим десяток категорий по 1-2 объекта.
TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Officer", "Rev": "Officer", "Col": "Officer", "Major": "Officer",
    "Capt": "Officer", "Don": "Royalty", "Sir": "Royalty", "Lady": "Royalty",
    "Countess": "Royalty", "Jonkheer": "Royalty", "Dona": "Royalty",
    "Mme": "Mrs", "Ms": "Miss", "Mlle": "Miss",
}


def load() -> pd.DataFrame:
    """Скачать датасет с Kaggle (в репозитории данные не хранятся)."""
    import kagglehub

    path = kagglehub.dataset_download(DATASET_ID)
    csv_files = [f for f in sorted(os.listdir(path)) if f.endswith(".csv")]
    return pd.read_csv(os.path.join(path, csv_files[0]))


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Сводка по признакам: тип, число пропусков, число уникальных значений."""
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })
    return summary


def _extract_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", str(name))
    if not match:
        return "Other"
    return TITLE_MAP.get(match.group(1).strip(), "Other")


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature engineering + кодирование категорий. Пропуски сохраняются как NaN."""
    df = df.copy()

    # Title из имени: сильный признак (пол + социальный статус + возрастная группа)
    df["Title"] = df["Name"].map(_extract_title)
    # Палуба — первая буква номера каюты; у 77% пассажиров каюта неизвестна
    df["Deck"] = df["Cabin"].astype(str).str[0].where(df["Cabin"].notna())
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Категориальные признаки кодируем порядковыми кодами; NaN остаётся NaN.
    for column in ["Sex", "Embarked", "Title", "Deck"]:
        codes = pd.Categorical(df[column]).codes.astype(float)
        codes[codes < 0] = np.nan  # pandas кодирует пропуск как -1
        df[column] = codes

    features = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
        "Embarked", "Title", "Deck", "FamilySize", "IsAlone",
    ]
    X = df[features].astype(float)
    y = df[TARGET].astype(int)
    return X, y


def median_impute(X_train: pd.DataFrame, *frames: pd.DataFrame):
    """Заполнение пропусков медианой обучающей выборки.

    Нужно только для эталонной реализации sklearn и для базового CART —
    они не умеют работать с NaN.
    """
    medians = X_train.median()
    return tuple(frame.fillna(medians) for frame in (X_train, *frames))
