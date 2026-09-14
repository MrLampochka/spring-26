from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import sparse

DATASET_ID = "abhikjha/movielens-100k"


def _find_file(root: str, name: str) -> str:
    for directory, _, files in os.walk(root):
        if name in files:
            return os.path.join(directory, name)
    raise FileNotFoundError(f"{name} не найден в {root}")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Скачать MovieLens с Kaggle (в репозитории данные не хранятся)."""
    import kagglehub

    path = kagglehub.dataset_download(DATASET_ID)
    ratings = pd.read_csv(_find_file(path, "ratings.csv"))
    movies = pd.read_csv(_find_file(path, "movies.csv"))
    return ratings, movies


def describe(ratings: pd.DataFrame) -> pd.DataFrame:
    n_users = ratings["userId"].nunique()
    n_items = ratings["movieId"].nunique()
    per_user = ratings.groupby("userId").size()
    per_item = ratings.groupby("movieId").size()
    return pd.DataFrame({
        "значение": [
            len(ratings), n_users, n_items,
            f"{len(ratings) / (n_users * n_items) * 100:.2f}%",
            f"{ratings['rating'].mean():.3f}",
            f"{per_user.min()} / {per_user.median():.0f} / {per_user.max()}",
            f"{per_item.min()} / {per_item.median():.0f} / {per_item.max()}",
        ]},
        index=["оценок", "пользователей", "фильмов", "заполненность матрицы",
               "средняя оценка", "оценок на пользователя (min/med/max)",
               "оценок на фильм (min/med/max)"])


def filter_rare(ratings: pd.DataFrame, min_user_ratings=5, min_item_ratings=5):
    """Убрать пользователей и фильмы со слишком малым числом оценок.

    Объекты с двумя-тремя оценками не дают ничего ни обучению, ни оценке
    качества, но заметно раздувают размер матрицы.
    """
    while True:
        item_counts = ratings["movieId"].value_counts()
        ratings = ratings[ratings["movieId"].isin(item_counts[item_counts >= min_item_ratings].index)]
        user_counts = ratings["userId"].value_counts()
        ratings = ratings[ratings["userId"].isin(user_counts[user_counts >= min_user_ratings].index)]

        item_counts = ratings["movieId"].value_counts()
        if item_counts.min() >= min_item_ratings:
            break
    return ratings.reset_index(drop=True)


def split_by_user(ratings: pd.DataFrame, test_size=0.2, random_state=42):
    """Разбиение по пользователям: у каждого часть оценок уходит в тест."""
    rng = np.random.default_rng(random_state)
    test_mask = np.zeros(len(ratings), dtype=bool)

    for _, index in ratings.groupby("userId").indices.items():
        n_test = int(round(len(index) * test_size))
        if n_test == 0 or len(index) - n_test < 2:
            continue
        test_mask[rng.choice(index, size=n_test, replace=False)] = True

    return ratings[~test_mask].reset_index(drop=True), ratings[test_mask].reset_index(drop=True)


def build_matrix(train: pd.DataFrame, test: pd.DataFrame):
    """Построить разреженную матрицу оценок и индексы пользователей/фильмов."""
    users = np.sort(train["userId"].unique())
    items = np.sort(train["movieId"].unique())
    user_index = {u: i for i, u in enumerate(users)}
    item_index = {m: i for i, m in enumerate(items)}

    rows = train["userId"].map(user_index).values
    cols = train["movieId"].map(item_index).values
    R = sparse.csr_matrix((train["rating"].values.astype(float), (rows, cols)),
                          shape=(len(users), len(items)))

    # В тесте оставляем только пары, у которых и пользователь, и фильм есть
    # в обучении — иначе это задача холодного старта, а не оценка модели.
    known = test["userId"].isin(user_index) & test["movieId"].isin(item_index)
    test = test[known]
    test_users = test["userId"].map(user_index).values
    test_items = test["movieId"].map(item_index).values
    test_ratings = test["rating"].values.astype(float)

    return R, (test_users, test_items, test_ratings), user_index, item_index
