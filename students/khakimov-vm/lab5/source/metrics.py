from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def ndcg_at_k(users, y_true, y_pred, k=10) -> float:
    """NDCG@k: качество ранжирования, усреднённое по пользователям.

    DCG@k = Σ (2^rel_p - 1) / log2(p + 1), нормируется на идеальный порядок.
    """
    users = np.asarray(users)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    scores = []
    discount = 1.0 / np.log2(np.arange(2, k + 2))

    for user in np.unique(users):
        mask = users == user
        if mask.sum() < 2:
            # Ранжировать нечего: у пользователя один тестовый фильм
            continue

        true_ratings = y_true[mask]
        order = np.argsort(-y_pred[mask])
        gains = (2 ** true_ratings[order] - 1)[:k]
        dcg = float(np.sum(gains * discount[:len(gains)]))

        ideal_gains = (2 ** np.sort(true_ratings)[::-1] - 1)[:k]
        idcg = float(np.sum(ideal_gains * discount[:len(ideal_gains)]))

        if idcg > 0:
            scores.append(dcg / idcg)

    return float(np.mean(scores)) if scores else float("nan")


def coverage(predicted_matrix, k=10) -> float:
    """Доля каталога, попадающая хотя бы в чей-то топ-k."""
    top_items = set()
    for row in predicted_matrix:
        top_items.update(np.argsort(-row)[:k].tolist())
    return len(top_items) / predicted_matrix.shape[1]
