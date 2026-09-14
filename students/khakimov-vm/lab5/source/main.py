import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import ElasticNet

import plots
from dataset import build_matrix, describe, filter_rare, load, split_by_user
from metrics import coverage, mae, ndcg_at_k, rmse
from models import ALSMatrixFactorization, BaselinePredictor, SLIM

warnings.filterwarnings("ignore")
SEED = 42
RATING_RANGE = (0.5, 5.0)


def clip(predictions):
    return np.clip(predictions, *RATING_RANGE)


def score(name, y_true, y_pred, users, fit_time):
    return {"модель": name, "RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred),
            "NDCG@10": ndcg_at_k(users, y_true, y_pred, k=10), "обучение, с": fit_time}


def slim_elasticnet(S, l1_reg, l2_reg, n_neighbors=250):
    """Эталон: та же задача, решаемая ElasticNet из sklearn.

    Параметризация разная. sklearn минимизирует
        1/(2n)·||y - Xw||² + a·r·||w||_1 + 0.5·a·(1-r)·||w||²,
    наша реализация — 0.5·||y - Xw||² + l1·||w||_1 + 0.5·l2·||w||².
    Домножив первое на n: l1 = n·a·r, l2 = n·a·(1-r). Без этого пересчёта
    сравниваются модели с разной силой регуляризации.
    """
    S = sparse.csr_matrix(S)
    n_items, n_samples = S.shape[1], S.shape[0]
    alpha = (l1_reg + l2_reg) / n_samples
    l1_ratio = l1_reg / (l1_reg + l2_reg)
    neighbors = SLIM.cosine_neighbors(S, n_neighbors) if n_neighbors < n_items - 1 else None

    rows, cols, values = [], [], []
    for j in range(n_items):
        index = neighbors[j] if neighbors is not None else np.setdiff1d(np.arange(n_items), [j])
        index = index[index != j]
        if len(index) == 0:
            continue
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, positive=True,
                           fit_intercept=False, max_iter=100, tol=1e-4,
                           selection="random", random_state=SEED)
        model.fit(S[:, index], np.asarray(S[:, j].todense()).ravel())
        nonzero = np.nonzero(model.coef_)[0]
        rows.extend(index[nonzero])
        cols.extend([j] * len(nonzero))
        values.extend(model.coef_[nonzero])

    return sparse.csr_matrix((values, (rows, cols)), shape=(n_items, n_items)), alpha, l1_ratio


def main():
    # 1. Данные
    ratings, movies = load()
    print("\n[1] Датасет загружен")
    print(describe(ratings).to_string())
    plots.dataset_stats(ratings)

    ratings = filter_rare(ratings, min_user_ratings=10, min_item_ratings=20)
    print("\nПосле фильтрации редких пользователей и фильмов:")
    print(describe(ratings).to_string())

    train, test = split_by_user(ratings, test_size=0.2, random_state=SEED)
    R, (test_users, test_items, test_ratings), _, item_index = build_matrix(train, test)
    print(f"\nМатрица: {R.shape[0]} x {R.shape[1]}, {R.nnz} оценок "
          f"({R.nnz / np.prod(R.shape) * 100:.2f}% заполнено), тест {len(test_ratings)}")

    # 2. Базовый предиктор
    print("\n[2] Базовый предиктор (mu + b_u + b_i)")
    results = []
    start = time.perf_counter()
    baseline = BaselinePredictor().fit(R)
    baseline_time = time.perf_counter() - start
    baseline_prediction = baseline.predict(test_users, test_items)
    results.append(score("базовый предиктор (mu + b_u + b_i)", test_ratings,
                         clip(baseline_prediction), test_users, baseline_time))
    print(f"mu = {baseline.mu_:.4f}, RMSE = {results[-1]['RMSE']:.4f}, "
          f"NDCG@10 = {results[-1]['NDCG@10']:.4f}")

    S = baseline.residuals(R)     # SLIM и ALS учатся на остатках

    # 3. SLIM
    print("\n[3] SLIM: подбор регуляризации")
    rows = []
    for l1 in (0.05, 0.1, 0.5, 1.0, 3.0, 10.0):
        start = time.perf_counter()
        model = SLIM(l1_reg=l1, l2_reg=5.0).fit(S)
        elapsed = time.perf_counter() - start
        prediction = clip(baseline_prediction + model.predict(S, test_users, test_items))
        rows.append({"l1_reg": l1, "RMSE": rmse(test_ratings, prediction),
                     "NDCG@10": ndcg_at_k(test_users, test_ratings, prediction, k=10),
                     "ненулевых в W": model.W_.nnz,
                     "разреженность W, %": model.sparsity_ * 100, "обучение, с": elapsed})
        print(f"  λ={l1:<5} RMSE={rows[-1]['RMSE']:.4f}  NDCG={rows[-1]['NDCG@10']:.4f}  "
              f"nnz(W)={model.W_.nnz:>7}  ({elapsed:.1f} c)", flush=True)

    table = pd.DataFrame(rows)
    plots.slim_regularization(table)
    best_l1 = float(table.loc[table["RMSE"].idxmin(), "l1_reg"])
    print(f"\nЛучшее λ по RMSE: {best_l1}")

    start = time.perf_counter()
    slim = SLIM(l1_reg=best_l1, l2_reg=5.0).fit(S)
    slim_time = time.perf_counter() - start
    slim_prediction = clip(baseline_prediction + slim.predict(S, test_users, test_items))
    results.append(score("своя: SLIM", test_ratings, slim_prediction, test_users, slim_time))
    print(f"SLIM: RMSE = {results[-1]['RMSE']:.4f}, NDCG@10 = {results[-1]['NDCG@10']:.4f}, "
          f"W заполнена на {(1 - slim.sparsity_) * 100:.2f}%")

    # 4. Эталонный SLIM
    print("\n[4] Эталон: SLIM через sklearn ElasticNet")
    start = time.perf_counter()
    W, alpha, l1_ratio = slim_elasticnet(S, best_l1, 5.0)
    reference_time = time.perf_counter() - start
    print(f"те же λ={best_l1}, β=5.0 в параметризации sklearn: "
          f"alpha={alpha:.6f}, l1_ratio={l1_ratio:.4f}")

    full = np.asarray((S @ W).todense())
    prediction = clip(baseline_prediction + full[test_users, test_items])
    results.append(score("эталон: SLIM на sklearn ElasticNet", test_ratings,
                         prediction, test_users, reference_time))
    print(f"RMSE = {results[-1]['RMSE']:.4f}, NDCG@10 = {results[-1]['NDCG@10']:.4f}, "
          f"nnz(W) = {W.nnz} (у своей {slim.W_.nnz})")

    # 5. Латентная модель
    print("\n[5] Латентная модель: ALS-факторизация")
    coo = R.tocoo()
    rows = []
    for f in (5, 10, 20, 40, 60, 80, 100):
        model = ALSMatrixFactorization(n_factors=f, n_iter=15, random_state=SEED).fit(R)
        rows.append({"n_factors": f,
                     "train RMSE": rmse(coo.data, model.predict(coo.row, coo.col)),
                     "RMSE": rmse(test_ratings, clip(model.predict(test_users, test_items)))})
        print(f"  f={f:<4} train RMSE={rows[-1]['train RMSE']:.4f}  "
              f"test RMSE={rows[-1]['RMSE']:.4f}", flush=True)

    factors = pd.DataFrame(rows)
    best_f = int(factors.loc[factors["RMSE"].idxmin(), "n_factors"])
    print(f"\nЛучшее число факторов: {best_f}")

    start = time.perf_counter()
    als = ALSMatrixFactorization(n_factors=best_f, n_iter=20, random_state=SEED).fit(R)
    als_time = time.perf_counter() - start
    als_prediction = clip(als.predict(test_users, test_items))
    results.append(score("своя: ALS-факторизация", test_ratings, als_prediction,
                         test_users, als_time))
    print(f"ALS: RMSE = {results[-1]['RMSE']:.4f}, NDCG@10 = {results[-1]['NDCG@10']:.4f}")
    plots.als_factors(factors, als.history_)

    # 6. Эталонная латентная модель
    print("\n[6] Эталон: sklearn TruncatedSVD по остаткам")
    start = time.perf_counter()
    svd = TruncatedSVD(n_components=best_f, random_state=SEED)
    reconstructed = svd.fit_transform(S) @ svd.components_
    svd_time = time.perf_counter() - start
    prediction = clip(baseline_prediction + reconstructed[test_users, test_items])
    results.append(score("эталон: sklearn TruncatedSVD", test_ratings, prediction,
                         test_users, svd_time))
    print(f"RMSE = {results[-1]['RMSE']:.4f}, NDCG@10 = {results[-1]['NDCG@10']:.4f}, "
          f"объяснённая дисперсия {svd.explained_variance_ratio_.sum():.3f}")

    # 7. Итог
    print("\n[7] Итоговое сравнение")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    plots.comparison(summary)

    hybrid = clip(0.5 * slim_prediction + 0.5 * als_prediction)
    print(f"\nГибрид (SLIM + ALS, 50/50): RMSE = {rmse(test_ratings, hybrid):.4f}, "
          f"NDCG@10 = {ndcg_at_k(test_users, test_ratings, hybrid, k=10):.4f}")

    matrix = clip(baseline.mu_ + baseline.bu_[:, None] + baseline.bi_[None, :]
                  + slim.predict_matrix(S))
    print(f"Покрытие каталога топ-10 рекомендациями SLIM: "
          f"{coverage(matrix, k=10) * 100:.1f}% фильмов")

    titles = movies.set_index("movieId")["title"].to_dict()
    inverse = {v: k for k, v in item_index.items()}
    scores = matrix[0].copy()
    scores[R[0].indices] = -np.inf          # уже просмотренное не рекомендуем
    print("\nТоп-5 рекомендаций SLIM для первого пользователя:")
    for rank, item in enumerate(np.argsort(-scores)[:5], 1):
        print(f"  {rank}. {titles.get(inverse[item], '?')} (прогноз {scores[item]:.2f})")

    print(f"\nГрафики: {plots.IMAGES}")


if __name__ == "__main__":
    main()
