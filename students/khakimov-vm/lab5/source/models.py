from __future__ import annotations

import numpy as np
from scipy import sparse


class BaselinePredictor:
    """r = mu + b_u + b_i; смещения оцениваются попеременно с L2."""

    def __init__(self, n_iter=15, reg_user=10.0, reg_item=10.0):
        self.n_iter = n_iter
        self.reg_user = reg_user
        self.reg_item = reg_item

    def fit(self, R):
        R = sparse.csr_matrix(R).tocoo()
        n_users, n_items = R.shape
        rows, cols, values = R.row, R.col, R.data

        self.mu_ = float(values.mean())
        self.bu_ = np.zeros(n_users)
        self.bi_ = np.zeros(n_items)
        user_counts = np.bincount(rows, minlength=n_users)
        item_counts = np.bincount(cols, minlength=n_items)

        for _ in range(self.n_iter):
            residual = values - self.mu_ - self.bi_[cols]
            self.bu_ = np.bincount(rows, weights=residual, minlength=n_users) / (
                self.reg_user + user_counts)
            residual = values - self.mu_ - self.bu_[rows]
            self.bi_ = np.bincount(cols, weights=residual, minlength=n_items) / (
                self.reg_item + item_counts)
        return self

    def predict(self, users, items):
        return self.mu_ + self.bu_[users] + self.bi_[items]

    def residuals(self, R):
        """Матрица остатков: из каждой оценки вычтен базовый прогноз."""
        R = sparse.csr_matrix(R).tocoo()
        residual = R.data - self.predict(R.row, R.col)
        return sparse.csr_matrix((residual, (R.row, R.col)), shape=R.shape)


class SLIM:
    """Sparse Linear Method: R_hat = R·W с разреженной item-item матрицей W.

    По столбцам распадается на независимые elastic net, решаются покоординатным
    спуском. n_neighbors ограничивает кандидатов ближайшими по косинусу.
    """

    def __init__(self, l1_reg=1.0, l2_reg=5.0, max_iter=30, tol=1e-4,
                 n_neighbors=250, positive=True):
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.tol = tol
        self.n_neighbors = n_neighbors
        self.positive = positive

    @staticmethod
    def cosine_neighbors(S, n_neighbors):
        """Для каждого фильма — индексы n_neighbors самых похожих по косинусу."""
        norms = np.sqrt(np.asarray(S.multiply(S).sum(axis=0)).ravel()) + 1e-9
        similarity = (S.T @ S).toarray() / np.outer(norms, norms)
        np.fill_diagonal(similarity, -np.inf)
        return np.argsort(-similarity, axis=1)[:, :n_neighbors]

    def fit(self, S):
        S = sparse.csr_matrix(S)
        n_items = S.shape[1]

        # Грам-матрица: покоординатному спуску нужны только скалярные
        # произведения столбцов, а не сами столбцы
        G = np.asarray((S.T @ S).todense())
        diagonal = np.diag(G).copy()
        neighbors = (self.cosine_neighbors(S, self.n_neighbors)
                     if self.n_neighbors and self.n_neighbors < n_items - 1 else None)

        rows, cols, values = [], [], []
        for j in range(n_items):
            index = (neighbors[j] if neighbors is not None
                     else np.setdiff1d(np.arange(n_items), [j]))
            index = index[index != j]
            if len(index) == 0:
                continue

            G_sub = G[np.ix_(index, index)]
            g_j = G[index, j]
            denominator = diagonal[index] + self.l2_reg

            w = np.zeros(len(index))
            Gw = np.zeros(len(index))       # поддерживаем G_sub @ w инкрементально

            for _ in range(self.max_iter):
                change = 0.0
                for k in range(len(index)):
                    # производная без вклада самой координаты k
                    rho = g_j[k] - (Gw[k] - G_sub[k, k] * w[k])
                    if self.positive:
                        new = max(0.0, rho - self.l1_reg) / denominator[k]
                    else:
                        new = np.sign(rho) * max(0.0, abs(rho) - self.l1_reg) / denominator[k]
                    delta = new - w[k]
                    if delta != 0.0:
                        Gw += delta * G_sub[:, k]
                        w[k] = new
                        change = max(change, abs(delta))
                if change < self.tol:
                    break

            nonzero = np.nonzero(w)[0]
            rows.extend(index[nonzero])
            cols.extend([j] * len(nonzero))
            values.extend(w[nonzero])

        self.W_ = sparse.csr_matrix((values, (rows, cols)), shape=(n_items, n_items))
        self.sparsity_ = 1.0 - self.W_.nnz / (n_items * (n_items - 1))
        return self

    def predict_matrix(self, S):
        return np.asarray((sparse.csr_matrix(S) @ self.W_).todense())

    def predict(self, S, users, items):
        return np.asarray((sparse.csr_matrix(S) @ self.W_)[users, items]).ravel()


class ALSMatrixFactorization:
    """r = mu + b_u + b_i + p_u·q_i, обучается по ALS.

    Каждый шаг — гребневый МНК в явном виде, только по наблюдённым ячейкам.
    """

    def __init__(self, n_factors=20, n_iter=20, reg=0.1, reg_bias=5.0, random_state=42):
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.reg = reg
        self.reg_bias = reg_bias
        self.random_state = random_state

    def fit(self, R):
        R_csr = sparse.csr_matrix(R)
        R_csc = R_csr.tocsc()
        n_users, n_items = R_csr.shape
        rng = np.random.default_rng(self.random_state)

        self.mu_ = float(R_csr.data.mean())
        self.bu_, self.bi_ = np.zeros(n_users), np.zeros(n_items)
        self.P_ = rng.normal(0, 0.05, (n_users, self.n_factors))
        self.Q_ = rng.normal(0, 0.05, (n_items, self.n_factors))

        coo = R_csr.tocoo()
        rows, cols, values = coo.row, coo.col, coo.data
        user_counts = np.bincount(rows, minlength=n_users)
        item_counts = np.bincount(cols, minlength=n_items)
        eye = np.eye(self.n_factors)
        self.history_ = []

        for _ in range(self.n_iter):
            # 1. смещения в явном виде при фиксированных факторах
            interaction = np.sum(self.P_[rows] * self.Q_[cols], axis=1)
            residual = values - self.mu_ - self.bi_[cols] - interaction
            self.bu_ = np.bincount(rows, weights=residual, minlength=n_users) / (
                self.reg_bias + user_counts)
            residual = values - self.mu_ - self.bu_[rows] - interaction
            self.bi_ = np.bincount(cols, weights=residual, minlength=n_items) / (
                self.reg_bias + item_counts)

            # 2. факторы пользователей, затем 3. факторы фильмов
            for u in range(n_users):
                start, end = R_csr.indptr[u], R_csr.indptr[u + 1]
                if start == end:
                    continue
                items = R_csr.indices[start:end]
                target = R_csr.data[start:end] - self.mu_ - self.bu_[u] - self.bi_[items]
                Qu = self.Q_[items]
                self.P_[u] = np.linalg.solve(Qu.T @ Qu + self.reg * len(items) * eye,
                                             Qu.T @ target)

            for i in range(n_items):
                start, end = R_csc.indptr[i], R_csc.indptr[i + 1]
                if start == end:
                    continue
                users = R_csc.indices[start:end]
                target = R_csc.data[start:end] - self.mu_ - self.bu_[users] - self.bi_[i]
                Pi = self.P_[users]
                self.Q_[i] = np.linalg.solve(Pi.T @ Pi + self.reg * len(users) * eye,
                                             Pi.T @ target)

            self.history_.append(
                float(np.sqrt(np.mean((values - self.predict(rows, cols)) ** 2))))
        return self

    def predict(self, users, items):
        return (self.mu_ + self.bu_[users] + self.bi_[items]
                + np.sum(self.P_[users] * self.Q_[items], axis=1))
