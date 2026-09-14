from __future__ import annotations

import numpy as np

LOG_2PI = np.log(2.0 * np.pi)


def logsumexp(a, axis=None, keepdims=False):
    """Устойчивое log Σ exp(a): максимум выносится за скобку."""
    peak = np.max(a, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    result = np.log(np.sum(np.exp(a - peak), axis=axis, keepdims=True)) + peak
    return result if keepdims else np.squeeze(result, axis=axis)


class GaussianMixtureEM:
    """GMM, обучаемая EM. covariance_type: full | diag | spherical."""

    def __init__(self, n_components=3, covariance_type="full", max_iter=200,
                 tol=1e-4, reg_covar=1e-6, n_init=5, random_state=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.random_state = random_state

    def _kmeans_plus_plus(self, X, rng):
        """Центры разносятся по выборке: вероятность выбора ~ квадрату
        расстояния до ближайшего уже выбранного центра."""
        centers = [X[rng.integers(len(X))]]
        closest = np.sum((X - centers[0]) ** 2, axis=1)

        for _ in range(1, self.n_components):
            total = closest.sum()
            index = rng.integers(len(X)) if total <= 0 else rng.choice(len(X), p=closest / total)
            centers.append(X[index])
            closest = np.minimum(closest, np.sum((X - centers[-1]) ** 2, axis=1))
        return np.array(centers)

    def _initialize(self, X, rng):
        K, d = self.n_components, X.shape[1]
        means = self._kmeans_plus_plus(X, rng)
        weights = np.full(K, 1.0 / K)

        # Стартуем с ковариации всей выборки — широкое безопасное приближение
        global_cov = np.cov(X.T) + self.reg_covar * np.eye(d)
        if self.covariance_type == "full":
            covariances = np.array([global_cov.copy() for _ in range(K)])
        elif self.covariance_type == "diag":
            covariances = np.tile(np.diag(global_cov), (K, 1))
        else:
            covariances = np.full(K, float(np.mean(np.diag(global_cov))))
        return weights, means, covariances

    def _log_gaussian(self, X, means, covariances):
        """Матрица log N(x_i | mu_k, Sigma_k) формы (n, K)."""
        n, d = X.shape
        out = np.empty((n, len(means)))

        for k in range(len(means)):
            diff = X - means[k]
            if self.covariance_type == "full":
                # Sigma = L Lt => (x-mu)t Sigma^-1 (x-mu) = ||L^-1 (x-mu)||²,
                # log det Sigma = 2 Σ log diag(L). Обращать Sigma не нужно.
                L = np.linalg.cholesky(covariances[k])
                mahalanobis = np.sum(np.linalg.solve(L, diff.T) ** 2, axis=0)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
            elif self.covariance_type == "diag":
                mahalanobis = np.sum(diff ** 2 / covariances[k], axis=1)
                log_det = np.sum(np.log(covariances[k]))
            else:
                mahalanobis = np.sum(diff ** 2, axis=1) / covariances[k]
                log_det = d * np.log(covariances[k])
            out[:, k] = -0.5 * (d * LOG_2PI + log_det + mahalanobis)
        return out

    def _log_weighted(self, X, weights, means, covariances):
        return self._log_gaussian(X, means, covariances) + np.log(weights)

    def _e_step(self, X, weights, means, covariances):
        """Ответственности gamma и текущее среднее лог-правдоподобие."""
        weighted = self._log_weighted(X, weights, means, covariances)
        log_norm = logsumexp(weighted, axis=1, keepdims=True)      # log p(x_i)
        return np.exp(weighted - log_norm), float(np.mean(log_norm))

    def _m_step(self, X, gamma):
        """Взвешенные оценки весов, средних и ковариаций."""
        n, d = X.shape
        nk = gamma.sum(axis=0) + 10 * np.finfo(float).eps   # эффективный размер компоненты
        weights = nk / n
        means = (gamma.T @ X) / nk[:, None]

        if self.covariance_type == "full":
            covariances = np.empty((self.n_components, d, d))
            for k in range(self.n_components):
                diff = X - means[k]
                covariances[k] = (gamma[:, k] * diff.T) @ diff / nk[k]
                covariances[k].flat[:: d + 1] += self.reg_covar
        else:
            second = (gamma.T @ (X ** 2)) / nk[:, None]
            covariances = second - means ** 2 + self.reg_covar
            if self.covariance_type == "spherical":
                covariances = np.mean(covariances, axis=1)
        return weights, means, covariances

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best = None

        for _ in range(self.n_init):
            weights, means, covariances = self._initialize(X, rng)
            history, previous, converged, n_iter = [], -np.inf, False, 0

            for iteration in range(self.max_iter):
                gamma, value = self._e_step(X, weights, means, covariances)
                weights, means, covariances = self._m_step(X, gamma)
                history.append(value)
                n_iter = iteration + 1
                if abs(value - previous) < self.tol:
                    converged = True
                    break
                previous = value

            _, value = self._e_step(X, weights, means, covariances)
            history.append(value)

            if best is None or value > best[0]:
                best = (value, weights, means, covariances, history, n_iter, converged)

        (self.lower_bound_, self.weights_, self.means_, self.covariances_,
         self.history_, self.n_iter_, self.converged_) = best
        self.n_features_in_ = X.shape[1]
        return self

    def score_samples(self, X):
        """log p(x_i) для каждого объекта."""
        X = np.asarray(X, dtype=float)
        return logsumexp(self._log_weighted(X, self.weights_, self.means_,
                                            self.covariances_), axis=1)

    def score(self, X):
        """Среднее логарифмическое правдоподобие на объект."""
        return float(np.mean(self.score_samples(X)))

    def predict_proba(self, X):
        weighted = self._log_weighted(np.asarray(X, dtype=float), self.weights_,
                                      self.means_, self.covariances_)
        return np.exp(weighted - logsumexp(weighted, axis=1, keepdims=True))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def n_parameters(self):
        K, d = self.n_components, self.n_features_in_
        covariance = {"full": K * d * (d + 1) // 2, "diag": K * d}.get(self.covariance_type, K)
        return int(K * d + K - 1 + covariance)

    def bic(self, X):
        n = len(X)
        return float(-2 * self.score(X) * n + self.n_parameters() * np.log(n))

    def aic(self, X):
        return float(-2 * self.score(X) * len(X) + 2 * self.n_parameters())


class GMMBayesClassifier:
    """Байес на смесях: P(y=c|x) ~ P(y=c)·p(x|y=c), где p(x|c) — обученная EM GMM."""

    def __init__(self, n_components=2, covariance_type="full", reg_covar=1e-4,
                 n_init=3, max_iter=200, random_state=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.priors_ = np.array([np.mean(y == c) for c in self.classes_])
        self.models_ = [
            GaussianMixtureEM(n_components=self.n_components,
                              covariance_type=self.covariance_type,
                              reg_covar=self.reg_covar, n_init=self.n_init,
                              max_iter=self.max_iter,
                              random_state=self.random_state).fit(X[y == c])
            for c in self.classes_]
        return self

    def predict_proba(self, X):
        joint = np.column_stack([model.score_samples(X) + np.log(prior)
                                 for model, prior in zip(self.models_, self.priors_)])
        return np.exp(joint - logsumexp(joint, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
