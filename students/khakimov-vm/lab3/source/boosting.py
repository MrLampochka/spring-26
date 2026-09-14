from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor

EPS = 1e-12


class SquaredLoss:
    """L = ½(y - F)². Антиградиент равен остатку, шаг Ньютона не нужен."""

    is_classification = False

    def init(self, y):
        return float(np.mean(y))          # константа, минимизирующая MSE

    def gradient(self, y, F):
        return y - F

    def update_leaves(self, tree, X, y, F, residual, mask):
        return                            # среднее остатков в листе уже оптимально

    def value(self, y, F):
        return float(np.mean((y - F) ** 2) / 2)


class LogisticLoss:
    """Логистическая потеря; антиградиент y - p, длина шага — по Ньютону."""

    is_classification = True

    @staticmethod
    def sigmoid(F):
        return 1.0 / (1.0 + np.exp(-np.clip(F, -50, 50)))

    def init(self, y):
        p = float(np.clip(np.mean(y), EPS, 1 - EPS))
        return float(np.log(p / (1 - p)))  # логит базовой частоты

    def gradient(self, y, F):
        return y - self.sigmoid(F)

    def update_leaves(self, tree, X, y, F, residual, mask):
        """Шаг Ньютона: gamma_j = Σ r_i / Σ p_i(1-p_i) по объектам листа.

        Записываем прямо в дерево, подменяя значения листьев, — дальше можно
        пользоваться обычным tree.predict.
        """
        leaves = tree.apply(X[mask])
        r = residual[mask]
        p = self.sigmoid(F[mask])
        hessian = p * (1.0 - p)

        for leaf in np.unique(leaves):
            here = leaves == leaf
            denominator = float(hessian[here].sum())
            gamma = 0.0 if denominator < EPS else float(
                np.clip(r[here].sum() / denominator, -10, 10))
            tree.tree_.value[leaf, 0, 0] = gamma

    def value(self, y, F):
        p = np.clip(self.sigmoid(F), EPS, 1 - EPS)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


LOSSES = {"squared_error": SquaredLoss, "log_loss": LogisticLoss}


class GradientBoosting:
    """Градиентный бустинг для регрессии (MSE) и бинарной классификации."""

    _params = ("n_estimators", "learning_rate", "max_depth", "min_samples_leaf",
               "min_samples_split", "subsample", "loss", "max_features", "random_state")

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=1, min_samples_split=2, subsample=1.0,
                 loss="squared_error", max_features=None, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.loss = loss
        self.max_features = max_features
        self.random_state = random_state

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self._params}

    def set_params(self, **params):
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n = len(y)
        self.loss_ = LOSSES[self.loss]()
        if self.loss_.is_classification:
            self.classes_ = np.unique(y.astype(int))
            y = (y == self.classes_[-1]).astype(float)

        rng = np.random.default_rng(self.random_state)
        self.init_ = self.loss_.init(y)
        F = np.full(n, self.init_)
        self.trees_, self.train_loss_ = [], []

        for _ in range(self.n_estimators):
            residual = self.loss_.gradient(y, F)          # 1. антиградиент

            if self.subsample < 1.0:                      # 2. стохастический бустинг
                size = max(1, int(round(self.subsample * n)))
                mask = np.zeros(n, dtype=bool)
                mask[rng.choice(n, size, replace=False)] = True
            else:
                mask = np.ones(n, dtype=bool)

            tree = DecisionTreeRegressor(                 # 3. дерево на антиградиент
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split, max_features=self.max_features,
                random_state=int(rng.integers(0, 2 ** 31 - 1)))
            tree.fit(X[mask], residual[mask])

            self.loss_.update_leaves(tree, X, y, F, residual, mask)   # 4. шаг Ньютона
            F += self.learning_rate * tree.predict(X)                 # 5. шаг композиции

            self.trees_.append(tree)
            self.train_loss_.append(self.loss_.value(y, F))
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.init_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F

    def staged_predict(self, X):
        """Предсказания после каждого добавленного дерева — для кривых обучения."""
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.init_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
            yield self.classes_[(F > 0).astype(int)] if self.loss_.is_classification else F.copy()

    def predict(self, X):
        F = self.decision_function(X)
        return self.classes_[(F > 0).astype(int)] if self.loss_.is_classification else F

    def predict_proba(self, X):
        p = LogisticLoss.sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    @property
    def feature_importances_(self):
        """Среднее ненормированных важностей по деревьям, нормировка в конце."""
        total = np.mean([tree.tree_.compute_feature_importances(normalize=False)
                         for tree in self.trees_], axis=0)
        return total / total.sum() if total.sum() > 0 else total
