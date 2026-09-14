from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier


class BaseForest(BaseEstimator, ClassifierMixin):
    """Общая часть RF и RSM. От BaseEstimator наследуемся только ради GridSearchCV."""

    method = "rf"

    def __init__(self, n_estimators=100, max_features="sqrt", max_depth=None,
                 min_samples_leaf=1, min_samples_split=2, bootstrap=True,
                 random_state=42):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.random_state = random_state

    def n_features_used(self, n_features):
        """Перевод max_features в число признаков."""
        value = self.max_features
        if value is None:
            return n_features
        if value == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if value == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(value, float):
            return max(1, int(round(value * n_features)))
        return max(1, min(int(value), n_features))

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_features_in_ = n_features

        rng = np.random.default_rng(self.random_state)
        k = self.n_features_used(n_features)
        self.trees_, self.features_, self.oob_ = [], [], []

        for _ in range(self.n_estimators):
            if self.bootstrap:
                sample = rng.integers(0, n_samples, size=n_samples)
                in_bag = np.zeros(n_samples, dtype=bool)
                in_bag[sample] = True
            else:
                sample, in_bag = np.arange(n_samples), np.ones(n_samples, dtype=bool)

            if self.method == "rsm":
                features, tree_max_features = np.sort(rng.choice(n_features, k, replace=False)), None
            else:
                features, tree_max_features = np.arange(n_features), k

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split, max_features=tree_max_features,
                random_state=int(rng.integers(0, 2 ** 31 - 1)))
            tree.fit(X[np.ix_(sample, features)], y[sample])

            self.trees_.append(tree)
            self.features_.append(features)
            self.oob_.append(~in_bag)

        self.oob_score_ = self._oob_score(X, y) if self.bootstrap else np.nan
        return self

    def _proba(self, tree, X_subset):
        """predict_proba дерева, разложенное по классам ансамбля.

        Дерево могло не увидеть какой-то класс в своей бутстрэп-выборке.
        """
        proba = tree.predict_proba(X_subset)
        if np.array_equal(tree.classes_, self.classes_):
            return proba
        full = np.zeros((len(X_subset), len(self.classes_)))
        full[:, np.searchsorted(self.classes_, tree.classes_)] = proba
        return full

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        total = np.zeros((len(X), len(self.classes_)))
        for tree, features in zip(self.trees_, self.features_):
            total += self._proba(tree, X[:, features])
        return total / len(self.trees_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def _oob_score(self, X, y):
        """Голосуют только те деревья, для которых объект был вне бутстрэпа."""
        votes = np.zeros((len(X), len(self.classes_)))
        counts = np.zeros(len(X), dtype=int)

        for tree, features, oob in zip(self.trees_, self.features_, self.oob_):
            if not oob.any():
                continue
            votes[oob] += self._proba(tree, X[np.ix_(np.where(oob)[0], features)])
            counts[oob] += 1

        covered = counts > 0
        self.oob_coverage_ = float(covered.mean())
        return float(np.mean(self.classes_[np.argmax(votes[covered], axis=1)] == y[covered]))

    def oob_importances(self, X, y, n_repeats=3, random_state=42):
        """Важности OOB^j: падение точности на OOB при перестановке признака j."""
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        rng = np.random.default_rng(random_state)
        total = np.zeros(self.n_features_in_)
        used = np.zeros(self.n_features_in_, dtype=int)

        for tree, features, oob in zip(self.trees_, self.features_, self.oob_):
            index = np.where(oob)[0]
            if len(index) == 0:
                continue
            X_oob, y_oob = X[index][:, features], y[index]
            base = np.mean(tree.predict(X_oob) == y_oob)

            for position, feature in enumerate(features):
                drops = []
                for _ in range(n_repeats):
                    shuffled = X_oob.copy()
                    shuffled[:, position] = shuffled[rng.permutation(len(index)), position]
                    drops.append(base - np.mean(tree.predict(shuffled) == y_oob))
                total[feature] += np.mean(drops)
                used[feature] += 1

        return total / np.maximum(used, 1)


class RandomForest(BaseForest):
    """Бутстрэп объектов + случайное подпространство признаков в каждом узле."""
    method = "rf"


class RandomSubspaceMethod(BaseForest):
    """Одно случайное подпространство признаков на всё дерево."""
    method = "rsm"
