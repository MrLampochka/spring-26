from __future__ import annotations

import numpy as np


class Node:
    """Узел дерева. Лист хранит распределение классов, внутренний — сплит."""

    __slots__ = ("feature", "threshold", "left", "right", "proba", "q", "gain", "n")

    def __init__(self, proba, n, feature=None, threshold=None, q=(0.5, 0.5), gain=None):
        self.feature = feature      # None => лист
        self.threshold = threshold
        self.left = self.right = None
        self.proba = proba          # вектор вероятностей классов
        self.q = q                  # доли ушедших влево/вправо (для NaN)
        self.gain = gain
        self.n = n                  # взвешенное число объектов

    @property
    def is_leaf(self):
        return self.feature is None


class DecisionTree:
    """Решающее дерево с вероятностной обработкой пропусков."""

    def __init__(self, max_depth=6, min_samples_split=10, min_samples_leaf=1,
                 criterion="gini", random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        self.root = self._grow(X, self._encode(y), np.ones(len(y)), depth=0)
        return self

    def _encode(self, y):
        """Метки -> индексы 0..K-1, чтобы работать через bincount."""
        return np.searchsorted(self.classes_, y)

    def _grow(self, X, y, w, depth):
        proba = self._proba(y, w)
        node = Node(proba, float(w.sum()))

        if (depth >= self.max_depth or w.sum() < self.min_samples_split
                or len(np.unique(y)) <= 1):
            return node

        split = self._best_split(X, y, w)
        if split is None:
            return node

        feature, threshold, gain = split
        column = X[:, feature]
        known = ~np.isnan(column)
        left, right, missing = known & (column <= threshold), known & (column > threshold), ~known

        w_left, w_right = w[left].sum(), w[right].sum()
        q_left = w_left / (w_left + w_right)

        node.feature, node.threshold, node.gain = feature, threshold, gain
        node.q = (q_left, 1.0 - q_left)

        # Объект с пропуском идёт в обе ветви с весами q_left / q_right
        for child, mask, q in (("left", left, q_left), ("right", right, 1.0 - q_left)):
            idx = np.concatenate([np.where(mask)[0], np.where(missing)[0]])
            weights = np.concatenate([w[mask], w[missing] * q])
            setattr(node, child, self._grow(X[idx], y[idx], weights, depth + 1))
        return node

    def _proba(self, y, w):
        total = w.sum()
        counts = np.bincount(y, weights=w, minlength=len(self.classes_))
        return counts / total if total > 0 else counts

    def _impurity(self, counts, total):
        if total <= 0:
            return 0.0
        p = counts / total
        if self.criterion == "gini":
            return 1.0 - float(p @ p)
        p = p[p > 0]
        return -float(p @ np.log2(p))

    def _best_split(self, X, y, w):
        """Лучшая пара (признак, порог). Прирост штрафуется долей известных значений."""
        best, best_gain = None, 0.0
        total_weight = w.sum()

        for feature in range(X.shape[1]):
            column = X[:, feature]
            known = ~np.isnan(column)
            if known.sum() < 2:
                continue

            order = np.argsort(column[known], kind="mergesort")
            values, labels, weights = column[known][order], y[known][order], w[known][order]
            w_known = weights.sum()
            if w_known < self.min_samples_split:
                continue

            totals = np.bincount(labels, weights=weights, minlength=len(self.classes_))
            parent = self._impurity(totals, w_known)
            penalty = w_known / total_weight   # штраф Quinlan за пропуски

            left_counts = np.zeros(len(self.classes_))
            w_left = 0.0
            for i in range(1, len(values)):
                left_counts[labels[i - 1]] += weights[i - 1]
                w_left += weights[i - 1]
                if values[i] == values[i - 1]:
                    continue
                w_right = w_known - w_left
                if min(w_left, w_right) < self.min_samples_leaf:
                    continue

                children = (w_left * self._impurity(left_counts, w_left)
                            + w_right * self._impurity(totals - left_counts, w_right)) / w_known
                gain = penalty * (parent - children)
                if gain > best_gain:
                    best_gain = gain
                    best = (feature, float((values[i - 1] + values[i]) / 2), float(gain))
        return best

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._walk(row, self.root) for row in X])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def _walk(self, row, node):
        if node.is_leaf:
            return node.proba
        value = row[node.feature]
        if np.isnan(value):
            # Объект "растекается" по обеим ветвям с весами q
            return (node.q[0] * self._walk(row, node.left)
                    + node.q[1] * self._walk(row, node.right))
        return self._walk(row, node.left if value <= node.threshold else node.right)

    def prune(self, X_val, y_val):
        """Обрезка снизу вверх: в каждом узле выбирается лучший из 4 вариантов."""
        X_val = np.asarray(X_val, dtype=float)
        self.root = self._prune(self.root, X_val, self._encode(np.asarray(y_val, dtype=int)))
        return self

    def _prune(self, node, X, y):
        if node.is_leaf or len(y) == 0:
            return node

        column = X[:, node.feature]
        known = ~np.isnan(column)
        left, right = known & (column <= node.threshold), known & (column > node.threshold)
        node.left = self._prune(node.left, X[left], y[left])
        node.right = self._prune(node.right, X[right], y[right])

        leaf = Node(node.proba, node.n)
        # варианты: поддерево / лист / левая ветвь / правая ветвь
        options = [node, leaf, node.left, node.right]
        errors = [self._errors(option, X, y) for option in options]
        return options[int(np.argmin(errors))]

    def _errors(self, node, X, y):
        predictions = [np.argmax(self._walk(row, node)) for row in X]
        return int(np.sum(np.array(predictions, dtype=int) != y))

    def n_leaves(self, node=None):
        node = node or self.root
        return 1 if node.is_leaf else self.n_leaves(node.left) + self.n_leaves(node.right)

    def depth(self, node=None):
        node = node or self.root
        return 0 if node.is_leaf else 1 + max(self.depth(node.left), self.depth(node.right))
