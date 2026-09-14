import copy
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

import plots
from dataset import describe, load, median_impute, preprocess
from tree import DecisionTree

warnings.filterwarnings("ignore")
SEED = 42


def evaluate(name, model, X, y, fit_time):
    pred = model.predict(X)
    row = {"модель": name, "accuracy": accuracy_score(y, pred),
           "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0),
           "f1": f1_score(y, pred, zero_division=0), "обучение, с": fit_time}
    row["ROC-AUC"] = roc_auc_score(y, model.predict_proba(X)[:, 1])
    return row


def timed(model, X, y):
    start = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - start


def grid_search(grid, X, y, cv):
    """Свой перебор: sklearn 1.7+ требует от сторонних оценщиков __sklearn_tags__."""
    rows = []
    for params in ParameterGrid(grid):
        scores = [f1_score(y[val], DecisionTree(**params).fit(X[tr], y[tr]).predict(X[val]),
                           zero_division=0)
                  for tr, val in cv.split(X, y)]
        rows.append({**params, "CV f1": np.mean(scores), "std": np.std(scores)})
    table = pd.DataFrame(rows).sort_values("CV f1", ascending=False)
    best = {k: table.iloc[0][k] for k in grid}
    return {k: int(v) if isinstance(v, np.integer) else v for k, v in best.items()}, table


def main():
    # 1. Данные
    raw = load()
    print(f"\n[1] Датасет: {raw.shape[0]} объектов, {raw.shape[1]} признаков")
    print(describe(raw).to_string())
    plots.missing_values(raw)

    X_df, y_series = preprocess(raw)
    print(f"\nПосле предобработки: {X_df.shape[1]} признаков, "
          f"{int(X_df.isna().sum().sum())} пропусков оставлено для дерева")
    print(X_df.isna().sum()[lambda s: s > 0].to_string())
    print(f"Классы: {dict(y_series.value_counts())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.25, random_state=SEED, stratify=y_series)
    X_train_i, X_test_i = median_impute(X_train, X_test)   # для sklearn: он не умеет NaN

    features = list(X_df.columns)
    Xtr, Xte, ytr, yte = X_train.values, X_test.values, y_train.values, y_test.values
    Xtr_i, Xte_i = X_train_i.values, X_test_i.values
    print(f"train: {len(ytr)}, test: {len(yte)}")

    # 2. Подбор гиперпараметров
    print("\n[2] Подбор гиперпараметров (5-fold CV по f1)")
    grid = {"max_depth": [3, 4, 5, 6, 8, 10], "min_samples_split": [2, 10, 20],
            "min_samples_leaf": [1, 5, 10], "criterion": ["gini"]}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    start = time.perf_counter()
    best, table = grid_search(grid, Xtr, ytr, cv)
    print(f"{len(table)} комбинаций за {time.perf_counter() - start:.1f} c")
    print(f"лучшие: {best}, CV f1 = {table.iloc[0]['CV f1']:.4f}")
    print(table.head(5).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 3. Редукция
    print("\n[3] Редукция дерева")
    X_sub, X_val, y_sub, y_val = train_test_split(
        Xtr, ytr, test_size=0.25, random_state=SEED, stratify=ytr)
    print(f"рост дерева: {len(y_sub)}, валидация для обрезки: {len(y_val)}")

    rows, models = [], {}
    configs = [("оптимальная (по CV)", best),
               ("переобученная", {**best, "max_depth": 12,
                                  "min_samples_split": 2, "min_samples_leaf": 1})]
    for label, params in configs:
        before = DecisionTree(**params).fit(X_sub, y_sub)
        after = copy.deepcopy(before).prune(X_val, y_val)
        models[label] = (before, after)
        for state, model in (("до", before), ("после", after)):
            rows.append({"конфигурация": label, "редукция": state,
                         "листьев": model.n_leaves(), "глубина": model.depth(),
                         "f1 train": f1_score(y_sub, model.predict(X_sub), zero_division=0),
                         "f1 val": f1_score(y_val, model.predict(X_val), zero_division=0),
                         "f1 test": f1_score(yte, model.predict(Xte), zero_division=0),
                         "acc test": accuracy_score(yte, model.predict(Xte))})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    plots.pruning_effect(*models["переобученная"], Xte, yte)

    # 4. Сравнение с эталоном
    print("\n[4] Сравнение на тестовой выборке")
    results = []

    own, t = timed(DecisionTree(**best), Xtr, ytr)
    results.append(evaluate("своя: пропуски через вероятности", own, Xte, yte, t))

    imputed, t = timed(DecisionTree(**best), Xtr_i, ytr)
    results.append(evaluate("своя: та же, но с импутацией медианой", imputed, Xte_i, yte, t))

    # Редуцированное дерево берём из шага 3: обрезать по X_val дерево, которое
    # на X_val обучалось, нельзя — оценка ошибки была бы смещённой. Поэтому
    # здесь дерево выросло на X_sub (75% обучающей выборки) и обрезано по X_val.
    grown, t = timed(DecisionTree(**best), X_sub, y_sub)
    pruned = copy.deepcopy(grown).prune(X_val, y_val)
    results.append(evaluate("своя: вероятности + редукция (75% train)", pruned, Xte, yte, t))

    reference, t = timed(DecisionTreeClassifier(
        random_state=SEED, **{k: v for k, v in best.items() if k != "criterion"}), Xtr_i, ytr)
    results.append(evaluate("эталон: sklearn DecisionTreeClassifier", reference, Xte_i, yte, t))

    table = pd.DataFrame(results)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    plots.models_comparison(table)

    # 5. Кривая по глубине
    print("\n[5] Качество в зависимости от глубины")
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    own_train, own_test, reference_test = [], [], []
    for depth in depths:
        model = DecisionTree(max_depth=depth, min_samples_split=10,
                             min_samples_leaf=5).fit(Xtr, ytr)
        own_train.append(f1_score(ytr, model.predict(Xtr), zero_division=0))
        own_test.append(f1_score(yte, model.predict(Xte), zero_division=0))
        sk = DecisionTreeClassifier(max_depth=depth, min_samples_split=10,
                                    min_samples_leaf=5, random_state=SEED).fit(Xtr_i, ytr)
        reference_test.append(f1_score(yte, sk.predict(Xte_i), zero_division=0))

    print(pd.DataFrame({"глубина": depths, "своё train": own_train, "своё test": own_test,
                        "sklearn test": reference_test}).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    plots.depth_curve(depths, own_train, own_test, reference_test)
    plots.tree_structure(own, features)

    print(f"\nГрафики: {plots.IMAGES}")
    print(f"Своё дерево f1={table.iloc[2]['f1']:.4f} против sklearn "
          f"{table.iloc[3]['f1']:.4f} ({table.iloc[2]['f1'] - table.iloc[3]['f1']:+.4f})")


if __name__ == "__main__":
    main()
