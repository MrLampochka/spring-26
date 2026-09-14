import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

import plots
from dataset import TARGET, describe, load, preprocess
from ensemble import BaseForest, RandomForest, RandomSubspaceMethod

warnings.filterwarnings("ignore")
SEED = 42
N_TREES = 200


def oob_scorer(estimator, X, y):
    """Скорер для GridSearchCV: по заданию параметры подбираются по OOB.

    X и y не нужны — оценка уже посчитана внутри ансамбля на объектах вне
    бутстрэпа, поэтому утечки нет.
    """
    return estimator.oob_score_


def timed(model, X, y):
    start = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - start


def main():
    # 1. Данные
    raw = load()
    print(f"\n[1] Датасет: {raw.shape[0]} объектов, {raw.shape[1]} колонок")
    print(describe(raw).to_string())

    X_df, y_series, classes = preprocess(raw)
    print(f"\nПосле предобработки: {X_df.shape[0]} объектов, {X_df.shape[1]} признаков, "
          f"классы {classes}, пропусков {int(X_df.isna().sum().sum())}")
    plots.class_balance(raw[TARGET].value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.25, random_state=SEED, stratify=y_series)
    Xtr, Xte, ytr, yte = X_train.values, X_test.values, y_train.values, y_test.values
    features = list(X_df.columns)
    print(f"train: {len(ytr)}, test: {len(yte)}")

    # 2. Подбор гиперпараметров по OOB
    print("\n[2] GridSearchCV, критерий отбора — OOB")
    grid = {"n_estimators": [100], "max_features": ["sqrt", 0.5, None],
            "max_depth": [10, 20, None], "min_samples_leaf": [1, 3, 10]}
    # Один "фолд" на всей выборке: OOB считается внутри модели, внешняя CV не нужна
    single_fold = [(np.arange(len(ytr)), np.arange(len(ytr)))]

    best = {}
    for name, cls in (("Random Forest", RandomForest), ("RSM", RandomSubspaceMethod)):
        search = GridSearchCV(cls(random_state=SEED), grid, scoring=oob_scorer,
                              cv=single_fold, n_jobs=-1, refit=False)
        start = time.perf_counter()
        search.fit(Xtr, ytr)
        best[name] = dict(search.best_params_)

        table = pd.DataFrame(search.cv_results_)[
            ["param_max_features", "param_max_depth", "param_min_samples_leaf", "mean_test_score"]
        ].rename(columns={"mean_test_score": "OOB accuracy"}).sort_values(
            "OOB accuracy", ascending=False)
        print(f"\n{name}: {len(table)} комбинаций за {time.perf_counter() - start:.1f} c")
        print(f"лучшие: {search.best_params_}, OOB = {search.best_score_:.4f}")
        print(table.head(5).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 3. Сходимость OOB
    print("\n[3] Сходимость OOB по числу деревьев")
    n_values = [5, 10, 25, 50, 100, 150, 200, 300]
    curves = {}
    for name, cls in (("Random Forest", RandomForest), ("RSM", RandomSubspaceMethod)):
        curves[name] = [cls(**{**best["Random Forest"], "n_estimators": n},
                            random_state=SEED).fit(Xtr, ytr).oob_score_ for n in n_values]
    print(pd.DataFrame(curves, index=n_values).rename_axis("деревьев").to_string(
        float_format=lambda v: f"{v:.4f}"))
    plots.oob_convergence(n_values, curves)

    # 4. Сравнение с эталонами
    print("\n[4] Сравнение с эталонами sklearn")
    rf_params = {**best["Random Forest"], "n_estimators": N_TREES}
    rsm_params = {**best["RSM"], "n_estimators": N_TREES}
    # BaggingClassifier переводит долю в число через int(), своя реализация — round().
    # Передаём целое, посчитанное одинаково, иначе сравнение неэквивалентно.
    k = BaseForest(max_features=rsm_params["max_features"]).n_features_used(len(features))
    k_half = BaseForest(max_features=0.5).n_features_used(len(features))

    def bagging(n_features, params):
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=params["max_depth"],
                                             min_samples_leaf=params["min_samples_leaf"],
                                             random_state=SEED),
            n_estimators=params["n_estimators"], max_features=n_features,
            bootstrap=True, oob_score=True, n_jobs=1, random_state=SEED)

    candidates = [
        ("своя: Random Forest", RandomForest(**rf_params, random_state=SEED)),
        ("своя: RSM", RandomSubspaceMethod(**rsm_params, random_state=SEED)),
        ("эталон: sklearn RandomForest", RandomForestClassifier(
            **{k_: v for k_, v in rf_params.items()}, oob_score=True,
            bootstrap=True, n_jobs=1, random_state=SEED)),
        ("эталон: sklearn Bagging (RSM)", bagging(k, rsm_params)),
        (f"своя: RSM ({k_half} из {len(features)} признаков)",
         RandomSubspaceMethod(**{**rsm_params, "max_features": 0.5}, random_state=SEED)),
        (f"эталон: sklearn Bagging ({k_half} из {len(features)})",
         bagging(k_half, rsm_params)),
        ("одно дерево (baseline)", DecisionTreeClassifier(random_state=SEED)),
    ]

    results, models = [], {}
    for name, model in candidates:
        model, elapsed = timed(model, Xtr, ytr)
        models[name] = model
        prediction = model.predict(Xte)
        results.append({"модель": name,
                        "OOB": getattr(model, "oob_score_", np.nan),
                        "accuracy": accuracy_score(yte, prediction),
                        "f1_macro": f1_score(yte, prediction, average="macro"),
                        "обучение, с": elapsed})

    table = pd.DataFrame(results)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    plots.comparison(table)
    plots.confusion(models["своя: Random Forest"], Xte, yte, classes)

    # 5. Важности через OOB^j
    print("\n[5] Важность признаков через OOB^j")
    forest = models["своя: Random Forest"]
    start = time.perf_counter()
    values = forest.oob_importances(Xtr, ytr, n_repeats=3, random_state=SEED)
    print(f"посчитано за {time.perf_counter() - start:.1f} c")

    importances = pd.DataFrame(
        {"OOB^j": values,
         "Gini (sklearn)": models["эталон: sklearn RandomForest"].feature_importances_},
        index=features).sort_values("OOB^j", ascending=False)
    print(importances.to_string(float_format=lambda v: f"{v:.4f}"))
    plots.importances(importances)

    top5 = list(importances.head(5).index)
    columns = [features.index(f) for f in top5]
    reduced, elapsed = timed(RandomForest(**rf_params, random_state=SEED), Xtr[:, columns], ytr)
    print(f"\nЛес только на топ-5 {top5}:")
    print(f"  accuracy = {accuracy_score(yte, reduced.predict(Xte[:, columns])):.4f} "
          f"против {table.iloc[0]['accuracy']:.4f} на всех {len(features)}, {elapsed:.2f} с")

    print(f"\nГрафики: {plots.IMAGES}")


if __name__ == "__main__":
    main()
