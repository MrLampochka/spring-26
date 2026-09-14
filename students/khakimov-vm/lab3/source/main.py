import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeRegressor

import plots
from boosting import GradientBoosting
from dataset import describe, load, preprocess

warnings.filterwarnings("ignore")
SEED, N_TREES, LR, DEPTH = 42, 200, 0.1, 4


def rmse(y, pred):
    return float(np.sqrt(mean_squared_error(y, pred)))


def cross_validate(make_model, X, y, cv):
    """Кросс-валидация с замером времени обучения на каждом фолде."""
    scores = {"RMSE": [], "MAE": [], "R2": [], "время": []}
    for train, val in cv.split(X, y):
        model = make_model()
        start = time.perf_counter()
        model.fit(X[train], y[train])
        scores["время"].append(time.perf_counter() - start)

        pred = model.predict(X[val])
        scores["RMSE"].append(rmse(y[val], pred))
        scores["MAE"].append(mean_absolute_error(y[val], pred))
        scores["R2"].append(r2_score(y[val], pred))

    return {"RMSE": np.mean(scores["RMSE"]), "RMSE std": np.std(scores["RMSE"]),
            "MAE": np.mean(scores["MAE"]), "R2": np.mean(scores["R2"]),
            "обучение, с": np.mean(scores["время"])}


def main():
    # 1. Данные
    raw = load()
    print(f"\n[1] Датасет: {raw.shape[0]} объектов, {raw.shape[1]} колонок")
    print(describe(raw).to_string())

    X_df, y_series = preprocess(raw)
    features = list(X_df.columns)
    X, y = X_df.values, y_series.values
    print(f"\nПосле предобработки: {X.shape[0]} объектов, {X.shape[1]} признаков")
    print(f"Цена (тыс. $): среднее {y.mean():.1f}, медиана {np.median(y):.1f}, std {y.std():.1f}")
    plots.target_distribution(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"train: {len(y_train)}, test: {len(y_test)}")

    # 2. Кросс-валидация
    print(f"\n[2] 5-фолдовая CV, {N_TREES} деревьев, lr={LR}, глубина {DEPTH}")
    cv = KFold(5, shuffle=True, random_state=SEED)
    candidates = {
        "своя: GradientBoosting": lambda: GradientBoosting(
            n_estimators=N_TREES, learning_rate=LR, max_depth=DEPTH, random_state=SEED),
        "своя: GB + subsample=0.8": lambda: GradientBoosting(
            n_estimators=N_TREES, learning_rate=LR, max_depth=DEPTH,
            subsample=0.8, random_state=SEED),
        "эталон: sklearn GradientBoosting": lambda: GradientBoostingRegressor(
            n_estimators=N_TREES, learning_rate=LR, max_depth=DEPTH, random_state=SEED),
        "эталон: sklearn HistGradientBoosting": lambda: HistGradientBoostingRegressor(
            max_iter=N_TREES, learning_rate=LR, max_depth=DEPTH, random_state=SEED),
        "baseline: одно дерево": lambda: DecisionTreeRegressor(
            max_depth=DEPTH, random_state=SEED),
        "baseline: RandomForest": lambda: RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=SEED),
    }

    rows = {}
    for name, factory in candidates.items():
        print(f"  считаю {name} ...", flush=True)
        rows[name] = cross_validate(factory, X_train, y_train, cv)
    table = pd.DataFrame(rows).T
    print()
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    plots.comparison(table)

    # 3. Финальные модели и кривые обучения
    print("\n[3] Обучение на всей выборке и качество на тесте")
    own = GradientBoosting(n_estimators=N_TREES, learning_rate=LR,
                           max_depth=DEPTH, random_state=SEED)
    start = time.perf_counter(); own.fit(X_train, y_train)
    own_time = time.perf_counter() - start

    reference = GradientBoostingRegressor(n_estimators=N_TREES, learning_rate=LR,
                                          max_depth=DEPTH, random_state=SEED)
    start = time.perf_counter(); reference.fit(X_train, y_train)
    reference_time = time.perf_counter() - start

    print(pd.DataFrame([
        {"модель": name, "RMSE": rmse(y_test, model.predict(X_test)),
         "MAE": mean_absolute_error(y_test, model.predict(X_test)),
         "R2": r2_score(y_test, model.predict(X_test)), "обучение, с": elapsed}
        for name, model, elapsed in (("своя реализация", own, own_time),
                                     ("sklearn", reference, reference_time))
    ]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    own_train_curve = [rmse(y_train, p) for p in own.staged_predict(X_train)]
    own_test_curve = [rmse(y_test, p) for p in own.staged_predict(X_test)]
    reference_curve = [rmse(y_test, p) for p in reference.staged_predict(X_test)]
    plots.learning_curves(np.arange(1, N_TREES + 1), own_train_curve,
                          own_test_curve, reference_curve)

    checkpoints = [1, 10, 25, 50, 100, 150, 200]
    print("\nКривая обучения:")
    print(pd.DataFrame({
        "деревьев": checkpoints,
        "своя train": [own_train_curve[i - 1] for i in checkpoints],
        "своя test": [own_test_curve[i - 1] for i in checkpoints],
        "sklearn test": [reference_curve[i - 1] for i in checkpoints],
    }).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # 4. Гиперпараметры
    print("\n[4] Влияние learning_rate и глубины")
    lr_curves = {}
    for lr in [0.01, 0.05, 0.1, 0.3, 0.5]:
        model = GradientBoosting(n_estimators=N_TREES, learning_rate=lr,
                                 max_depth=DEPTH, random_state=SEED).fit(X_train, y_train)
        lr_curves[lr] = [rmse(y_test, p) for p in model.staged_predict(X_test)]
    print(pd.DataFrame({
        "learning_rate": list(lr_curves),
        f"test RMSE @{N_TREES}": [c[-1] for c in lr_curves.values()],
        "лучший RMSE": [min(c) for c in lr_curves.values()],
        "деревьев до лучшего": [int(np.argmin(c) + 1) for c in lr_curves.values()],
    }).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    depths = [1, 2, 3, 4, 6, 8]
    depth_scores = []
    for depth in depths:
        model = GradientBoosting(n_estimators=N_TREES, learning_rate=LR,
                                 max_depth=depth, random_state=SEED).fit(X_train, y_train)
        depth_scores.append((rmse(y_train, model.predict(X_train)),
                             rmse(y_test, model.predict(X_test))))
    print()
    print(pd.DataFrame({"max_depth": depths,
                        "train RMSE": [s[0] for s in depth_scores],
                        "test RMSE": [s[1] for s in depth_scores]}).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    plots.hyperparameters(lr_curves, depths, depth_scores, N_TREES)

    # 5. Важности
    print("\n[5] Важность признаков")
    importances = pd.DataFrame({"своя реализация": own.feature_importances_,
                                "sklearn": reference.feature_importances_},
                               index=features).sort_values("своя реализация", ascending=False)
    print(importances.to_string(float_format=lambda v: f"{v:.4f}"))
    plots.importances(importances)

    # 6. Классификация
    print("\n[6] Бустинг с логистической потерей")
    print("Задача: дом дороже медианы?")
    y_binary = (y > np.median(y)).astype(int)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=SEED, stratify=y_binary)

    rows = []
    for name, model in (
        ("своя: GB (log_loss)", GradientBoosting(n_estimators=N_TREES, learning_rate=LR,
                                                 max_depth=DEPTH, loss="log_loss",
                                                 random_state=SEED)),
        ("эталон: sklearn GBClassifier", GradientBoostingClassifier(
            n_estimators=N_TREES, learning_rate=LR, max_depth=DEPTH, random_state=SEED)),
    ):
        start = time.perf_counter()
        model.fit(Xc_train, yc_train)
        rows.append({"модель": name,
                     "accuracy": accuracy_score(yc_test, model.predict(Xc_test)),
                     "ROC-AUC": roc_auc_score(yc_test, model.predict_proba(Xc_test)[:, 1]),
                     "обучение, с": time.perf_counter() - start})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    accuracies = [
        accuracy_score(yc_train[val], GradientBoosting(
            n_estimators=N_TREES, learning_rate=LR, max_depth=DEPTH, loss="log_loss",
            random_state=SEED).fit(Xc_train[tr], yc_train[tr]).predict(Xc_train[val]))
        for tr, val in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xc_train, yc_train)
    ]
    print(f"\nСвоя реализация, accuracy на 5-фолдовой CV: "
          f"{np.mean(accuracies):.4f} +- {np.std(accuracies):.4f}")
    print(f"\nГрафики: {plots.IMAGES}")


if __name__ == "__main__":
    main()
