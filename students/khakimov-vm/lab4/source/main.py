import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import plots
from dataset import binarize_quality, describe, load, preprocess
from gmm import GaussianMixtureEM, GMMBayesClassifier

warnings.filterwarnings("ignore")
SEED = 42


def timed(model, X):
    start = time.perf_counter()
    model.fit(X)
    return model, time.perf_counter() - start


def main():
    # 1. Данные
    raw = load()
    print(f"\n[1] Датасет: {raw.shape[0]} объектов, {raw.shape[1]} колонок")
    print(describe(raw).to_string())

    X, quality, features = preprocess(raw)
    print(f"\nПосле дедупликации и стандартизации: {X.shape[0]} объектов, "
          f"{X.shape[1]} признаков")
    print(f"Оценки качества: {dict(pd.Series(quality).value_counts().sort_index())}")

    X_train, X_test = train_test_split(X, test_size=0.25, random_state=SEED)
    print(f"train: {len(X_train)}, test: {len(X_test)}")

    # 2. Сходимость EM
    print("\n[2] Сходимость EM-алгоритма")
    curves = {}
    for k in (2, 3, 5, 8):
        model = GaussianMixtureEM(n_components=k, n_init=1, reg_covar=1e-4,
                                  random_state=SEED).fit(X_train)
        curves[f"K = {k} ({model.n_iter_} итераций)"] = model.history_
    plots.em_convergence(curves)

    demo = GaussianMixtureEM(n_components=4, n_init=1, reg_covar=1e-4,
                             random_state=SEED).fit(X_train)
    print(f"K=4, full: сошёлся за {demo.n_iter_} итераций (converged={demo.converged_})")
    for i in (0, 1, 2, 4, 9, 19, len(demo.history_) - 1):
        if i < len(demo.history_):
            print(f"  итерация {i + 1:>3}: {demo.history_[i]: .4f}")
    print("Правдоподобие монотонно не убывает: "
          f"{all(b >= a - 1e-9 for a, b in zip(demo.history_, demo.history_[1:]))}")

    # 3. Выбор числа компонент
    print("\n[3] Выбор модели: правдоподобие, AIC, BIC")
    rows = []
    for cov_type in ("spherical", "diag", "full"):
        for k in (1, 2, 3, 4, 5, 6, 8, 10, 12):
            model = GaussianMixtureEM(n_components=k, covariance_type=cov_type,
                                      n_init=3, reg_covar=1e-4, random_state=SEED).fit(X_train)
            rows.append({"covariance_type": cov_type, "K": k,
                         "параметров": model.n_parameters(),
                         "train log-lik": model.score(X_train),
                         "test log-lik": model.score(X_test),
                         "AIC": model.aic(X_train), "BIC": model.bic(X_train)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    plots.model_selection(table)

    best = table.loc[table["BIC"].idxmin()]
    best_test = table.loc[table["test log-lik"].idxmax()]
    print(f"\nМинимум BIC: {best['covariance_type']}, K={int(best['K'])} "
          f"(BIC={best['BIC']:.1f}, test log-lik={best['test log-lik']:.4f})")
    print(f"Максимум правдоподобия на тесте: {best_test['covariance_type']}, "
          f"K={int(best_test['K'])} ({best_test['test log-lik']:.4f})")
    best_k, best_cov = int(best["K"]), str(best["covariance_type"])

    # 4. Сравнение с эталоном
    print("\n[4] Сравнение с sklearn.mixture.GaussianMixture")
    rows = []
    for cov_type in ("diag", "full"):
        for k in sorted({best_k, 3, 5}):
            own, own_time = timed(GaussianMixtureEM(
                n_components=k, covariance_type=cov_type, n_init=5,
                reg_covar=1e-4, random_state=SEED), X_train)
            reference, reference_time = timed(GaussianMixture(
                n_components=k, covariance_type=cov_type, n_init=5,
                reg_covar=1e-4, random_state=SEED), X_train)
            rows.append({"cov": cov_type, "K": k,
                         "своя train": own.score(X_train),
                         "sklearn train": reference.score(X_train),
                         "своя test": own.score(X_test),
                         "sklearn test": reference.score(X_test),
                         "своя BIC": own.bic(X_train), "sklearn BIC": reference.bic(X_train),
                         "своя, с": own_time, "sklearn, с": reference_time,
                         "итераций своя": own.n_iter_, "итераций sklearn": reference.n_iter_})
    comparison = pd.DataFrame(rows)
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nМаксимальное расхождение на тесте: "
          f"{np.max(np.abs(comparison['своя test'] - comparison['sklearn test'])):.5f}")

    # 5. Структура компонент
    print("\n[5] Структура найденных компонент")
    own = GaussianMixtureEM(n_components=best_k, covariance_type=best_cov,
                            n_init=10, reg_covar=1e-4, random_state=SEED).fit(X)
    reference = GaussianMixture(n_components=best_k, covariance_type=best_cov,
                                n_init=10, reg_covar=1e-4, random_state=SEED).fit(X)
    labels_own, labels_reference = own.predict(X), reference.predict(X)

    print(f"Веса (своя):    {np.round(np.sort(own.weights_)[::-1], 4)}")
    print(f"Веса (sklearn): {np.round(np.sort(reference.weights_)[::-1], 4)}")
    print(f"ARI своя vs sklearn: {adjusted_rand_score(labels_own, labels_reference):.4f}")
    print(f"ARI своя vs оценка качества: {adjusted_rand_score(quality, labels_own):.4f}")
    print("\nСоответствие компонент и экспертных оценок:")
    print(pd.crosstab(pd.Series(labels_own, name="компонента"),
                      pd.Series(quality, name="качество")).to_string())

    plots.clusters_pca(X, labels_own, labels_reference, quality, seed=SEED)
    plots.density_1d(X, own, features, features.index("alcohol"))

    # 6. GMM-байес против наивного байеса
    print("\n[6] GMM-байес против наивного байеса")
    print("Задача: отличить хорошее вино (quality >= 6) от остального")
    y = binarize_quality(quality)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y)
    print(f"баланс классов: {dict(pd.Series(y).value_counts())}")

    rows = []
    for n_components in (1, 2, 3, 5):
        start = time.perf_counter()
        model = GMMBayesClassifier(n_components=n_components, reg_covar=1e-3,
                                   n_init=3, random_state=SEED).fit(Xc_train, yc_train)
        rows.append({"модель": f"GMM-байес, {n_components} комп./класс",
                     "accuracy": accuracy_score(yc_test, model.predict(Xc_test)),
                     "f1": f1_score(yc_test, model.predict(Xc_test)),
                     "ROC-AUC": roc_auc_score(yc_test, model.predict_proba(Xc_test)[:, 1]),
                     "время, с": time.perf_counter() - start})

    start = time.perf_counter()
    nb = GaussianNB().fit(Xc_train, yc_train)
    rows.append({"модель": "эталон: sklearn GaussianNB",
                 "accuracy": accuracy_score(yc_test, nb.predict(Xc_test)),
                 "f1": f1_score(yc_test, nb.predict(Xc_test)),
                 "ROC-AUC": roc_auc_score(yc_test, nb.predict_proba(Xc_test)[:, 1]),
                 "время, с": time.perf_counter() - start})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nГрафики: {plots.IMAGES}")


if __name__ == "__main__":
    main()
