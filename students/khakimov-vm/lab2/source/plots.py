from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

IMAGES = Path(__file__).resolve().parent.parent / "images"
IMAGES.mkdir(exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(IMAGES / name, dpi=130)
    plt.close(fig)


def class_balance(counts):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values, color="#4f81bd")
    for i, value in enumerate(counts.values):
        ax.text(i, value + 30, str(value), ha="center", fontsize=9)
    ax.set_xlabel("Класс физической подготовки")
    ax.set_ylabel("Число объектов")
    ax.set_title("Баланс классов, Body Performance")
    _save(fig, "class_balance.png")


def oob_convergence(n_values, curves):
    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, scores), color in zip(curves.items(), ["#4f81bd", "#c0504d"]):
        ax.plot(n_values, scores, "o-", color=color, label=name)
    ax.set_xlabel("Число базовых деревьев")
    ax.set_ylabel("OOB accuracy")
    ax.set_title("Сходимость OOB-оценки по числу деревьев")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, "oob_convergence.png")


def importances(table):
    df = table.sort_values("OOB^j")
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y - 0.2, df["OOB^j"], height=0.4, color="#4f81bd", label="OOB$^j$ (своя)")
    ax.barh(y + 0.2, df["Gini (sklearn)"], height=0.4, color="#9bbb59", label="MDI Gini (sklearn)")
    ax.set_yticks(y); ax.set_yticklabels(df.index)
    ax.set_xlabel("Важность признака")
    ax.set_title("Важность признаков: OOB-перестановки против встроенного Gini")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    _save(fig, "feature_importances.png")


def comparison(results):
    colors = ["#4f81bd", "#c0504d", "#9bbb59", "#8064a2", "#f79646", "#4bacc6", "#bfbfbf"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].barh(results["модель"], results["accuracy"], color=colors[:len(results)])
    for i, value in enumerate(results["accuracy"]):
        axes[0].text(value + 0.003, i, f"{value:.4f}", va="center", fontsize=8)
    axes[0].set_xlim(results["accuracy"].min() - 0.03, results["accuracy"].max() + 0.02)
    axes[0].set_xlabel("Accuracy на тесте")
    axes[0].set_title("Качество")

    axes[1].barh(results["модель"], results["обучение, с"], color=colors[:len(results)])
    for i, value in enumerate(results["обучение, с"]):
        axes[1].text(value * 1.02, i, f"{value:.2f} с", va="center", fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Время обучения, с (log)")
    axes[1].set_title("Скорость обучения")
    axes[1].set_yticklabels([])

    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Собственные ансамбли против эталонных реализаций sklearn")
    _save(fig, "models_comparison.png")


def confusion(model, X, y, classes):
    cm = confusion_matrix(y, model.predict(X))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=11,
                color="white" if value > cm.max() / 2 else "black")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Предсказание"); ax.set_ylabel("Истина")
    ax.set_title("Матрица ошибок, свой Random Forest")
    _save(fig, "confusion_matrix.png")
