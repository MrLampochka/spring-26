from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

IMAGES = Path(__file__).resolve().parent.parent / "images"
IMAGES.mkdir(exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(IMAGES / name, dpi=130)
    plt.close(fig)


def target_distribution(y):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y, bins=60, color="#4f81bd", edgecolor="white")
    ax.axvline(np.median(y), color="#c0504d", linestyle="--",
               label=f"медиана = {np.median(y):.1f} тыс. $")
    ax.set_xlabel("Медианная стоимость дома, тыс. $")
    ax.set_ylabel("Число округов")
    ax.set_title("Распределение целевой переменной, California Housing")
    ax.legend()
    _save(fig, "target_distribution.png")


def learning_curves(steps, own_train, own_test, reference_test):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(steps, own_train, ":", color="#4f81bd", linewidth=1.6, label="своя, train")
    # Тестовые кривые совпадают: свою рисуем толстой, эталон — пунктиром поверх
    ax.plot(steps, own_test, "-", color="#4f81bd", linewidth=3.5, label="своя, test")
    ax.plot(steps, reference_test, "--", color="#9bbb59", linewidth=1.6,
            label="sklearn, test (совпадает)")
    ax.set_xlabel("Число деревьев в композиции")
    ax.set_ylabel("RMSE, тыс. $")
    ax.set_title("Кривые обучения градиентного бустинга")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, "learning_curves.png")


def hyperparameters(lr_curves, depths, depth_scores, n_trees):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for lr, curve in lr_curves.items():
        axes[0].plot(np.arange(1, len(curve) + 1), curve, label=f"lr = {lr}")
    axes[0].set_xlabel("Число деревьев"); axes[0].set_ylabel("Test RMSE, тыс. $")
    axes[0].set_title("Влияние learning_rate")

    axes[1].plot(depths, [s[0] for s in depth_scores], "o--", color="#4f81bd", label="train")
    axes[1].plot(depths, [s[1] for s in depth_scores], "o-", color="#c0504d", label="test")
    axes[1].set_xlabel("Глубина базового дерева"); axes[1].set_ylabel("RMSE, тыс. $")
    axes[1].set_title(f"Влияние глубины ({n_trees} деревьев)")

    for ax in axes:
        ax.grid(alpha=0.3); ax.legend()
    _save(fig, "hyperparameters.png")


def comparison(results):
    colors = ["#4f81bd", "#9bbb59", "#8064a2", "#f79646", "#c0504d", "#4bacc6"]
    labels = list(results.index)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].barh(labels, results["RMSE"], xerr=results["RMSE std"], color=colors[:len(labels)])
    for i, value in enumerate(results["RMSE"]):
        axes[0].text(value + 0.4, i, f"{value:.2f}", va="center", fontsize=8)
    axes[0].set_xlabel("RMSE на кросс-валидации, тыс. $ (меньше — лучше)")
    axes[0].set_title("Качество")

    axes[1].barh(labels, results["обучение, с"], color=colors[:len(labels)])
    for i, value in enumerate(results["обучение, с"]):
        axes[1].text(value * 1.05, i, f"{value:.2f} с", va="center", fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Среднее время обучения на фолде, с (log)")
    axes[1].set_title("Скорость обучения")
    axes[1].set_yticklabels([])

    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Собственный градиентный бустинг против эталонных реализаций")
    _save(fig, "models_comparison.png")


def importances(table):
    df = table.sort_values("своя реализация")
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y - 0.2, df["своя реализация"], height=0.4, color="#4f81bd", label="своя")
    ax.barh(y + 0.2, df["sklearn"], height=0.4, color="#9bbb59", label="sklearn")
    ax.set_yticks(y); ax.set_yticklabels(df.index)
    ax.set_xlabel("Важность признака")
    ax.set_title("Важность признаков в градиентном бустинге")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    _save(fig, "feature_importances.png")
