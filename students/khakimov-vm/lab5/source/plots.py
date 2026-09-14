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


def dataset_stats(ratings):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].hist(ratings["rating"], bins=np.arange(0.25, 5.5, 0.5),
                 color="#4f81bd", edgecolor="white")
    axes[0].set_xlabel("Оценка"); axes[0].set_ylabel("Количество")
    axes[0].set_title("Распределение оценок")

    axes[1].hist(ratings.groupby("userId").size(), bins=50, color="#9bbb59", edgecolor="white")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Оценок у пользователя"); axes[1].set_ylabel("Пользователей (log)")
    axes[1].set_title("Активность пользователей")

    per_item = ratings.groupby("movieId").size().sort_values(ascending=False)
    axes[2].plot(np.arange(1, len(per_item) + 1), per_item.values, color="#c0504d")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel("Ранг фильма (log)"); axes[2].set_ylabel("Оценок (log)")
    axes[2].set_title("«Длинный хвост» популярности")

    fig.suptitle("MovieLens: структура данных")
    _save(fig, "dataset_stats.png")


def slim_regularization(table):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(table["l1_reg"], table["RMSE"], "o-", color="#4f81bd")
    axes[0].set_ylabel("RMSE на тесте")
    axes[0].set_title("Качество в зависимости от силы L1")

    axes[1].plot(table["l1_reg"], table["разреженность W, %"], "o-", color="#c0504d")
    axes[1].set_ylabel("Доля нулей в W, %")
    axes[1].set_title("Разреженность матрицы весов")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("λ (коэффициент L1)")
        ax.grid(alpha=0.3)

    fig.suptitle("SLIM: компромисс между точностью и разреженностью")
    _save(fig, "slim_regularization.png")


def als_factors(table, history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(table["n_factors"], table["RMSE"], "o-", color="#4f81bd", label="test")
    axes[0].plot(table["n_factors"], table["train RMSE"], "o--", color="#9bbb59", label="train")
    axes[0].set_xlabel("Число латентных факторов f"); axes[0].set_ylabel("RMSE")
    axes[0].set_title("Влияние размерности скрытого пространства")
    axes[0].legend()

    axes[1].plot(np.arange(1, len(history) + 1), history, "o-", color="#c0504d")
    axes[1].set_xlabel("Итерация ALS"); axes[1].set_ylabel("Train RMSE")
    axes[1].set_title("Сходимость ALS")

    for ax in axes:
        ax.grid(alpha=0.3)
    _save(fig, "als_factors.png")


def comparison(results):
    colors = ["#bfbfbf", "#4f81bd", "#8db4e2", "#c0504d", "#e59a98"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    panels = [("RMSE", "RMSE (меньше — лучше)"),
              ("NDCG@10", "NDCG@10 (больше — лучше)"),
              ("обучение, с", "Время обучения, с (log)")]

    for ax, (column, title) in zip(axes, panels):
        ax.barh(results["модель"], results[column], color=colors[:len(results)])
        for i, value in enumerate(results[column]):
            if column == "обучение, с":
                label = f" {value:.3f} с" if value < 1 else f" {value:.1f} с"
            else:
                label = f" {value:.4f}"
            ax.text(value, i, label, va="center", fontsize=8)

        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        if column == "обучение, с":
            ax.set_xscale("log")
        else:
            span = results[column].max() - results[column].min()
            ax.set_xlim(results[column].min() - span * 0.35,
                        results[column].max() + span * 0.35)
        if ax is not axes[0]:
            ax.set_yticklabels([])

    fig.suptitle("Сравнение рекомендательных моделей на MovieLens")
    _save(fig, "models_comparison.png")
