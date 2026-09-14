from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

IMAGES = Path(__file__).resolve().parent.parent / "images"
IMAGES.mkdir(exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(IMAGES / name, dpi=130)
    plt.close(fig)


def em_convergence(curves):
    fig, ax = plt.subplots(figsize=(8, 5))
    for (label, history), color in zip(curves.items(),
                                       ["#4f81bd", "#c0504d", "#9bbb59", "#8064a2"]):
        ax.plot(np.arange(1, len(history) + 1), history, color=color, label=label)
    ax.set_xlabel("Итерация EM")
    ax.set_ylabel("Среднее log-правдоподобие на объект")
    ax.set_title("Сходимость EM: правдоподобие монотонно не убывает")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, "em_convergence.png")


def model_selection(table):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for cov_type, group in table.groupby("covariance_type"):
        axes[0].plot(group["K"], group["BIC"], "o-", label=cov_type)
        axes[1].plot(group["K"], group["test log-lik"], "o-", label=cov_type)

    axes[0].set_ylabel("BIC (меньше — лучше)")
    axes[0].set_title("Выбор модели по BIC")
    axes[1].set_ylabel("log-правдоподобие на тесте (больше — лучше)")
    axes[1].set_title("Правдоподобие на отложенной выборке")
    for ax in axes:
        ax.set_xlabel("Число компонент K")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Подбор числа компонент и структуры ковариаций")
    _save(fig, "model_selection.png")


def clusters_pca(X, labels_own, labels_reference, quality, seed=42):
    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(X)
    K = int(max(labels_own.max(), labels_reference.max())) + 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [("Своя GMM (EM)", labels_own, True),
              ("sklearn GaussianMixture", labels_reference, True),
              ("Истинная оценка качества", quality, False)]

    for ax, (title, colors, discrete) in zip(axes, panels):
        if discrete:
            # Номера компонент дискретны: непрерывная шкала с делениями 0.5, 1.5
            # вводила бы в заблуждение
            scatter = ax.scatter(Z[:, 0], Z[:, 1], c=colors, s=8, alpha=0.7,
                                 cmap=plt.get_cmap("tab10", K), vmin=-0.5, vmax=K - 0.5)
            bar = fig.colorbar(scatter, ax=ax, shrink=0.85, ticks=np.arange(K))
            bar.set_label("номер компоненты")
        else:
            scatter = ax.scatter(Z[:, 0], Z[:, 1], c=colors, s=8, alpha=0.7, cmap="viridis")
            bar = fig.colorbar(scatter, ax=ax, shrink=0.85, ticks=np.unique(colors))
            bar.set_label("оценка эксперта")
        ax.set_title(title)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    fig.suptitle(f"Проекция на 2 главные компоненты "
                 f"(объясняют {pca.explained_variance_ratio_.sum():.0%} дисперсии)")
    _save(fig, "clusters_pca.png")


def density_1d(X, model, feature_names, index):
    """Восстановленная маргинальная плотность против гистограммы данных."""
    values = X[:, index]
    grid = np.linspace(values.min() - 0.5, values.max() + 0.5, 400)

    density = np.zeros_like(grid)
    for k in range(model.n_components):
        mean = model.means_[k, index]
        if model.covariance_type == "full":
            variance = model.covariances_[k][index, index]
        elif model.covariance_type == "diag":
            variance = model.covariances_[k][index]
        else:
            variance = model.covariances_[k]
        density += model.weights_[k] * (np.exp(-0.5 * (grid - mean) ** 2 / variance)
                                        / np.sqrt(2 * np.pi * variance))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=50, density=True, color="#d9d9d9", edgecolor="white",
            label="гистограмма данных")
    ax.plot(grid, density, color="#c0504d", linewidth=2,
            label=f"смесь из {model.n_components} компонент")
    ax.set_xlabel(f"{feature_names[index]} (стандартизованный)")
    ax.set_ylabel("Плотность")
    ax.set_title("Восстановленная плотность распределения признака")
    ax.legend()
    _save(fig, "density_1d.png")
