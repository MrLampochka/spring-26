from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

IMAGES = Path(__file__).resolve().parent.parent / "images"
IMAGES.mkdir(exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(IMAGES / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def missing_values(raw):
    share = (raw.isna().mean() * 100).sort_values()
    share = share[share > 0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(share.index, share.values, color="#c0504d")
    for i, value in enumerate(share.values):
        ax.text(value + 0.7, i, f"{value:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, share.max() * 1.18)
    ax.set_xlabel("Доля пропусков, %")
    ax.set_title("Пропущенные значения в датасете Titanic")
    _save(fig, "missing_values.png")


def depth_curve(depths, own_train, own_test, reference_test):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, own_train, "o--", color="#4f81bd", label="своя реализация, train")
    ax.plot(depths, own_test, "o-", color="#4f81bd", label="своя реализация, test")
    ax.plot(depths, reference_test, "s-", color="#9bbb59", label="sklearn, test")
    ax.set_xlabel("Максимальная глубина дерева")
    ax.set_ylabel("F1-score")
    ax.set_title("Переобучение дерева при росте глубины")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, "depth_curve.png")


def models_comparison(results):
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)
    colors = ["#4f81bd", "#c0504d", "#9bbb59", "#8064a2"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (_, row) in enumerate(results.iterrows()):
        values = [row[m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=row["модель"],
                      color=colors[i % len(colors)])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008,
                    f"{value:.3f}", ha="center", fontsize=7)

    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение реализаций решающего дерева (Titanic, тест)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "models_comparison.png")


def pruning_effect(before, after, X_test, y_test):
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (title, model) in zip(axes, [("До редукции", before), ("После редукции", after)]):
        cm = confusion_matrix(y_test, model.predict(X_test))
        ax.imshow(cm, cmap="Blues")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=12,
                    color="white" if value > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["погиб", "выжил"]); ax.set_yticklabels(["погиб", "выжил"])
        ax.set_xlabel("Предсказание"); ax.set_ylabel("Истина")
        ax.set_title(f"{title}\nлистьев: {model.n_leaves()}, глубина: {model.depth()}")
    _save(fig, "pruning_confusion.png")


def tree_structure(model, feature_names, max_depth=3):
    """Схема дерева до заданной глубины; на рёбрах подписаны веса q."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.08)
    ax.set_title(f"Решающее дерево (первые {max_depth} уровней), критерий Джини", fontsize=13)

    def draw(node, x, y, span, depth):
        cut = not node.is_leaf and depth >= max_depth
        if node.is_leaf or cut:
            text = ("..." if cut else
                    f"класс={np.argmax(node.proba)}\n"
                    + ", ".join(f"P({i})={p:.2f}" for i, p in enumerate(node.proba))
                    + f"\nn={node.n:.0f}")
            color = "#e0e0e0" if cut else "#c9e7c9"
        else:
            name = (feature_names[node.feature] if node.feature < len(feature_names)
                    else f"X[{node.feature}]")
            text = f"{name} <= {node.threshold:.2f}\ngain={node.gain:.4f}\nn={node.n:.0f}"
            color = "#cfe2f3"

        ax.text(x, y, text, ha="center", va="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=color,
                          edgecolor="black", linewidth=0.7))
        if node.is_leaf or cut:
            return

        dy = 1.0 / (max_depth + 1)
        for child, cx, q in ((node.left, x - span / 2, node.q[0]),
                             (node.right, x + span / 2, node.q[1])):
            ax.plot([x, cx], [y - 0.02, y - dy + 0.02], color="black", linewidth=0.7, zorder=0)
            ax.text((x + cx) / 2, y - dy / 2, f"q={q:.2f}", fontsize=6.5, color="#555")
            draw(child, cx, y - dy, span / 2, depth + 1)

    draw(model.root, x=0.5, y=1.0, span=0.5, depth=0)
    _save(fig, "tree.png")
