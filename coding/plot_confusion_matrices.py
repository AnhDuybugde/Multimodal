from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


MULTICLASS_SOURCE_LABELS = ("ambient", "leaf", "trunk", "twig")
MULTICLASS_PLOT_LABELS = ("leaf", "twig", "trunk", "ambient")
BINARY_PLOT_LABELS = ("contact", "non-contact")
DEFAULT_MODES = ("ast", "audio", "clap", "fusion", "video")


def load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def reorder_matrix(cm: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    source_labels = MULTICLASS_SOURCE_LABELS
    plot_labels = MULTICLASS_PLOT_LABELS
    order = [source_labels.index(label) for label in plot_labels]
    return cm[np.ix_(order, order)], plot_labels


def collapse_to_binary(cm: np.ndarray) -> np.ndarray:
    """Collapse multiclass matrix to rows/cols: contact, non-contact."""
    non_contact_true = cm[0]
    contact_true = cm[1:].sum(axis=0)
    collapsed = np.array(
        [
            [contact_true[1:].sum(), contact_true[0]],
            [non_contact_true[1:].sum(), non_contact_true[0]],
        ],
        dtype=int,
    )
    return collapsed


def binary_metrics_from_matrix(cm: np.ndarray) -> tuple[float, float, float, float]:
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / cm.sum() if cm.sum() else 0.0
    return f1, precision, recall, accuracy


def annotate_matrix(cm: np.ndarray) -> np.ndarray:
    row_totals = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)
    return np.array(
        [
            [f"{count}\n{percent:.1%}" for count, percent in zip(count_row, pct_row)]
            for count_row, pct_row in zip(cm, pct)
        ]
    )


def plot_matrix(ax, result: dict, task: str = "multiclass", title_prefix: str = "") -> None:
    metrics = result["best_val_metrics"]
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    if task == "binary":
        cm = collapse_to_binary(cm)
        labels = BINARY_PLOT_LABELS
    else:
        cm, labels = reorder_matrix(cm)
    annotations = annotate_matrix(cm)

    sns.heatmap(
        cm,
        ax=ax,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=0.6,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"fontsize": 9},
    )
    mode = result.get("mode", "model").upper()
    if task == "binary":
        f1, precision, recall, accuracy = binary_metrics_from_matrix(cm)
        ax.set_title(
            f"{title_prefix}{mode} Binary Contact\nF1={f1:.3f}  P={precision:.3f}  R={recall:.3f}  Acc={accuracy:.3f}",
            fontsize=11,
            pad=10,
        )
    else:
        macro_f1 = metrics["macro_f1"]
        accuracy = metrics["accuracy"]
        contact_f1 = metrics["binary_contact_f1"]
        ax.set_title(
            f"{title_prefix}{mode} Multiclass\nMacro F1={macro_f1:.3f}  Acc={accuracy:.3f}  Contact F1={contact_f1:.3f}",
            fontsize=11,
            pad=10,
        )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)


def save_single(result: dict, output_dir: Path, task: str = "multiclass") -> Path:
    fig_size = (5.2, 4.6) if task == "binary" else (6.2, 5.2)
    fig, ax = plt.subplots(figsize=fig_size)
    plot_matrix(ax, result, task=task)
    fig.tight_layout()
    out_path = output_dir / f"confusion_{result['mode']}_{task}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_combined(results: list[dict], output_dir: Path, task: str = "multiclass") -> Path:
    order = {mode: index for index, mode in enumerate(DEFAULT_MODES)}
    results = sorted(results, key=lambda r: order.get(r["mode"], len(order)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    for ax, result in zip(axes.flat, results):
        plot_matrix(ax, result, task=task)
    for ax in axes.flat[len(results) :]:
        ax.axis("off")
    task_title = "Binary Contact" if task == "binary" else "Multiclass"
    fig.suptitle(f"{task_title} Confusion Matrices on Robot Test Split", fontsize=16, y=1.02)
    fig.tight_layout()
    suffix = "binary_all" if task == "binary" else "matrices_all"
    out_path = output_dir / f"confusion_{suffix}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot paper-style confusion matrices from saved JSON result files."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("output"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--tasks", nargs="+", choices=("multiclass", "binary"), default=["multiclass", "binary"])
    args = parser.parse_args()

    json_files = [args.input_dir / f"{mode}.json" for mode in args.modes]
    missing = [path for path in json_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing JSON result files: " + ", ".join(map(str, missing)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [load_result(path) for path in json_files]
    written = []
    for task in args.tasks:
        written.extend(save_single(result, args.output_dir, task=task) for result in results)
        written.append(save_combined(results, args.output_dir, task=task))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
