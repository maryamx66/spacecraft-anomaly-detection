from __future__ import annotations
import os
import sys
import json
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Route the output back out to the main directory's evaluation_graphs folder
OUTPUT_DIR = os.path.join(parent_dir, "evaluation_graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class Metrics:
    variant_name: str;  accuracy: float;  precision: float
    recall: float;      f1_score: float
    tp: int; fp: int;   tn: int;  fn: int
    best_val_loss: float = 0.0

    @property
    def confusion_matrix(self):
        return np.array([[self.tn, self.fp], [self.fn, self.tp]])


def load_metrics(path=os.path.join(OUTPUT_DIR, "evaluation_summary.json")):
    if not os.path.exists(path):
        print("[ERROR] evaluation_summary.json not found. Run evaluation.py first.")
        exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Metrics(d["variant_name"], d["accuracy"], d["precision"],
                    d["recall"], d["f1_score"],
                    d["tp"], d["fp"], d["tn"], d["fn"],
                    d.get("best_val_loss", 0.0)) for d in data]


_LABELS = ["Normal (0)", "Anomaly (1)"]


def _draw(ax, cm, title, normalize=False, cmap="Blues", fs=11):
    disp = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) if normalize else cm
    fmt  = ".2%" if normalize else "d"
    ax.imshow(disp, interpolation="nearest", cmap=cmap, vmin=0, vmax=(1.0 if normalize else None))
    ax.set_title(title, fontsize=fs, fontweight="bold", pad=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(_LABELS, fontsize=fs-1)
    ax.set_yticks([0, 1]); ax.set_yticklabels(_LABELS, fontsize=fs-1, rotation=45, ha="right")
    ax.set_xlabel("Predicted Class", fontsize=fs-1)
    ax.set_ylabel("Actual Class",    fontsize=fs-1)
    thresh = disp.max() / 2
    for i, j in itertools.product(range(2), range(2)):
        v = disp[i, j]
        ax.text(j, i, format(v, fmt) if normalize else str(int(v)),
                ha="center", va="center", fontsize=fs+1,
                color="white" if v > thresh else "black", fontweight="bold")


def plot_grid(metrics, normalize=False):
    n = len(metrics); ncols = min(n, 3); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5.5, nrows*4.8), squeeze=False)
    axes = axes.flatten()
    sorted_m = sorted(metrics, key=lambda x: x.f1_score, reverse=True)
    for ax, m in zip(axes, sorted_m):
        cmap  = "Oranges" if m == sorted_m[0] else "Blues"
        title = f"{m.variant_name.replace('_', chr(10))}\nF1={m.f1_score:.3f}"
        _draw(ax, m.confusion_matrix, title, normalize=normalize, cmap=cmap, fs=9)
    for ax in axes[len(sorted_m):]:
        ax.set_visible(False)
    tag = "(Normalized %)" if normalize else "(Raw Counts)"
    fig.suptitle(f"Ablation Study — Confusion Matrices {tag}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"confusion_matrix_grid_{'norm' if normalize else 'raw'}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


def plot_best_detail(metrics):
    m = max(metrics, key=lambda x: x.f1_score)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    _draw(ax1, m.confusion_matrix, f"{m.variant_name}\nRaw Counts",    normalize=False)
    _draw(ax2, m.confusion_matrix, f"{m.variant_name}\nNormalized (%)", normalize=True, cmap="Greens")
    fig.suptitle(f"Best Model Detail  |  F1={m.f1_score:.3f}  "
                 f"Acc={m.accuracy:.3f}  Prec={m.precision:.3f}  Rec={m.recall:.3f}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"cm_best_model_{m.variant_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


def plot_metrics_bar(metrics):
    sorted_m = sorted(metrics, key=lambda x: x.f1_score, reverse=True)
    names = [m.variant_name.replace("_", "\n") for m in sorted_m]
    x = np.arange(len(sorted_m)); w = 0.2
    fig, ax = plt.subplots(figsize=(max(10, len(sorted_m)*2), 5))
    ax.bar(x-1.5*w, [m.accuracy  for m in sorted_m], w, label="Accuracy",  color="#6DB33F", alpha=0.85)
    ax.bar(x-0.5*w, [m.precision for m in sorted_m], w, label="Precision", color="#4A90D9", alpha=0.85)
    ax.bar(x+0.5*w, [m.recall    for m in sorted_m], w, label="Recall",    color="#9B59B6", alpha=0.85)
    ax.bar(x+1.5*w, [m.f1_score  for m in sorted_m], w, label="F1-Score",  color="#E87722", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Ablation Study — All Metrics Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3)
    for i, (a, p, r, f) in enumerate(zip(
            [m.accuracy  for m in sorted_m], [m.precision for m in sorted_m],
            [m.recall    for m in sorted_m], [m.f1_score  for m in sorted_m])):
        for dx, v in zip([-1.5*w, -0.5*w, 0.5*w, 1.5*w], [a, p, r, f]):
            ax.text(i+dx, v+0.02, f"{v:.2f}", ha="center", fontsize=7,
                    fontweight="bold", rotation=90)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ablation_all_metrics_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


def print_report(metrics):
    sep = "-" * 58
    print(f"\n{'='*58}\n  CONFUSION MATRIX REPORT\n{'='*58}")
    for m in sorted(metrics, key=lambda x: x.f1_score, reverse=True):
        print(f"\n  {m.variant_name}")
        print(sep)
        print(f"  {'':18} Predicted Normal   Predicted Anomaly")
        print(f"  Actual Normal   {m.tn:>16,d}   {m.fp:>17,d}")
        print(f"  Actual Anomaly  {m.fn:>16,d}   {m.tp:>17,d}")
        print(sep)
        print(f"  Accuracy : {m.accuracy:.4f}  |  Precision : {m.precision:.4f}")
        print(f"  Recall   : {m.recall:.4f}  |  F1-Score  : {m.f1_score:.4f}")
    best = max(metrics, key=lambda x: x.f1_score)
    print(f"\n  * Best Model: {best.variant_name}  (F1={best.f1_score:.4f})")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    metrics = load_metrics()
    print(f"[INFO] {len(metrics)} models loaded.\n")
    plot_grid(metrics, normalize=False)
    plot_grid(metrics, normalize=True)
    plot_best_detail(metrics)
    plot_metrics_bar(metrics)
    print_report(metrics)
    print(f"[DONE] All plots saved to '{OUTPUT_DIR}/'")
