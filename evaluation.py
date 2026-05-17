from __future__ import annotations
print("demo")
import os
import json
import copy
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from preprocessing import PreprocessConfig
from data_loader import build_telemanom_pipeline
from model import TimeSeriesAutoencoder
print("demo")
from metrics import (
    compute_reconstruction_errors,
    aggregate_errors_to_scalar,
    nonparametric_dynamic_threshold,
    moving_average_threshold,
    compute_classification_metrics,
    channel_wise_errors,
    error_by_anomaly_class,
    ClassificationMetrics,
)
print("demo")

SAVED_MODELS_DIR  = "saved_models"
OUTPUT_DIR        = "evaluation_results"
DEVICE            = torch.device(
    "cuda"  if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[INFO] Evaluation device: {DEVICE}")

print("[INFO] Preparing data pipeline...")

config = PreprocessConfig(
    sequence_length=50,
    stride=10,
    val_size=0.15,
    random_state=42,
)

result = build_telemanom_pipeline(config=config, max_channels=None)

x_test : np.ndarray = result.x_test
y_test : np.ndarray = result.y_test.astype(int)

print(f"[INFO] X_test shape : {x_test.shape}")
print(f"[INFO] y_test shape : {y_test.shape}")
print(f"[INFO] Anomaly ratio: {y_test.mean():.3%}")

def load_model_and_predict(
    checkpoint_path: str,
    x_test: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, dict]:

    checkpoint = torch.load(checkpoint_path, map_location=device)

    meta = {k: checkpoint[k] for k in
            ["variant_name", "use_cnn", "use_attention",
             "use_skip", "use_gru", "input_dim",
             "seq_len", "rnn_hidden_size", "latent_dim",
             "best_val_loss"]}

    model = TimeSeriesAutoencoder(
        input_dim      = meta["input_dim"],
        seq_len        = meta["seq_len"],
        rnn_hidden_size= meta["rnn_hidden_size"],
        latent_dim     = meta["latent_dim"],
        use_gru        = meta["use_gru"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x_tensor = torch.tensor(x_test, dtype=torch.float32)
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(x_tensor), batch_size):
            batch = x_tensor[i : i + batch_size].to(device)
            pred  = model(
                batch,
                use_cnn      = meta["use_cnn"],
                use_attention= meta["use_attention"],
                use_skip     = meta["use_skip"],
            )
            all_preds.append(pred.cpu().numpy())

    x_pred = np.concatenate(all_preds, axis=0)
    return x_pred, meta

with open(os.path.join(SAVED_MODELS_DIR, "training_summary.json")) as f:
    training_summary = json.load(f)

all_metrics   : list[ClassificationMetrics] = []
all_errors    : dict[str, np.ndarray]       = {}
all_preds_dict: dict[str, np.ndarray]       = {}

THRESHOLD_METHODS = {
    "dynamic_nonparametric": lambda e: nonparametric_dynamic_threshold(e, window=30, z_score=3.0)[1],
    "moving_average"        : lambda e: moving_average_threshold(e, window=20, multiplier=2.5)[1],
}

PRIMARY_THRESHOLD = "dynamic_nonparametric"

print("\n" + "=" * 65)
print("  ABLATION EVALUATION STARTING")
print("=" * 65)

for item in training_summary:
    variant = item["variant_name"]
    ckpt    = os.path.join(SAVED_MODELS_DIR, f"{variant}.pth")

    if not os.path.exists(ckpt):
        print(f"[UYARI] {ckpt} not found, skipping.")
        continue

    print(f"\n[→] Evaluating: {variant}")

    x_pred, meta = load_model_and_predict(ckpt, x_test, DEVICE)
    all_preds_dict[variant] = x_pred

    err_dict = compute_reconstruction_errors(x_test, x_pred, metric="mse")
    errors   = aggregate_errors_to_scalar(err_dict["mse"], method="mean")
    all_errors[variant] = errors

    _, y_pred = nonparametric_dynamic_threshold(errors, window=30, z_score=3.0)

    best_val = item.get("best_val_loss", 0.0)
    m = compute_classification_metrics(
        y_test, y_pred,
        variant_name   = variant,
        threshold_type = PRIMARY_THRESHOLD,
        best_val_loss  = best_val,
    )
    all_metrics.append(m)

    print(f"    Accuracy : {m.accuracy:.4f}  |  Precision: {m.precision:.4f}")
    print(f"    Recall   : {m.recall:.4f}  |  F1-Score : {m.f1_score:.4f}")
    print(f"    TP={m.tp}  FP={m.fp}  TN={m.tn}  FN={m.fn}")

results_list = [m.as_dict() for m in all_metrics]
output_json  = os.path.join(OUTPUT_DIR, "evaluation_summary.json")

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results_list, f, indent=4, ensure_ascii=False)

print(f"\n[SAVED] {output_json}")

def plot_ablation_comparison(
    metrics: list[ClassificationMetrics],
    save_path: str,
) -> None:

    names  = [m.variant_name.replace("_", "\n") for m in metrics]
    f1s    = [m.f1_score   for m in metrics]
    precs  = [m.precision  for m in metrics]
    recs   = [m.recall     for m in metrics]
    accs   = [m.accuracy   for m in metrics]

    best_idx = int(np.argmax(f1s))
    colors   = ["#E87722" if i == best_idx else "#4A90D9" for i in range(len(metrics))]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    fig.suptitle("Ablation Study — Test Set Performance Comparison",
                 fontsize=15, fontweight="bold", y=1.01)

    metric_data = [
        (accs,  "Accuracy",  "#6DB33F"),
        (precs, "Precision", "#4A90D9"),
        (recs,  "Recall",    "#9B59B6"),
        (f1s,   "F1-Score",  "#E87722"),
    ]

    for ax, (vals, label, color) in zip(axes, metric_data):
        bar_colors = [color if i != best_idx else "#E87722"
                      for i in range(len(metrics))]
        if label == "F1-Score":
            bar_colors = ["#E87722" if i == best_idx else "#4A90D9"
                          for i in range(len(metrics))]

        bars = ax.barh(names, vals, color=bar_colors, edgecolor="white",
                       linewidth=0.8, height=0.55)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1.0)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.tick_params(axis="y", labelsize=8)

        for bar, val in zip(bars, vals):
            ax.text(
                min(val + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {save_path}")

plot_ablation_comparison(
    all_metrics,
    os.path.join(OUTPUT_DIR, "ablation_comparison_metrics.png"),
)

def plot_f1_vs_val_loss(
    metrics: list[ClassificationMetrics],
    save_path: str,
) -> None:

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(metrics)))

    for m, c in zip(metrics, colors):
        ax.scatter(m.best_val_loss, m.f1_score, color=c, s=120,
                   label=m.variant_name, zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(
            m.variant_name.split("_")[0],
            (m.best_val_loss, m.f1_score),
            textcoords="offset points", xytext=(5, 5), fontsize=8,
        )

    ax.set_xlabel("Best Validation Loss (Val MSE)", fontsize=11)
    ax.set_ylabel("F1-Score (Test)", fontsize=11)
    ax.set_title("Validation Loss vs Test F1-Score", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {save_path}")

plot_f1_vs_val_loss(
    all_metrics,
    os.path.join(OUTPUT_DIR, "f1_vs_val_loss.png"),
)

def plot_anomaly_scores(
    errors     : np.ndarray,
    thresholds : np.ndarray,
    y_true     : np.ndarray,
    variant_name: str,
    save_path  : str,
    max_points : int = 2000,
) -> None:

    n = min(len(errors), max_points)
    x = np.arange(n)
    e = errors[:n]
    t = thresholds[:n]
    y = y_true[:n]

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.fill_between(x, 0, e.max() * 1.1,
                    where=(y == 1), color="red", alpha=0.15,
                    label="True Anomaly")
    ax.plot(x, e, color="#4A90D9", linewidth=0.9, alpha=0.85, label="Error Score")
    ax.plot(x, t, color="#E87722", linewidth=1.4, linestyle="--",
            label="Dynamic Threshold")

    ax.set_xlabel("Sample Index", fontsize=10)
    ax.set_ylabel("Reconstruction Error (MSE)", fontsize=10)
    ax.set_title(f"Anomaly Score & Dynamic Threshold — {variant_name}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {save_path}")

FULL_MODEL = "full_lstm_cnn_attention_skip"
if FULL_MODEL in all_errors:
    errors_full  = all_errors[FULL_MODEL]
    thresholds_f, _ = nonparametric_dynamic_threshold(
        errors_full, window=30, z_score=3.0
    )
    plot_anomaly_scores(
        errors_full, thresholds_f, y_test,
        variant_name = FULL_MODEL,
        save_path    = os.path.join(OUTPUT_DIR, "anomaly_score_full_model.png"),
    )

if FULL_MODEL in all_preds_dict:
    ch_errors = channel_wise_errors(x_test, all_preds_dict[FULL_MODEL])
    mse_ch    = ch_errors["mse_per_channel"]
    ranking   = ch_errors["channel_ranking"]

    top_n = 15
    top_ch  = ranking[:top_n]
    top_mse = mse_ch[top_ch]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        [f"Ch{c}" for c in top_ch], top_mse,
        color="#4A90D9", edgecolor="white", linewidth=0.8,
    )
    ax.bar(
        [f"Ch{top_ch[0]}"], [top_mse[0]],
        color="#E87722", edgecolor="white", linewidth=0.8,
        label="Highest Error"
    )
    ax.set_xlabel("Channel", fontsize=11)
    ax.set_ylabel("Average MSE", fontsize=11)
    ax.set_title(f"Channel-Wise Reconstruction Error (Top {top_n})",
                 fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "channel_error_analysis.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"[SAVED] channel_error_analysis.png")

    with open(os.path.join(OUTPUT_DIR, "channel_errors.json"), "w") as f:
        json.dump({
            "mse_per_channel": mse_ch.tolist(),
            "top10_worst_channels": ranking[:10].tolist(),
        }, f, indent=4)

print("\n" + "=" * 75)
print(f"{'Variant':<38} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'ValLoss':>9}")
print("-" * 75)

for m in sorted(all_metrics, key=lambda x: x.f1_score, reverse=True):
    marker = " ★" if m == max(all_metrics, key=lambda x: x.f1_score) else ""
    print(
        f"{m.variant_name:<38} "
        f"{m.accuracy:>6.3f} "
        f"{m.precision:>6.3f} "
        f"{m.recall:>6.3f} "
        f"{m.f1_score:>6.3f} "
        f"{m.best_val_loss:>9.6f}"
        f"{marker}"
    )

print("=" * 75)
print(f"\n[COMPLETED] All results '{OUTPUT_DIR}/' klasörüne kaydedildi.")
