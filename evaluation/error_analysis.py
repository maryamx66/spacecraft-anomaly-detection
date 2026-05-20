from __future__ import annotations
import os, json, glob, sys

# 1. Point Python back to the main directory so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(current_dir)
sys.path.append(main_dir)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import PreprocessConfig
from data_loader   import build_telemanom_pipeline
from model         import TimeSeriesAutoencoder
from metrics import (
    compute_reconstruction_errors, aggregate_errors_to_scalar,
    nonparametric_dynamic_threshold, channel_wise_errors,
)

# 2. Route the paths to the main directory instead of the subfolder
SAVED_MODELS_DIR = os.path.join(main_dir, "saved_models")
OUTPUT_DIR       = os.path.join(main_dir, "evaluation_results")
FULL_MODEL_NAME  = "full_lstm_cnn_attention_skip"
DEVICE = torch.device("cuda"  if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    config = PreprocessConfig(sequence_length=50, stride=10, val_size=0.15, random_state=42)
    return build_telemanom_pipeline(config=config, max_channels=None)


def load_model(checkpoint_path, device):
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = TimeSeriesAutoencoder(
        input_dim=ckpt["input_dim"], seq_len=ckpt["seq_len"],
        rnn_hidden_size=ckpt["rnn_hidden_size"], latent_dim=ckpt["latent_dim"],
        use_gru=ckpt["use_gru"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def predict_full(model, x_np, meta, device, batch_size=256):
    x_t = torch.tensor(x_np, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_t), batch_size):
            b = x_t[i:i+batch_size].to(device)
            preds.append(model(b, use_cnn=meta["use_cnn"],
                               use_attention=meta["use_attention"],
                               use_skip=meta["use_skip"]).cpu().numpy())
    return np.concatenate(preds)


# ── 1. Misclassified samples ──────────────────────────────────────────────────
def plot_misclassified_samples(x_true, x_pred, y_true, y_pred_bin, n_examples=4, channel=0):
    fp_idx = np.where((y_pred_bin==1) & (y_true==0))[0]
    fn_idx = np.where((y_pred_bin==0) & (y_true==1))[0]
    rng = np.random.default_rng(42)
    fp_samples = rng.choice(fp_idx, min(n_examples, len(fp_idx)), replace=False)
    fn_samples = rng.choice(fn_idx, min(n_examples, len(fn_idx)), replace=False)
    n_rows = max(len(fp_samples), len(fn_samples), 1)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, n_rows*2.8+1.5), squeeze=False)
    fig.suptitle(
        f"Misclassified Samples — Channel {channel}\n"
        f"Left: False Positives (FP={len(fp_idx):,})   Right: False Negatives (FN={len(fn_idx):,})",
        fontsize=12, fontweight="bold")
    def _draw(ax, idx, kind, color):
        if idx is None: ax.axis("off"); return
        t = x_true[idx, :, channel]; p = x_pred[idx, :, channel]; ts = np.arange(len(t))
        ax.plot(ts, t, color="steelblue", label="Ground Truth", linewidth=1.3)
        ax.plot(ts, p, color=color, label="Prediction", linewidth=1.3, linestyle="--")
        ax.fill_between(ts, t, p, alpha=0.25, color=color)
        ax.set_title(f"{kind} | sample #{idx} | MSE={((t-p)**2).mean():.4f}",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.25)
    for row in range(n_rows):
        _draw(axes[row,0], fp_samples[row] if row<len(fp_samples) else None, "FP — False Alarm",    "#E87722")
        _draw(axes[row,1], fn_samples[row] if row<len(fn_samples) else None, "FN — Missed Anomaly", "#D9534F")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "misclassified_samples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[SAVED] {path}")


# ── 2. Overfitting analysis ───────────────────────────────────────────────────
def plot_overfitting_analysis():
    history_files = glob.glob(os.path.join(SAVED_MODELS_DIR, "*_history.json"))
    if not history_files: print("  [INFO] No history files found."); return
    fig, axes = plt.subplots(len(history_files), 1,
                              figsize=(10, len(history_files)*2.8), sharex=False)
    if len(history_files) == 1: axes = [axes]
    fig.suptitle("Overfitting Analysis — Train / Val Loss Curves",
                 fontsize=14, fontweight="bold", y=1.01)
    flags = {}
    for ax, hpath in zip(axes, sorted(history_files)):
        with open(hpath) as f: h = json.load(f)
        name    = h["variant_name"]
        train_l = np.array(h["train_losses"])
        val_l   = np.array(h["val_losses"])
        epochs  = np.arange(1, len(train_l)+1)
        gap     = (train_l[-5:] - val_l[-5:]).mean()
        flags[name] = bool(gap > 0.02)
        label = "Overfitting Risk" if flags[name] else "Healthy"
        ax.plot(epochs, train_l, color="#4A90D9", linewidth=1.4, label="Train Loss")
        ax.plot(epochs, val_l,   color="#E87722", linewidth=1.4, linestyle="--", label="Validation Loss")
        ax.fill_between(epochs, train_l, val_l, where=(train_l>val_l),
                        alpha=0.15, color="red", label="Train > Val (risk)")
        ax.fill_between(epochs, train_l, val_l, where=(train_l<=val_l), alpha=0.10, color="green")
        ax.set_title(f"{name}  |  Gap={gap:.4f}  |  {label}",
                     fontsize=9, fontweight="bold")
        ax.set_ylabel("MSE Loss", fontsize=9); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Epoch", fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "overfitting_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[SAVED] {path}")
    print("\n  Overfitting Summary:")
    for name, flag in flags.items():
        print(f"    {'[Risk]  ' if flag else '[OK]    '} {name}")


# ── 3. Error distribution ─────────────────────────────────────────────────────
def plot_error_distribution(errors, y_true, thresholds, variant):
    normal_err  = errors[y_true==0]
    anomaly_err = errors[y_true==1]
    dyn_thresh  = thresholds.mean()
    sep_ratio   = anomaly_err.mean() / (normal_err.mean() + 1e-8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.linspace(0, np.percentile(errors, 99), 60)
    ax1.hist(normal_err,  bins=bins, alpha=0.65, color="#4A90D9",
             label=f"Normal (n={len(normal_err):,})", density=True)
    ax1.hist(anomaly_err, bins=bins, alpha=0.65, color="#D9534F",
             label=f"Anomaly (n={len(anomaly_err):,})", density=True)
    ax1.axvline(dyn_thresh, color="#E87722", linestyle="--", linewidth=1.6,
                label=f"Avg. Dynamic Threshold = {dyn_thresh:.4f}")
    ax1.set_xlabel("Reconstruction Error (MSE)", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.set_title(f"Error Distribution — {variant}\nSeparation Ratio: {sep_ratio:.2f}x",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25)
    ax2.boxplot([normal_err, anomaly_err], labels=["Normal", "Anomaly"], patch_artist=True,
                boxprops=dict(facecolor="#4A90D9", alpha=0.7),
                medianprops=dict(color="white", linewidth=2),
                flierprops=dict(marker=".", markersize=3, alpha=0.3))
    patches = ax2.patches
    if len(patches) >= 2: patches[1].set_facecolor("#D9534F"); patches[1].set_alpha(0.7)
    ax2.axhline(dyn_thresh, color="#E87722", linestyle="--", linewidth=1.4, label="Avg. Threshold")
    ax2.set_ylabel("Reconstruction Error", fontsize=10)
    ax2.set_title("Box Plot Comparison", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"error_distribution_{variant}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[SAVED] {path}")
    print(f"  Normal mean error  : {normal_err.mean():.5f}")
    print(f"  Anomaly mean error : {anomaly_err.mean():.5f}")
    print(f"  Separation ratio   : {sep_ratio:.2f}x")


# ── 4. Attention visualization (BONUS) ───────────────────────────────────────
def plot_attention_heatmap(model, x_true, y_true, errors, meta, device, n_samples=3):
    normal_top  = np.argsort(errors[y_true==0])[:n_samples]
    anomaly_top = np.argsort(errors[y_true==1])[::-1][:n_samples]
    normal_idx  = np.where(y_true==0)[0][normal_top]
    anomaly_idx = np.where(y_true==1)[0][anomaly_top]
    sample_indices = list(normal_idx) + list(anomaly_idx)
    sample_labels  = ([f"Normal #{i}" for i in normal_idx] +
                      [f"Anomaly #{i}" for i in anomaly_idx])

    captured = []
    def make_hook(_buf):
        def _h(module, inp, out):
            v = out[1] if isinstance(out, tuple) and len(out) > 1 else out
            if v is not None:
                _buf.append(v.detach().cpu().float().numpy())
        return _h

    fig, axes = plt.subplots(len(sample_indices), 2, figsize=(14, len(sample_indices)*3),
                              gridspec_kw={"width_ratios": [3, 1]}, squeeze=False)
    fig.suptitle("Attention Visualization — Time-step Weights (BONUS)",
                 fontsize=13, fontweight="bold", y=1.01)

    for row, (idx, lbl) in enumerate(zip(sample_indices, sample_labels)):
        ax_ts, ax_att = axes[row, 0], axes[row, 1]
        color = "#D9534F" if "Anomaly" in lbl else "#4A90D9"
        ax_ts.plot(x_true[idx, :, 0], color=color, linewidth=1.3)
        ax_ts.set_facecolor("#FFF5F5" if "Anomaly" in lbl else "white")
        ax_ts.set_title(f"{lbl} | MSE={errors[idx]:.4f}", fontsize=9,
                        fontweight="bold", color=color)
        ax_ts.set_xlabel("Time Step", fontsize=8)
        ax_ts.set_ylabel("Value", fontsize=8)
        ax_ts.grid(True, alpha=0.25)

        buf = []
        hooks = []
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.MultiheadAttention):
                hooks.append(mod.register_forward_hook(make_hook(buf)))
        if not hooks:
            for name, mod in model.named_modules():
                if "attn" in name.lower() or "attention" in name.lower():
                    hooks.append(mod.register_forward_hook(make_hook(buf)))
                    break

        xt = torch.tensor(x_true[idx][np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            model(xt, use_cnn=meta["use_cnn"],
                  use_attention=meta["use_attention"], use_skip=meta["use_skip"])
        for h in hooks: h.remove()

        w = None
        if buf:
            w = buf[0]
            if w.ndim == 4:   w = w.mean(axis=(0, 1))
            elif w.ndim == 3: w = w.mean(axis=0)
            elif w.ndim == 2 and w.shape[0] == 1: w = w[0]

        if w is not None and w.ndim == 2 and w.shape[0] > 1:
            im = ax_att.imshow(w, cmap="hot", aspect="auto")
            ax_att.set_title("Attention Map", fontsize=9)
            ax_att.set_xlabel("Key", fontsize=8); ax_att.set_ylabel("Query", fontsize=8)
            plt.colorbar(im, ax=ax_att, fraction=0.046, pad=0.04)
            captured.append(idx)
        elif w is not None and w.ndim == 1:
            ax_att.barh(np.arange(len(w)), w, color="coral")
            ax_att.set_title("Attention Weights", fontsize=9); ax_att.invert_yaxis()
            captured.append(idx)
        else:
            ts_err = ((x_true[idx, :, 0] - 0) ** 2)
            ax_att.bar(np.arange(len(ts_err)), ts_err, color=color, alpha=0.7)
            ax_att.set_title("Per-step MSE", fontsize=9)
            ax_att.set_xlabel("Time Step", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "attention_visualization.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[SAVED] {path}")
    if captured:
        print(f"  Attention weights captured for {len(captured)} samples.")
    else:
        print("  [INFO] Attention hook not available for this architecture.")
        print("         Fallback: per-step MSE shown instead.")


# ── 5. Failure mode analysis — 4-panel histogram ─────────────────────────────
def plot_failure_modes(errors, y_true, y_pred, variant):
    tp_idx = np.where((y_pred==1) & (y_true==1))[0]
    tn_idx = np.where((y_pred==0) & (y_true==0))[0]
    fp_idx = np.where((y_pred==1) & (y_true==0))[0]
    fn_idx = np.where((y_pred==0) & (y_true==1))[0]

    categories = [
        (tp_idx, "TP — Correct Anomaly",  "#28A745"),
        (tn_idx, "TN — Correct Normal",   "#4A90D9"),
        (fp_idx, "FP — False Alarm",      "#E87722"),
        (fn_idx, "FN — Missed Anomaly",   "#D9534F"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Failure Mode Analysis — {variant}", fontsize=13, fontweight="bold")

    for ax, (idx, label, color) in zip(axes.flat, categories):
        if len(idx) == 0:
            ax.text(0.5, 0.5, "No Samples", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title(label, fontsize=10, fontweight="bold"); continue
        vals = errors[idx]
        ax.hist(vals, bins=30, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.4,
                   label=f"Mean = {vals.mean():.4f}")
        ax.set_title(f"{label}\nn={len(idx):,}   Mean={vals.mean():.4f}   Std={vals.std():.4f}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Error Score (MSE)", fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"failure_modes_{variant}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[SAVED] {path}")
    print(f"  FP (False Alarms)    : {len(fp_idx):>6,} samples  (mean error = {errors[fp_idx].mean() if len(fp_idx)>0 else 0:.4f})")
    print(f"  FN (Missed Anomalies): {len(fn_idx):>6,} samples  (mean error = {errors[fn_idx].mean() if len(fn_idx)>0 else 0:.4f})")


# ── 6. Component contribution — DELTA (not absolute F1) ──────────────────────
def plot_component_contribution():

    eval_json = os.path.join(
        OUTPUT_DIR,
        "evaluation_summary.json"
    )

    if not os.path.exists(eval_json):
        print("evaluation_summary.json not found.")
        return

    with open(eval_json) as f:
        evals = {
            d["variant_name"]: d
            for d in json.load(f)
        }

    # ---------------------------------------------------------
    # CORRECTED PAIRS
    # ---------------------------------------------------------

    pairs = {

        "CNN Block": (
            "full_lstm_cnn_attention_skip",
            "no_cnn_lstm_attention_skip"
        ),

        "Skip Connection": (
            "full_lstm_cnn_attention_skip",
            "no_skip_lstm_cnn_attention"
        ),

        "Attention Mechanism": (
            "full_lstm_cnn_attention_skip",
            "no_attention_lstm_cnn_skip"
        ),

        # FIXED ORDER
        "LSTM vs GRU": (
            "full_lstm_cnn_attention_skip",
            "full_gru_cnn_attention_skip"
        ),
    }

    labels = []
    deltas = []

    for label, (full_model, ablated_model) in pairs.items():

        full_f1 = evals.get(
            full_model,
            {}
        ).get("f1_score")

        ablated_f1 = evals.get(
            ablated_model,
            {}
        ).get("f1_score")

        if full_f1 is None or ablated_f1 is None:
            continue

        labels.append(label)
        deltas.append(full_f1 - ablated_f1)

    if not labels:
        return

    # ---------------------------------------------------------
    # COLORS
    # ---------------------------------------------------------

    colors = [
        "#28A745" if d > 0 else "#D9534F"
        for d in deltas
    ]

    # ---------------------------------------------------------
    # FIGURE
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.barh(
        labels,
        deltas,
        color=colors,
        edgecolor="white",
        height=0.55
    )

    # zero reference line
    ax.axvline(
        0,
        color="black",
        linewidth=1.5
    )

    # ---------------------------------------------------------
    # TITLE
    # ---------------------------------------------------------

    ax.set_title(
        "Ablation Study: Component Impact on F1-Score",
        fontsize=18,
        fontweight="bold",
        pad=16
    )

    ax.set_xlabel(
        "Δ F1-Score",
        fontsize=13
    )

    # ---------------------------------------------------------
    # DYNAMIC LIMITS
    # ---------------------------------------------------------

    min_delta = min(deltas)
    max_delta = max(deltas)

    padding = 0.004

    ax.set_xlim(
        min_delta - padding,
        max_delta + padding
    )

    # ---------------------------------------------------------
    # VALUE LABELS
    # ---------------------------------------------------------

    for bar, delta in zip(bars, deltas):

        y = (
            bar.get_y()
            + bar.get_height() / 2
        )

        label = f"{delta:+.4f}"

        if delta >= 0:

            ax.text(
                delta + 0.0005,
                y,
                label,
                va="center",
                ha="left",
                fontsize=12,
                fontweight="bold"
            )

        else:

            ax.text(
                delta - 0.0005,
                y,
                label,
                va="center",
                ha="right",
                fontsize=12,
                fontweight="bold"
            )

    # ---------------------------------------------------------
    # STYLE
    # ---------------------------------------------------------

    ax.grid(
        True,
        axis="x",
        alpha=0.25
    )

    ax.tick_params(
        axis="both",
        labelsize=12
    )

    plt.tight_layout()

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    path = os.path.join(
        OUTPUT_DIR,
        "component_contribution.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[SAVED] {path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Loading data...")
    data   = load_data()
    x_test = data.x_test
    y_test = data.y_test.astype(int)

    print("[INFO] Loading full model...")
    model, meta = load_model(
        os.path.join(SAVED_MODELS_DIR, f"{FULL_MODEL_NAME}.pth"), DEVICE)
    x_pred = predict_full(model, x_test, meta, DEVICE)

    err_dict   = compute_reconstruction_errors(x_test, x_pred, metric="mse")
    errors     = aggregate_errors_to_scalar(err_dict["mse"])
    thresholds, y_pred_bin = nonparametric_dynamic_threshold(errors, window=30, z_score=3.0)

    print("\n[1/6] Misclassified samples...")
    plot_misclassified_samples(x_test, x_pred, y_test, y_pred_bin)

    print("\n[2/6] Overfitting analysis...")
    plot_overfitting_analysis()

    print("\n[3/6] Error distribution...")
    plot_error_distribution(errors, y_test, thresholds, FULL_MODEL_NAME)

    print("\n[4/6] Attention visualization (BONUS)...")
    plot_attention_heatmap(model, x_test, y_test, errors, meta, DEVICE)

    print("\n[5/6] Failure mode analysis (4-panel histogram)...")
    plot_failure_modes(errors, y_test, y_pred_bin, FULL_MODEL_NAME)

    print("\n[6/6] Component contribution analysis (delta F1)...")
    plot_component_contribution()

    print(f"\n[DONE] All analyses saved to '{OUTPUT_DIR}/'")
