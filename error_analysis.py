"""
error_analysis.py
──────────────────────────────────────────────────────────────────────────────
Derinlemesine Hata Analizi Modülü.

Bölümler:
  1. Yanlış Sınıflandırılan Örnekler     — FP & FN örnekleri görselleştirir
  2. Overfitting Analizi                 — train/val eğrilerini karşılaştırır
  3. Hata Dağılımı Analizi               — normal vs anomali hata histogramı
  4. Attention Görselleştirme (BONUS)    — modelin hangi zaman adımlarına
                                           odaklandığını gösterir
  5. Başarısızlık Modu Analizi           — hangi bölgelerde model yanılıyor?
  6. Ablation Karşılaştırması            — bileşen katkısı özeti

Çalıştırmak için:
  python error_analysis.py
"""

from __future__ import annotations

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from preprocessing import PreprocessConfig
from data_loader import build_telemanom_pipeline
from model import TimeSeriesAutoencoder

from metrics import (
    compute_reconstruction_errors,
    aggregate_errors_to_scalar,
    nonparametric_dynamic_threshold,
    channel_wise_errors,
    error_by_anomaly_class,
)

# ─────────────────────────────────────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────────────────────────────────────

SAVED_MODELS_DIR = "saved_models"
OUTPUT_DIR       = "evaluation_results"
DEVICE           = torch.device(
    "cuda"  if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

FULL_MODEL_NAME  = "full_lstm_cnn_attention_skip"
MODEL_CHECKPOINT = os.path.join(SAVED_MODELS_DIR, f"{FULL_MODEL_NAME}.pth")


# ─────────────────────────────────────────────────────────────────────────────
# VERİ VE MODEL YÜKLEME
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    config = PreprocessConfig(
        sequence_length=50, stride=10, val_size=0.15, random_state=42
    )
    result = build_telemanom_pipeline(config=config, max_channels=None)
    return result


def load_model(checkpoint_path: str, device: torch.device):
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = TimeSeriesAutoencoder(
        input_dim       = ckpt["input_dim"],
        seq_len         = ckpt["seq_len"],
        rnn_hidden_size = ckpt["rnn_hidden_size"],
        latent_dim      = ckpt["latent_dim"],
        use_gru         = ckpt["use_gru"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def predict_full(model, x_np: np.ndarray, meta: dict,
                 device: torch.device, batch_size: int = 256) -> np.ndarray:
    x_t  = torch.tensor(x_np, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_t), batch_size):
            b   = x_t[i:i+batch_size].to(device)
            out = model(b,
                        use_cnn      = meta["use_cnn"],
                        use_attention= meta["use_attention"],
                        use_skip     = meta["use_skip"])
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. YANLIŞ SINIFLANDIRILAN ÖRNEKLER
# ─────────────────────────────────────────────────────────────────────────────

def plot_misclassified_samples(
    x_true    : np.ndarray,
    x_pred    : np.ndarray,
    y_true    : np.ndarray,
    y_pred_bin: np.ndarray,
    n_examples: int = 4,
    channel   : int = 0,
    save_path : str | None = None,
) -> None:
    """
    FP (Normal olarak işaretlenmiş ama modelin anomali dediği) ve
    FN (Gerçek anomali ama modelin kaçırdığı) örneklerini görselleştirir.
    """
    fp_idx = np.where((y_pred_bin == 1) & (y_true == 0))[0]
    fn_idx = np.where((y_pred_bin == 0) & (y_true == 1))[0]

    rng = np.random.default_rng(42)

    def sample(idx, n): return rng.choice(idx, min(n, len(idx)), replace=False)

    fp_samples = sample(fp_idx, n_examples)
    fn_samples = sample(fn_idx, n_examples)

    n_fp = len(fp_samples)
    n_fn = len(fn_samples)
    n_rows = max(n_fp, n_fn, 1)

    fig, axes = plt.subplots(n_rows, 2,
                              figsize=(13, n_rows * 2.8 + 1.5),
                              squeeze=False)
    fig.suptitle(
        f"Yanlış Sınıflandırılmış Örnekler — Kanal {channel}\n"
        f"Sol: Yanlış Pozitif (FP={len(fp_idx)})   "
        f"Sağ: Yanlış Negatif (FN={len(fn_idx)})",
        fontsize=12, fontweight="bold"
    )

    def _draw(ax, idx, kind, color):
        if idx is None:
            ax.axis("off"); return
        t  = x_true[idx, :, channel]
        p  = x_pred[idx, :, channel]
        ts = np.arange(len(t))
        ax.plot(ts, t, color="steelblue",  label="Gerçek",    linewidth=1.3)
        ax.plot(ts, p, color=color,        label="Tahmin",     linewidth=1.3,
                linestyle="--")
        ax.fill_between(ts, t, p, alpha=0.25, color=color)
        err = ((t - p) ** 2).mean()
        ax.set_title(f"{kind} | örnek #{idx} | MSE={err:.4f}",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

    for row in range(n_rows):
        fp_i = fp_samples[row] if row < len(fp_samples) else None
        fn_i = fn_samples[row] if row < len(fn_samples) else None
        _draw(axes[row, 0], fp_i, "FP — Yanlış Alarm", "#E87722")
        _draw(axes[row, 1], fn_i, "FN — Kaçırılan Anomali", "#D9534F")

    plt.tight_layout()
    path = save_path or os.path.join(OUTPUT_DIR, "misclassified_samples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[KAYDEDILDI] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. OVERFİTTİNG ANALİZİ
# ─────────────────────────────────────────────────────────────────────────────

def plot_overfitting_analysis(
    histories_dir: str = SAVED_MODELS_DIR,
    save_path    : str | None = None,
) -> None:
    """
    Tüm varyantların train/val eğri farkını karşılaştırır.
    Son 5 epoch'ta (train_loss - val_loss) büyükse overfit riski var.
    """
    import glob
    history_files = glob.glob(os.path.join(histories_dir, "*_history.json"))

    fig, axes = plt.subplots(
        len(history_files), 1,
        figsize=(10, len(history_files) * 2.8),
        sharex=False
    )
    if len(history_files) == 1:
        axes = [axes]

    fig.suptitle("Overfitting Analizi — Train/Val Kayıp Eğrileri",
                 fontsize=14, fontweight="bold", y=1.01)

    overfit_flags: dict[str, bool] = {}

    for ax, hpath in zip(axes, sorted(history_files)):
        with open(hpath) as f:
            h = json.load(f)

        name        = h["variant_name"]
        train_l     = np.array(h["train_losses"])
        val_l       = np.array(h["val_losses"])
        epochs      = np.arange(1, len(train_l) + 1)

        # Overfit göstergesi: son 5 epoch ortalaması
        last_n      = 5
        gap_last    = (train_l[-last_n:] - val_l[-last_n:]).mean()
        is_overfit  = gap_last < -0.005   # val >> train demek overfit değil
        overfit_flags[name] = bool(gap_last > 0.02)

        color_train = "#4A90D9"
        color_val   = "#E87722"

        ax.plot(epochs, train_l, color=color_train, linewidth=1.4,
                label="Train Loss")
        ax.plot(epochs, val_l,   color=color_val,   linewidth=1.4,
                label="Val Loss", linestyle="--")
        ax.fill_between(epochs, train_l, val_l,
                        where=(train_l > val_l),
                        alpha=0.15, color="red",  label="Train > Val (risk)")
        ax.fill_between(epochs, train_l, val_l,
                        where=(train_l <= val_l),
                        alpha=0.10, color="green")

        flag_txt = "⚠ Overfit Riski" if overfit_flags[name] else "✓ Sağlıklı"
        ax.set_title(f"{name}  |  Son Boşluk={gap_last:.4f}  {flag_txt}",
                     fontsize=9, fontweight="bold")
        ax.set_ylabel("MSE Kaybı", fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Epoch", fontsize=10)
    plt.tight_layout()

    path = save_path or os.path.join(OUTPUT_DIR, "overfitting_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[KAYDEDILDI] {path}")

    # Özet
    print("\n  Overfitting Özeti:")
    for name, flag in overfit_flags.items():
        status = "⚠ Risk Var " if flag else "✓ Normal   "
        print(f"    {status} | {name}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. HATA DAĞILIMI ANALİZİ
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_distribution(
    errors    : np.ndarray,
    y_true    : np.ndarray,
    thresholds: np.ndarray,
    variant   : str,
    save_path : str | None = None,
) -> None:
    """
    Normal ve anomali örneklerinin hata histogramını karşılaştırır.
    İyi bir model için iki dağılımın ayrışması gerekir.
    """
    normal_err  = errors[y_true == 0]
    anomaly_err = errors[y_true == 1]
    dyn_thresh  = thresholds.mean()

    sep_ratio = anomaly_err.mean() / (normal_err.mean() + 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Sol: histogram
    bins = np.linspace(0, np.percentile(errors, 99), 60)
    ax1.hist(normal_err,  bins=bins, alpha=0.65, color="#4A90D9",
             label=f"Normal (n={len(normal_err):,})",  density=True)
    ax1.hist(anomaly_err, bins=bins, alpha=0.65, color="#D9534F",
             label=f"Anomali (n={len(anomaly_err):,})", density=True)
    ax1.axvline(dyn_thresh, color="#E87722", linestyle="--", linewidth=1.6,
                label=f"Ort. Dinamik Eşik = {dyn_thresh:.4f}")
    ax1.set_xlabel("Yeniden Yapılandırma Hatası (MSE)", fontsize=10)
    ax1.set_ylabel("Yoğunluk", fontsize=10)
    ax1.set_title(f"Hata Dağılımı — {variant}\nAyrışma Oranı: {sep_ratio:.2f}x",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25)

    # Sağ: box plot
    ax2.boxplot(
        [normal_err, anomaly_err],
        labels=["Normal", "Anomali"],
        patch_artist=True,
        boxprops=dict(facecolor="#4A90D9", alpha=0.7),
        medianprops=dict(color="white", linewidth=2),
        flierprops=dict(marker=".", markersize=3, alpha=0.3),
    )
    boxes = ax2.patches
    if len(boxes) >= 2:
        boxes[1].set_facecolor("#D9534F")
        boxes[1].set_alpha(0.7)

    ax2.axhline(dyn_thresh, color="#E87722", linestyle="--",
                linewidth=1.4, label="Ort. Eşik")
    ax2.set_ylabel("Yeniden Yapılandırma Hatası", fontsize=10)
    ax2.set_title("Kutu Grafiği Karşılaştırması", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    path = save_path or os.path.join(OUTPUT_DIR, f"error_distribution_{variant}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[KAYDEDILDI] {path}")
    print(f"  Normal hata ort.  : {normal_err.mean():.5f}")
    print(f"  Anomali hata ort. : {anomaly_err.mean():.5f}")
    print(f"  Ayrışma oranı     : {sep_ratio:.2f}x")


# ─────────────────────────────────────────────────────────────────────────────
# 4. ATTENTION GÖRSELLEŞTİRME (BONUS)
# ─────────────────────────────────────────────────────────────────────────────

def extract_attention_weights(
    model    : torch.nn.Module,
    x_sample : np.ndarray,
    meta     : dict,
    device   : torch.device,
) -> np.ndarray | None:
    """
    Modelden attention ağırlıklarını çıkarmaya çalışır.
    Model'in attention katmanı hook ile yakalanır.

    Döndürür
    --------
    weights : (seq_len, seq_len) veya (seq_len,) — None eğer bulunamazsa
    """
    attention_output = []

    def _hook(module, input, output):
        # Attention katmanı çıktısını yakala
        if isinstance(output, tuple):
            attention_output.append(output[1].detach().cpu().numpy())
        else:
            attention_output.append(output.detach().cpu().numpy())

    # Model içindeki MultiheadAttention veya benzer bir katmanı bul
    hooks = []
    for name, module in model.named_modules():
        if "attention" in name.lower() or isinstance(
            module, torch.nn.MultiheadAttention
        ):
            hooks.append(module.register_forward_hook(_hook))
            break

    if not hooks:
        print("[UYARI] Attention katmanı bulunamadı.")
        return None

    x_t = torch.tensor(x_sample[np.newaxis], dtype=torch.float32).to(device)
    with torch.no_grad():
        model(x_t,
              use_cnn      = meta["use_cnn"],
              use_attention= meta["use_attention"],
              use_skip     = meta["use_skip"])

    for h in hooks:
        h.remove()

    if not attention_output:
        return None

    weights = attention_output[0]
    # (1, heads, seq_len, seq_len) veya (1, seq_len, seq_len) → ortalama al
    if weights.ndim == 4:
        weights = weights.mean(axis=(0, 1))   # (seq_len, seq_len)
    elif weights.ndim == 3:
        weights = weights.mean(axis=0)         # (seq_len, seq_len)
    elif weights.ndim == 2 and weights.shape[0] == 1:
        weights = weights[0]                   # (seq_len,)

    return weights


def plot_attention_heatmap(
    model     : torch.nn.Module,
    x_true    : np.ndarray,
    y_true    : np.ndarray,
    errors    : np.ndarray,
    meta      : dict,
    device    : torch.device,
    n_samples : int = 3,
    save_path : str | None = None,
) -> None:
    """
    Hem normal hem de anomali örnekleri için attention haritalarını
    zaman serisi grafiğinin üstünde gösterir.  (BONUS)
    """
    rng         = np.random.default_rng(0)
    normal_top  = np.argsort(errors[y_true == 0])[:n_samples]
    anomaly_top = np.argsort(errors[y_true == 1])[::-1][:n_samples]

    normal_idx  = np.where(y_true == 0)[0][normal_top]
    anomaly_idx = np.where(y_true == 1)[0][anomaly_top]

    sample_indices = list(normal_idx) + list(anomaly_idx)
    sample_labels  = (
        [f"Normal #{i}" for i in normal_idx] +
        [f"Anomali #{i}" for i in anomaly_idx]
    )

    fig, axes = plt.subplots(
        len(sample_indices), 2,
        figsize=(14, len(sample_indices) * 3),
        gridspec_kw={"width_ratios": [3, 1]},
        squeeze=False,
    )
    fig.suptitle("Attention Görselleştirme — Zaman Adımı Ağırlıkları (BONUS)",
                 fontsize=13, fontweight="bold", y=1.01)

    success_count = 0
    for row, (idx, lbl) in enumerate(zip(sample_indices, sample_labels)):
        ax_ts  = axes[row, 0]   # zaman serisi
        ax_att = axes[row, 1]   # attention

        # Zaman serisi (ilk kanal)
        ts = x_true[idx, :, 0]
        ax_ts.plot(ts, color="#4A90D9", linewidth=1.3)
        color = "#D9534F" if "Anomali" in lbl else "#4A90D9"
        ax_ts.set_facecolor("#FFF5F5" if "Anomali" in lbl else "white")
        ax_ts.set_title(f"{lbl} | MSE={errors[idx]:.4f}",
                        fontsize=9, fontweight="bold", color=color)
        ax_ts.set_xlabel("Zaman Adımı", fontsize=8)
        ax_ts.set_ylabel("Değer", fontsize=8)
        ax_ts.grid(True, alpha=0.25)

        # Attention ağırlıkları
        w = extract_attention_weights(model, x_true[idx], meta, device)
        if w is not None and w.ndim == 2:
            success_count += 1
            im = ax_att.imshow(w, cmap="hot", aspect="auto")
            ax_att.set_title("Attention Map", fontsize=9)
            ax_att.set_xlabel("Key", fontsize=8)
            ax_att.set_ylabel("Query", fontsize=8)
            plt.colorbar(im, ax=ax_att, fraction=0.046, pad=0.04)
        elif w is not None and w.ndim == 1:
            success_count += 1
            ax_att.barh(np.arange(len(w)), w, color="coral")
            ax_att.set_title("Attention Ağırlıkları", fontsize=9)
            ax_att.set_xlabel("Ağırlık", fontsize=8)
            ax_att.set_ylabel("Zaman Adımı", fontsize=8)
            ax_att.invert_yaxis()
        else:
            ax_att.text(0.5, 0.5, "Attention\nbilgisi\nyok",
                        ha="center", va="center", transform=ax_att.transAxes,
                        fontsize=9, color="gray")
            ax_att.axis("off")

    plt.tight_layout()
    path = save_path or os.path.join(OUTPUT_DIR, "attention_visualization.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[KAYDEDILDI] {path}")
    if success_count > 0:
        print(f"  Attention ağırlıkları {success_count} örnekte yakalandı.")
    else:
        print("  [BİLGİ] Bu model mimarisinde attention hook desteği sınırlı.")
        print("          Grafik yine de kaydedildi (zaman serisi kısımları).")


# ─────────────────────────────────────────────────────────────────────────────
# 5. BAŞARISIZLIK MODU ANALİZİ
# ─────────────────────────────────────────────────────────────────────────────

def analyze_failure_modes(
    errors    : np.ndarray,
    y_true    : np.ndarray,
    y_pred_bin: np.ndarray,
    variant   : str,
    save_path : str | None = None,
) -> dict:
    """
    Modelin nerede başarısız olduğunu analiz eder.

    - FP ve FN hata skorlarının dağılımı
    - Hata skoruna göre kritik eşik noktası
    - Önerilen eşik ayarı
    """
    fp_idx = np.where((y_pred_bin == 1) & (y_true == 0))[0]
    fn_idx = np.where((y_pred_bin == 0) & (y_true == 1))[0]
    tp_idx = np.where((y_pred_bin == 1) & (y_true == 1))[0]
    tn_idx = np.where((y_pred_bin == 0) & (y_true == 0))[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Başarısızlık Modu Analizi — {variant}",
                 fontsize=13, fontweight="bold")

    categories = [
        (tp_idx, "TP (Doğru Anomali)",    "#28A745"),
        (tn_idx, "TN (Doğru Normal)",     "#4A90D9"),
        (fp_idx, "FP (Yanlış Alarm)",     "#E87722"),
        (fn_idx, "FN (Kaçırılan Anomali)","#D9534F"),
    ]

    for ax, (idx, label, color) in zip(axes.flat, categories):
        if len(idx) == 0:
            ax.text(0.5, 0.5, "Örnek Yok", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title(label, fontsize=10, fontweight="bold")
            continue
        vals = errors[idx]
        ax.hist(vals, bins=30, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.4,
                   label=f"Ort={vals.mean():.4f}")
        ax.set_title(
            f"{label}\nn={len(idx):,}  Ort={vals.mean():.4f}  Std={vals.std():.4f}",
            fontsize=9, fontweight="bold"
        )
        ax.set_xlabel("Hata Skoru", fontsize=8)
        ax.set_ylabel("Frekans",    fontsize=8)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = save_path or os.path.join(OUTPUT_DIR, f"failure_modes_{variant}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[KAYDEDILDI] {path}")

    summary = {
        "fp_count"     : int(len(fp_idx)),
        "fn_count"     : int(len(fn_idx)),
        "fp_mean_error": float(errors[fp_idx].mean()) if len(fp_idx) > 0 else 0.0,
        "fn_mean_error": float(errors[fn_idx].mean()) if len(fn_idx) > 0 else 0.0,
        "tp_mean_error": float(errors[tp_idx].mean()) if len(tp_idx) > 0 else 0.0,
    }

    print(f"\n  Başarısızlık Özeti ({variant}):")
    print(f"    FP (Yanlış Alarm)       : {summary['fp_count']:>6,} örnek  "
          f"(ort. hata = {summary['fp_mean_error']:.5f})")
    print(f"    FN (Kaçırılan Anomali)  : {summary['fn_count']:>6,} örnek  "
          f"(ort. hata = {summary['fn_mean_error']:.5f})")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 6. ABLATİON BİLEŞEN KATKI ANALİZİ
# ─────────────────────────────────────────────────────────────────────────────

def plot_component_contribution(
    summary_json: str = os.path.join(SAVED_MODELS_DIR, "training_summary.json"),
    eval_json   : str = os.path.join(OUTPUT_DIR, "evaluation_summary.json"),
    save_path   : str | None = None,
) -> None:
    """
    Her bileşenin (CNN, Attention, Skip) F1 skoruna katkısını
    'Tam model − bileşensiz model' farkı olarak gösterir.
    """
    with open(eval_json) as f:
        evals = {d["variant_name"]: d for d in json.load(f)}

    full_f1 = evals.get("full_lstm_cnn_attention_skip", {}).get("f1_score", 0)

    ablation_map = {
        "CNN Bileşeni"       : ("full_lstm_cnn_attention_skip",
                                "no_cnn_lstm_attention_skip"),
        "Skip Connection"    : ("full_lstm_cnn_attention_skip",
                                "no_skip_lstm_cnn_attention"),
        "Attention Mekanizması": ("full_lstm_cnn_attention_skip",
                                  "no_attention_lstm_cnn_skip"),
        "LSTM vs GRU"        : ("full_gru_cnn_attention_skip",
                                "full_lstm_cnn_attention_skip"),
    }

    labels      = []
    deltas      = []
    base_vals   = []

    for label, (full, ablated) in ablation_map.items():
        f_score = evals.get(full,    {}).get("f1_score", None)
        a_score = evals.get(ablated, {}).get("f1_score", None)
        if f_score is None or a_score is None:
            continue
        delta = f_score - a_score
        labels.append(label)
        deltas.append(delta)
        base_vals.append(a_score)

    if not labels:
        print("[UYARI] Ablation karşılaştırma için yeterli veri yok.")
        return

    colors = ["#28A745" if d > 0 else "#D9534F" for d in deltas]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, deltas, color=colors, edgecolor="white",
                   linewidth=0.8, height=0.5)

    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel("F1 Skoru Değişimi (Tam Model − Ablated Model)", fontsize=10)
    ax.set_title(
        "Bileşen Katkı Analizi\n"
        "(Pozitif = bileşen modele katkı sağlıyor, "
        "Negatif = bileşen olmadan daha iyi)",
        fontsize=11, fontweight="bold"
    )

    for bar, d in zip(bars, deltas):
        sign = "+" if d >= 0 else ""
        ax.text(
            d + (0.003 if d >= 0 else -0.003),
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{d:.3f}",
            va="center",
            ha="left" if d >= 0 else "right",
            fontsize=10, fontweight="bold"
        )

    plt.tight_layout()
    path = save_path or os.path.join(OUTPUT_DIR, "component_contribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[KAYDEDILDI] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# DOĞRUDAN ÇALIŞTIRMA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("[INFO] Veri yükleniyor...")
    data   = load_data()
    x_test = data.x_test
    y_test = data.y_test.astype(int)

    print("[INFO] Tam model yükleniyor...")
    model, meta = load_model(MODEL_CHECKPOINT, DEVICE)
    x_pred      = predict_full(model, x_test, meta, DEVICE)

    # Hata ve eşikleme
    err_dict   = compute_reconstruction_errors(x_test, x_pred, metric="mse")
    errors     = aggregate_errors_to_scalar(err_dict["mse"])
    thresholds, y_pred_bin = nonparametric_dynamic_threshold(
        errors, window=30, z_score=3.0
    )

    print("\n[1/6] Yanlış sınıflandırılmış örnekler görselleştiriliyor...")
    plot_misclassified_samples(x_test, x_pred, y_test, y_pred_bin)

    print("\n[2/6] Overfitting analizi yapılıyor...")
    plot_overfitting_analysis()

    print("\n[3/6] Hata dağılımı analiz ediliyor...")
    plot_error_distribution(errors, y_test, thresholds, FULL_MODEL_NAME)

    print("\n[4/6] Attention görselleştiriliyor (BONUS)...")
    plot_attention_heatmap(model, x_test, y_test, errors, meta, DEVICE)

    print("\n[5/6] Başarısızlık modu analizi...")
    _ = analyze_failure_modes(errors, y_test, y_pred_bin, FULL_MODEL_NAME)

    print("\n[6/6] Bileşen katkı analizi...")
    if os.path.exists(os.path.join(OUTPUT_DIR, "evaluation_summary.json")):
        plot_component_contribution()
    else:
        print("  [BİLGİ] evaluation.py çalıştırılmadan bu adım atlandı.")

    print(f"\n[TAMAMLANDI] Tüm hata analizleri '{OUTPUT_DIR}/' klasörüne kaydedildi.")
