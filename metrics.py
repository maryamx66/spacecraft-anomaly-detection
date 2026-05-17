from __future__ import annotations
import numpy as np
from dataclasses import dataclass



from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)


class HFEvaluator:
    def __init__(self):
        self._predictions = []
        self._references  = []
        self._scores      = []

    def add_batch(self, predictions, references, scores=None):
        self._predictions.extend(np.asarray(predictions, dtype=int).tolist())
        self._references.extend( np.asarray(references,  dtype=int).tolist())
        if scores is not None:
            self._scores.extend(np.asarray(scores, dtype=float).tolist())

    def compute(self):
        y_pred = np.array(self._predictions, dtype=int)
        y_true = np.array(self._references,  dtype=int)
        results = {
            "accuracy" : float(accuracy_score( y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall"   : float(recall_score(   y_true, y_pred, zero_division=0)),
            "f1"       : float(f1_score(        y_true, y_pred, zero_division=0)),
        }
        if self._scores:
            y_scores = np.array(self._scores, dtype=float)
            try:
                results["roc_auc"] = float(roc_auc_score(y_true, y_scores))
            except ValueError:
                results["roc_auc"] = 0.0
        return results

    def reset(self):
        self._predictions.clear()
        self._references.clear()
        self._scores.clear()

    @staticmethod
    def compute_from_arrays(y_true, y_pred, y_scores=None):
        ev = HFEvaluator()
        ev.add_batch(predictions=y_pred, references=y_true, scores=y_scores)
        return ev.compute()

    @staticmethod
    def classification_report(y_true, y_pred, target_names=None):
        labels = target_names or ["Normal (0)", "Anomaly (1)"]
        return classification_report(y_true, y_pred, target_names=labels, zero_division=0)


def compute_reconstruction_errors(x_true, x_pred, metric="mse", reduce_channels=True):
    diff    = x_true - x_pred
    results = {}
    if metric in ("mse", "both"):
        sq = diff ** 2
        results["mse"] = sq.mean(axis=-1) if reduce_channels else sq
    if metric in ("mae", "both"):
        ab = np.abs(diff)
        results["mae"] = ab.mean(axis=-1) if reduce_channels else ab
    return results


def aggregate_errors_to_scalar(errors, method="mean"):
    if method == "mean": return errors.mean(axis=1)
    if method == "max":  return errors.max(axis=1)
    if method == "p95":  return np.percentile(errors, 95, axis=1)
    raise ValueError(f"Unknown method: {method}")


def nonparametric_dynamic_threshold(errors, window=30, z_score=3.0, min_percentile=0.95):
    N            = len(errors)
    thresholds   = np.zeros(N, dtype=np.float32)
    global_floor = np.percentile(errors, min_percentile * 100)
    for i in range(N):
        w = errors[max(0, i - window):i] if i > 0 else errors[:1]
        thresholds[i] = max(w.mean() + z_score * (w.std() + 1e-8), global_floor)
    return thresholds, errors > thresholds


def moving_average_threshold(errors, window=20, multiplier=2.5):
    N          = len(errors)
    thresholds = np.zeros(N, dtype=np.float32)
    for i in range(N):
        w = errors[max(0, i - window):i + 1]
        thresholds[i] = w.mean() + multiplier * (w.std() + 1e-8)
    return thresholds, errors > thresholds


@dataclass
class ClassificationMetrics:
    variant_name:   str
    accuracy:       float
    precision:      float
    recall:         float
    f1_score:       float
    tp: int;  fp: int;  tn: int;  fn: int
    roc_auc:        float = 0.0
    threshold_type: str   = "dynamic"
    best_val_loss:  float = 0.0

    @property
    def confusion_matrix(self):
        return np.array([[self.tn, self.fp], [self.fn, self.tp]])

    def as_dict(self):
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


def compute_classification_metrics(y_true, y_pred, y_scores=None,
                                    variant_name="model", threshold_type="dynamic",
                                    best_val_loss=0.0):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    ev = HFEvaluator()
    ev.add_batch(predictions=y_pred, references=y_true, scores=y_scores)
    results = ev.compute()
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return ClassificationMetrics(
        variant_name, results["accuracy"], results["precision"],
        results["recall"], results["f1"],
        tp, fp, tn, fn,
        results.get("roc_auc", 0.0), threshold_type, best_val_loss,
    )


def channel_wise_errors(x_true, x_pred):
    diff = x_true - x_pred
    mse  = (diff ** 2).mean(axis=(0, 1))
    mae  = np.abs(diff).mean(axis=(0, 1))
    return {"mse_per_channel": mse, "mae_per_channel": mae,
            "channel_ranking": np.argsort(mse)[::-1]}


def error_by_anomaly_class(errors, y_true):
    y = np.asarray(y_true, dtype=int)
    n = errors[y == 0]; a = errors[y == 1]
    return {
        "normal_mean_error" : float(n.mean()) if len(n) > 0 else 0.0,
        "anomaly_mean_error": float(a.mean()) if len(a) > 0 else 0.0,
        "separation_ratio"  : float(a.mean() / (n.mean() + 1e-8))
                              if len(n) > 0 and len(a) > 0 else 0.0,
    }


if __name__ == "__main__":
    print("=" * 58)
    print("  metrics.py — Demo Test")
    print("=" * 58)

    rng = np.random.default_rng(42)
    N, SEQ, C = 300, 50, 55
    x_true = rng.normal(0, 1, (N, SEQ, C)).astype(np.float32)
    x_pred = x_true + rng.normal(0, 0.1, (N, SEQ, C)).astype(np.float32)
    y_true = np.zeros(N, dtype=int)
    y_true[-60:] = 1
    x_pred[-60:] += rng.normal(0, 0.5, (60, SEQ, C))

    err_dict = compute_reconstruction_errors(x_true, x_pred, metric="both")
    errors   = aggregate_errors_to_scalar(err_dict["mse"])

    print(f"\n[1] Reconstruction Error")
    print(f"    Shape (N, seq_len)  : {err_dict['mse'].shape}")
    print(f"    Mean error — normal : {errors[y_true==0].mean():.5f}")
    print(f"    Mean error — anomaly: {errors[y_true==1].mean():.5f}")

    thr_dyn, y_pred_dyn = nonparametric_dynamic_threshold(errors)
    thr_ma,  y_pred_ma  = moving_average_threshold(errors)

    print(f"\n[2] Dynamic Thresholding")
    print(f"    Nonparametric  -> detected: {y_pred_dyn.sum():>4} / {N}")
    print(f"    Moving Average -> detected: {y_pred_ma.sum():>4} / {N}")

    ev = HFEvaluator()
    ev.add_batch(predictions=y_pred_dyn, references=y_true, scores=errors)
    results = ev.compute()

    print(f"\n[3] HFEvaluator Results")
    for k, v in results.items():
        print(f"    {k:<12}: {v:.4f}")

    m  = compute_classification_metrics(y_true, y_pred_dyn, y_scores=errors, variant_name="demo_model")
    cm = m.confusion_matrix

    print(f"\n[4] Confusion Matrix")
    print(f"                    Predicted Normal  Predicted Anomaly")
    print(f"    Actual Normal   {cm[0,0]:>16}  {cm[0,1]:>17}")
    print(f"    Actual Anomaly  {cm[1,0]:>16}  {cm[1,1]:>17}")

    print(f"\n[5] Classification Report")
    print(HFEvaluator.classification_report(y_true, y_pred_dyn))

    ch   = channel_wise_errors(x_true, x_pred)
    top3 = ch["channel_ranking"][:3]
    print(f"[6] Channel-wise Error (worst 3)")
    for rank, c in enumerate(top3, 1):
        print(f"    #{rank}  Channel {c:>2}  ->  MSE = {ch['mse_per_channel'][c]:.5f}")

    sep = error_by_anomaly_class(errors, y_true)
    print(f"\n[7] Anomaly / Normal Separation")
    print(f"    Normal  : {sep['normal_mean_error']:.5f}")
    print(f"    Anomaly : {sep['anomaly_mean_error']:.5f}")
    print(f"    Ratio   : {sep['separation_ratio']:.2f}x  {'Good v' if sep['separation_ratio'] > 2 else 'Weak x'}")

    print("\n" + "=" * 58)
    print("  All functions ran successfully.")
    print("=" * 58)
