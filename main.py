"""
main.py
=======
Full pipeline orchestrator for Multimodal Alzheimer's Detection.

Steps:
  1. speech.py       → NACC clinical + synthetic speech  → speech_model_complete.pt
  2. hw.py           → synthetic handwriting dataset     → hw_model_complete.pt
  3. speech_train.py → raw audio (librosa, 60+ features) → speech_audio_model_complete.pt
  4. Combine all trained models into one final ensemble  → final_model.pt
  5. test.py         → predictions on 3 sample patients using final_model.pt

Usage:
    python main.py                   # full run — train everything + combine
    python main.py --skip-speech     # skip NACC speech training
    python main.py --skip-hw         # skip handwriting training
    python main.py --skip-audio      # skip raw-audio training
    python main.py --skip-train      # skip ALL training, just combine + predict
    python main.py --regen-hw        # force-regenerate handwriting dataset
"""

import argparse
import subprocess
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
PLOTS_DIR  = BASE_DIR / "plots"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════
# SHARED MODEL ARCHITECTURE
# ══════════════════════════════════════════════
class MultimodalNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze()


# ══════════════════════════════════════════════
# BANNER
# ══════════════════════════════════════════════
def banner(text):
    line = "═" * 60
    print(f"\n{line}\n  {text}\n{line}")


# ══════════════════════════════════════════════
# HELPER — load metrics from a saved checkpoint
# ══════════════════════════════════════════════
def load_metrics(ckpt_name, fallback_modality):
    path = RESULT_DIR / ckpt_name
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get(
        "metrics",
        {
            "modality": fallback_modality,
            "auc": float("nan"),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        },
    )


# ══════════════════════════════════════════════
# STEP 1 — NACC SPEECH + CLINICAL
# ══════════════════════════════════════════════
def step_speech():
    banner("STEP 1 ▸ Speech + Clinical  (NACC dataset)")
    from speech import train_speech_model
    return train_speech_model()


# ══════════════════════════════════════════════
# STEP 2 — HANDWRITING
# ══════════════════════════════════════════════
def step_hw(regen=False):
    banner("STEP 2 ▸ Handwriting  (synthetic dataset)")
    from hw import generate_handwriting_dataset, train_hw_model, DATA_DIR

    hw_csv = DATA_DIR / "handwriting_dataset.csv"
    if regen or not hw_csv.exists():
        print("  Generating handwriting dataset ...")
        generate_handwriting_dataset(save=True)
    else:
        print(f"  Reusing existing dataset: {hw_csv}")
    return train_hw_model()


# ══════════════════════════════════════════════
# STEP 3 — RAW AUDIO  (speech_train.py)
# ══════════════════════════════════════════════
def step_audio():
    banner("STEP 3 ▸ Raw Audio  (librosa 60+ features + MultimodalNet)")
    train_dir = BASE_DIR / "train"
    if not train_dir.exists():
        print("  ⚠  'train/' folder not found — skipping audio model.")
        print("     Create train/healthy/ and train/Alzheimer/ with .wav files.")
        return None
    print("  Running speech_train.py …")
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "speech_train.py")], text=True
    )
    if result.returncode != 0:
        print("  ⚠  speech_train.py exited with an error — audio model skipped.")
        return None
    metrics = load_metrics("speech_audio_model_complete.pt", "Raw Audio")
    if metrics is None:
        print("  ⚠  speech_audio_model_complete.pt not found after training.")
    return metrics


# ══════════════════════════════════════════════
# STEP 4 — COMBINE ALL MODELS → final_model.pt
# ══════════════════════════════════════════════
CHECKPOINTS = [
    ("Speech+Clinical", "speech_model_complete.pt"),
    ("Handwriting", "hw_model_complete.pt"),
    ("Raw Audio", "speech_audio_model_complete.pt"),
]


def step_combine():
    banner("STEP 4 ▸ Combining all models → final_model.pt")

    loaded_meta = []
    sub_model_states = {}
    sub_model_networks = []
    all_sub_metrics = {}

    for name, fname in CHECKPOINTS:
        path = RESULT_DIR / fname
        if not path.exists():
            print(f"  ✗  {name:<25} — {fname} not found, skipping.")
            continue
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            feature_names = ckpt["feature_names"]
            metrics = ckpt.get("metrics", {})

            weight = metrics.get("auc", 1.0)
            if weight != weight:  # NaN guard
                weight = 1.0

            model = MultimodalNet(input_dim=len(feature_names))
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            loaded_meta.append(
                {
                    "name": name,
                    "feature_names": feature_names,
                    "imputer": ckpt["imputer"],
                    "scaler": ckpt["scaler"],
                    "weight": weight,
                }
            )
            sub_model_states[name] = ckpt["model_state_dict"]
            sub_model_networks.append(model)
            all_sub_metrics[name] = metrics

            print(
                f"  ✓  {name:<25} "
                f"{len(feature_names):>3} features  |  "
                f"AUC {metrics.get('auc', float('nan')):.4f}  |  "
                f"weight {weight:.4f}"
            )

        except Exception as e:
            print(f"  ✗  {name:<25} — failed: {e}")

    if not loaded_meta:
        print("\n  No models could be loaded — train at least one model first.")
        return False

    # ── Compute weighted-average summary metrics ─────────────────
    weights = [m["weight"] for m in loaded_meta]

    def wavg(key):
        vals = [
            all_sub_metrics.get(m["name"], {}).get(key, float("nan"))
            for m in loaded_meta
        ]
        valid = [(v, w) for v, w in zip(vals, weights) if v == v]
        if not valid:
            return float("nan")
        return sum(v * w for v, w in valid) / sum(w for _, w in valid)

    summary_metrics = {
        "modality": "Ensemble (all modalities)",
        "auc": wavg("auc"),
        "accuracy": wavg("accuracy"),
        "precision": wavg("precision"),
        "recall": wavg("recall"),
        "f1": wavg("f1"),
        "n_models": len(loaded_meta),
        "sub_model_names": [m["name"] for m in loaded_meta],
    }

    # ── Save final_model.pt ──────────────────────────────────────
    out_path = RESULT_DIR / "final_model.pt"
    torch.save(
        {
            "meta": loaded_meta,
            "sub_model_states": sub_model_states,
            "metrics": summary_metrics,
            "sub_metrics": all_sub_metrics,
            "n_sub_models": len(loaded_meta),
            "sub_model_names": [m["name"] for m in loaded_meta],
        },
        out_path,
    )

    print(
        f"\n  ✓  Saved → result/final_model.pt  "
        f"({len(loaded_meta)} sub-model{'s' if len(loaded_meta) > 1 else ''})"
    )
    print(f"\n  Weighted-average metrics across all sub-models:")
    for k, v in summary_metrics.items():
        if k not in ("modality", "sub_model_names", "n_models"):
            print(f"    {k.upper():<12}: {v:.4f}")

    return True


# ══════════════════════════════════════════════
# STEP 5 — PREDICTIONS  (test.py)
# ══════════════════════════════════════════════
def step_test():
    banner("STEP 5 ▸ Patient Predictions  (final_model.pt)")
    from test import run_test
    return run_test()


# ══════════════════════════════════════════════
# 3-WAY COMPARISON TABLE
# ══════════════════════════════════════════════
def fmt(val, best_val):
    if val != val:
        return f"{'N/A':>12}"
    marker = " ◀" if abs(val - best_val) < 1e-9 else "  "
    return f"{val:.4f}{marker}"


def print_comparison(speech_metrics, hw_metrics, audio_metrics):
    banner("MODALITY COMPARISON — TEST SET RESULTS")

    all_modalities = {
        "Speech+Clinical": speech_metrics,
        "Handwriting": hw_metrics,
        "Raw Audio": audio_metrics,
    }
    active = {name: m for name, m in all_modalities.items() if m is not None}
    if not active:
        print("  No metrics available.")
        return

    metrics_order = ["auc", "accuracy", "precision", "recall", "f1"]
    col_w, label_w = 14, 12

    header = f"\n  {'Metric':<{label_w}}"
    sep = f"  {'-'*label_w}"
    for name in active:
        header += f"  {name:>{col_w}}"
        sep += f"  {'-'*col_w}"
    print(header)
    print(sep)

    for m in metrics_order:
        vals = {n: d.get(m, float("nan")) for n, d in active.items()}
        valid = [v for v in vals.values() if v == v]
        best = max(valid) if valid else float("nan")
        row = f"  {m.upper():<{label_w}}"
        for name in active:
            row += f"  {fmt(vals[name], best):>{col_w}}"
        print(row)

    print(f"\n  ◀ = best score for that metric\n")

    def best_of(metric):
        vals = {n: m.get(metric, float("nan")) for n, m in active.items()}
        valid = {n: v for n, v in vals.items() if v == v}
        if not valid:
            return "N/A", float("nan")
        w = max(valid, key=valid.get)
        return w, valid[w]

    print("  Key observations:")
    for metric, label in [
        ("auc", "Best ROC-AUC"),
        ("f1", "Best F1     "),
        ("recall", "Best Recall "),
    ]:
        name, val = best_of(metric)
        note = (
            " ← sensitivity, critical for early detection" if metric == "recall" else ""
        )
        print(f"  * {label} : {name}  ({val:.4f}){note}")

    auc_vals = {n: m.get("auc", float("nan")) for n, m in active.items()}
    valid_auc = {n: v for n, v in auc_vals.items() if v == v}
    if len(valid_auc) >= 2:
        spread = max(valid_auc.values()) - min(valid_auc.values())
        note = (
            "models perform similarly"
            if spread < 0.05
            else "notable spread between modalities"
        )
        print(f"  * AUC spread   : {spread:.4f} — {note}")
    print()


# ══════════════════════════════════════════════════════════
# FIX 1 — RADAR CHART  (handles None modalities correctly)
# ══════════════════════════════════════════════════════════
def plot_radar_comparison(speech_metrics, hw_metrics, audio_metrics):
    """Radar chart comparing available models across AUC, Accuracy, Precision, Recall, F1."""

    all_modalities = {
        "Speech+Clinical": speech_metrics,
        "Handwriting": hw_metrics,
        "Raw Audio": audio_metrics,
    }
    # Only include modalities that are not None AND have at least one real (non-NaN) metric
    active = {
        name: m for name, m in all_modalities.items()
        if m is not None and any(
            isinstance(m.get(k), float) and m.get(k) == m.get(k)
            for k in ["auc", "accuracy", "precision", "recall", "f1"]
        )
    }

    if not active:
        print("  ⚠ No valid metrics available — skipping radar chart.")
        return

    metric_keys   = ["auc", "accuracy", "precision", "recall", "f1"]
    metric_labels = ["AUC", "Accuracy", "Precision", "Recall", "F1"]
    n = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", fontsize=9)
    ax.yaxis.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.7)

    palette = sns.color_palette("Set2", n_colors=max(len(active), 1))

    for idx, (name, m) in enumerate(active.items()):
        values = [float(m.get(k, 0.0) or 0.0) for k in metric_keys]
        values += values[:1]   # close the polygon

        color = palette[idx]
        ax.fill(angles, values, alpha=0.15, color=color)
        ax.plot(angles, values, linewidth=2.5, color=color, label=name)

        # Value annotations at each vertex
        for angle, val in zip(angles[:-1], values[:-1]):
            ax.text(
                angle, min(val + 0.07, 1.05),
                f"{val:.2f}",
                ha="center", va="center",
                fontsize=9, color=color, fontweight="bold",
            )

    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=11)
    plt.title(
        "Model Performance Radar — Multimodal Comparison",
        size=14, fontweight="bold", pad=25,
    )
    plt.tight_layout()

    out_path = PLOTS_DIR / "radar_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved radar chart → {out_path}")


# ══════════════════════════════════════════════════════════
# FIX 2 — BAR CHART  (was completely missing / broken)
# ══════════════════════════════════════════════════════════
def plot_metrics_comparison(speech_metrics, hw_metrics, audio_metrics):
    """
    Grouped bar chart comparing all available modalities across 5 metrics.
    Skips modalities where metrics are None or all-NaN.
    """
    all_modalities = {
        "Speech+Clinical": speech_metrics,
        "Handwriting":     hw_metrics,
        "Raw Audio":       audio_metrics,
    }
    metric_keys   = ["auc", "accuracy", "precision", "recall", "f1"]
    metric_labels = ["AUC", "Accuracy", "Precision", "Recall", "F1"]

    # Filter to modalities with real data
    active = {}
    for name, m in all_modalities.items():
        if m is None:
            continue
        vals = [m.get(k, float("nan")) for k in metric_keys]
        if any(v == v for v in vals):   # at least one non-NaN
            active[name] = [v if (v == v) else 0.0 for v in vals]

    if not active:
        print("  ⚠ No valid metrics available — skipping bar chart.")
        return

    n_groups  = len(metric_labels)
    n_bars    = len(active)
    bar_width = 0.7 / n_bars
    x         = np.arange(n_groups)
    palette   = sns.color_palette("Set2", n_colors=max(n_bars, 1))

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (name, values) in enumerate(active.items()):
        offset = (i - n_bars / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width,
            label=name, color=palette[i], edgecolor="white", linewidth=0.6,
        )
        # Value label on top of each bar
        for bar, val in zip(bars, values):
            if val > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=8, color="black",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Performance Comparison by Modality", fontsize=14, fontweight="bold", pad=15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, loc="lower right")
    sns.despine(ax=ax)

    plt.tight_layout()
    out_path = PLOTS_DIR / "metrics_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved bar chart       → {out_path}")


# ══════════════════════════════════════════════════════════
# NEW VIZ 1 — METRICS HEATMAP
# ══════════════════════════════════════════════════════════
def plot_metrics_heatmap(speech_metrics, hw_metrics, audio_metrics):
    """
    Seaborn annotated heatmap: rows = modalities, cols = metrics.
    Colour-encodes score magnitude (vmin=0.5 so the scale is meaningful).
    """
    all_modalities = {
        "Speech+Clinical": speech_metrics,
        "Handwriting":     hw_metrics,
        "Raw Audio":       audio_metrics,
    }
    metric_keys   = ["auc", "accuracy", "precision", "recall", "f1"]
    metric_labels = ["AUC", "Accuracy", "Precision", "Recall", "F1"]

    active = {name: m for name, m in all_modalities.items() if m is not None}
    if not active:
        print("  ⚠ No valid metrics — skipping metrics heatmap.")
        return

    data = []
    for name, m in active.items():
        row = [float(m.get(k, np.nan) or np.nan) for k in metric_keys]
        data.append(row)

    df = pd.DataFrame(data, index=list(active.keys()), columns=metric_labels)

    fig, ax = plt.subplots(figsize=(10, max(3, len(active) * 1.6)))
    sns.heatmap(
        df, annot=True, fmt=".3f",
        cmap=sns.color_palette("YlOrRd", as_cmap=True),
        linewidths=0.8, linecolor="white",
        vmin=0.5, vmax=1.0,
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
        cbar_kws={"shrink": 0.75, "label": "Score", "pad": 0.02},
    )
    ax.set_title(
        "Performance Metrics Heatmap — All Modalities",
        fontsize=15, fontweight="bold", pad=18,
    )
    ax.set_xlabel("Metric", fontsize=12, labelpad=8)
    ax.set_ylabel("Modality", fontsize=12, labelpad=8)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12, rotation=0)
    plt.tight_layout()

    out_path = PLOTS_DIR / "metrics_heatmap.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved metrics heatmap      → {out_path}")


# ══════════════════════════════════════════════════════════
# NEW VIZ 2 — SIMULATED CONFUSION MATRICES
# ══════════════════════════════════════════════════════════
def _cm_from_metrics(m, n_per_class=500):
    """Reconstruct a 2×2 confusion matrix from precision / recall."""
    recall    = float(m.get("recall",    0.0) or 0.0)
    precision = float(m.get("precision", 1e-9) or 1e-9)
    tp = int(round(recall * n_per_class))
    fn = n_per_class - tp
    fp = int(round(tp * (1.0 - precision) / precision))
    tn = max(0, n_per_class - fp)
    return np.array([[tn, fp], [fn, tp]])


def plot_simulated_confusion_matrices(speech_metrics, hw_metrics, audio_metrics):
    """
    Side-by-side seaborn heatmap confusion matrices for every available model.
    Values are reconstructed from precision / recall (500 samples per class).
    """
    all_modalities = {
        "Speech+Clinical": speech_metrics,
        "Handwriting":     hw_metrics,
        "Raw Audio":       audio_metrics,
    }
    active = {name: m for name, m in all_modalities.items() if m is not None}
    if not active:
        print("  ⚠ No valid metrics — skipping confusion matrices.")
        return

    n      = len(active)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8))
    if n == 1:
        axes = [axes]

    class_labels = ["Healthy", "AD"]

    for ax, (name, m) in zip(axes, active.items()):
        cm = _cm_from_metrics(m)
        # Normalise to percentages for the colour scale
        cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(
            cm_pct,
            annot=np.array([[f"{v}\n({p:.0f}%)" for v, p in zip(row_v, row_p)]
                            for row_v, row_p in zip(cm, cm_pct)]),
            fmt="", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels,
            linewidths=1.2, linecolor="white",
            annot_kws={"size": 12, "weight": "bold"},
            ax=ax, cbar=False,
            vmin=0, vmax=100,
        )
        ax.set_title(f"{name}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual",    fontsize=11)

        recall    = m.get("recall",    0)
        precision = m.get("precision", 0)
        f1        = m.get("f1",        0)
        ax.text(
            0.5, -0.14,
            f"Recall {recall:.3f}  |  Precision {precision:.3f}  |  F1 {f1:.3f}",
            ha="center", transform=ax.transAxes,
            fontsize=9, color="#555",
        )

    fig.suptitle(
        "Simulated Confusion Matrices  (500 samples per class)",
        fontsize=14, fontweight="bold", y=1.04,
    )
    plt.tight_layout()
    out_path = PLOTS_DIR / "confusion_matrices.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved confusion matrices   → {out_path}")


# ══════════════════════════════════════════════════════════
# NEW VIZ 3 — SCORE DISTRIBUTION KDE PLOTS
# ══════════════════════════════════════════════════════════
def _simulate_scores(auc, n=800, seed=0):
    """
    Generate synthetic Healthy / AD probability scores that approximate
    the given AUC.  Higher AUC → better separation between classes.
    """
    rng = np.random.RandomState(seed)
    sep = max(0.05, min(3.5, (auc - 0.5) * 7.0))
    healthy = np.clip(rng.normal(0.50 - sep * 0.12, 0.14, n), 0.01, 0.99)
    ad      = np.clip(rng.normal(0.50 + sep * 0.12, 0.14, n), 0.01, 0.99)
    return healthy, ad


def plot_score_distributions(speech_metrics, hw_metrics, audio_metrics):
    """
    KDE plots of simulated prediction-score distributions for each model,
    coloured by class (Healthy vs AD).  The vertical dashed line marks the
    default 0.5 decision threshold.
    """
    all_modalities = {
        "Speech+Clinical": speech_metrics,
        "Handwriting":     hw_metrics,
        "Raw Audio":       audio_metrics,
    }
    active = {name: m for name, m in all_modalities.items() if m is not None}
    if not active:
        print("  ⚠ No valid metrics — skipping score distributions.")
        return

    n     = len(active)
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 4.8), sharey=False)
    if n == 1:
        axes = [axes]

    pal = {"Healthy": "#27ae60", "AD": "#e74c3c"}

    for ax, (name, m) in zip(axes, active.items()):
        auc = float(m.get("auc", 0.75) or 0.75)
        healthy_sc, ad_sc = _simulate_scores(auc, seed=hash(name) % 997)

        df_plot = pd.DataFrame({
            "Score": np.concatenate([healthy_sc, ad_sc]),
            "Group": ["Healthy"] * len(healthy_sc) + ["AD"] * len(ad_sc),
        })

        sns.kdeplot(
            data=df_plot, x="Score", hue="Group",
            palette=pal, fill=True, alpha=0.30,
            linewidth=2.2, bw_adjust=0.9, ax=ax,
        )
        ax.axvline(0.5, color="#555", linestyle="--", linewidth=1.4,
                   alpha=0.75, label="Threshold = 0.5")
        ax.set_title(
            f"{name}\nAUC = {auc:.4f}",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("Predicted P(AD)", fontsize=11)
        ax.set_ylabel("Density",         fontsize=11)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9, framealpha=0.85)
        sns.despine(ax=ax)

        # Shaded "FP zone" and "FN zone" hints
        ax.axvspan(0.5, 1.0, alpha=0.04, color="#e74c3c")
        ax.axvspan(0.0, 0.5, alpha=0.04, color="#27ae60")

    fig.suptitle(
        "Simulated Score Distributions — Healthy vs AD",
        fontsize=14, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    out_path = PLOTS_DIR / "score_distributions.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved score distributions  → {out_path}")


# ══════════════════════════════════════════════════════════
# NEW VIZ 4 — ENSEMBLE WEIGHT BREAKDOWN
# ══════════════════════════════════════════════════════════
def plot_ensemble_weights(speech_metrics, hw_metrics, audio_metrics):
    """
    Horizontal bar chart showing each model's AUC-based contribution to the
    weighted ensemble, with individual AUC scores annotated.
    """
    all_modalities = {
        "Speech+Clinical": speech_metrics,
        "Handwriting":     hw_metrics,
        "Raw Audio":       audio_metrics,
    }
    raw = {
        name: float(m.get("auc", np.nan) or np.nan)
        for name, m in all_modalities.items()
        if m is not None
    }
    active = {k: v for k, v in raw.items() if not np.isnan(v)}
    if not active:
        print("  ⚠ No valid AUC values — skipping ensemble weight chart.")
        return

    total   = sum(active.values())
    names   = list(active.keys())
    aucs    = [active[n] for n in names]
    pcts    = [v / total * 100 for v in aucs]
    palette = sns.color_palette("Set2", len(names))

    fig, ax = plt.subplots(figsize=(10, max(2.8, len(names) * 1.3)))
    bars = ax.barh(names, pcts, color=palette, edgecolor="white",
                   height=0.55, linewidth=0)

    # Gradient-like effect by overlaying a semi-transparent white strip
    for bar in bars:
        ax.barh(
            bar.get_y() + bar.get_height() / 2,
            bar.get_width(), bar.get_height(),
            left=bar.get_x(), color="white", alpha=0.15,
        )

    for bar, auc, pct in zip(bars, aucs, pcts):
        ax.text(
            pct + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"AUC {auc:.4f}  ({pct:.1f}%)",
            va="center", ha="left",
            fontsize=11, color="#222",
        )

    ax.set_xlabel("Contribution to Ensemble Weight (%)", fontsize=12, labelpad=8)
    ax.set_xlim(0, max(pcts) * 1.38)
    ax.set_title(
        "Ensemble Model Weight Breakdown  (AUC-based weighting)",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.tick_params(axis="y", labelsize=12)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=True)
    plt.tight_layout()

    out_path = PLOTS_DIR / "ensemble_weights.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved ensemble weights     → {out_path}")


# ══════════════════════════════════════════════════════════
# NEW VIZ 5 — HANDWRITING FEATURE CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════
def plot_hw_correlation():
    """
    Lower-triangle seaborn heatmap of Pearson correlations between all
    handwriting features.  Loaded from Datasets/handwriting_dataset.csv
    if present; skipped gracefully otherwise.
    """
    hw_csv = BASE_DIR / "Datasets" / "handwriting_dataset.csv"
    if not hw_csv.exists():
        print("  ⚠ handwriting_dataset.csv not found — skipping correlation heatmap.")
        return

    df_hw = pd.read_csv(hw_csv, nrows=20_000)   # sample for speed
    feat_cols = [c for c in df_hw.columns if c != "label"]
    corr      = df_hw[feat_cols].corr()

    # Shorten long names for readability
    short = {c: c.replace("_mean", "\n_mean")
                 .replace("_std",  "\n_std")
                 .replace("_pct",  "\n_pct")
             for c in feat_cols}
    corr.index   = [short.get(c, c) for c in corr.index]
    corr.columns = [short.get(c, c) for c in corr.columns]

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # hide upper triangle

    fig, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(
        corr,
        mask=mask,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0, vmin=-1, vmax=1,
        square=True,
        linewidths=0.3, linecolor="#ddd",
        annot=True, fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.70, "label": "Pearson r", "pad": 0.02},
        ax=ax,
    )
    ax.set_title(
        "Handwriting Feature Correlation Matrix\n"
        "(lower triangle · Healthy + AD combined)",
        fontsize=14, fontweight="bold", pad=18,
    )
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    plt.tight_layout()

    out_path = PLOTS_DIR / "hw_correlation.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved correlation heatmap  → {out_path}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Alzheimer's Detection Pipeline"
    )
    parser.add_argument("--skip-speech", action="store_true",
                        help="Skip NACC speech+clinical training")
    parser.add_argument("--skip-hw", action="store_true",
                        help="Skip handwriting training")
    parser.add_argument("--skip-audio", action="store_true",
                        help="Skip raw-audio training")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip ALL training — combine + predict only")
    parser.add_argument("--regen-hw", action="store_true",
                        help="Force-regenerate the handwriting dataset")
    args = parser.parse_args()

    t0 = time.time()

    banner("MULTIMODAL ALZHEIMER'S DETECTION — FULL PIPELINE")
    print("\n  Modalities:")
    print("    1. Speech + Clinical   (NACC real dataset)")
    print("    2. Handwriting         (synthetic dataset)")
    print("    3. Raw Audio           (librosa 60+ acoustic features)")
    print("  Architecture : MultimodalNet — 4-layer MLP + BatchNorm + Dropout")
    print("  Final output : result/final_model.pt  (weighted ensemble of all)")

    speech_metrics = None
    hw_metrics     = None
    audio_metrics  = None

    # ── Step 1: NACC Speech + Clinical ──────────────────────────
    if args.skip_train or args.skip_speech:
        print("\n  Skipping NACC speech training — loading checkpoint …")
        speech_metrics = load_metrics("speech_model_complete.pt", "Speech+Clinical")
        print(f"  {'✓ loaded' if speech_metrics else '✗ not found — run without --skip-speech first'}")
    else:
        speech_metrics = step_speech()

    # ── Step 2: Handwriting ──────────────────────────────────────
    if args.skip_train or args.skip_hw:
        print("\n  Skipping handwriting training — loading checkpoint …")
        hw_metrics = load_metrics("hw_model_complete.pt", "Handwriting")
        print(f"  {'✓ loaded' if hw_metrics else '✗ not found — run without --skip-hw first'}")
    else:
        hw_metrics = step_hw(regen=args.regen_hw)

    # ── Step 3: Raw Audio ────────────────────────────────────────
    if args.skip_train or args.skip_audio:
        print("\n  Skipping audio training — loading checkpoint …")
        audio_metrics = load_metrics("speech_audio_model_complete.pt", "Raw Audio")
        print(f"  {'✓ loaded' if audio_metrics else '✗ not found — run without --skip-audio first'}")
    else:
        audio_metrics = step_audio()

    # ── Step 4: Combine all into final_model.pt ──────────────────
    combined = step_combine()
    if not combined:
        print("\n  ⚠  Skipping predictions — no models available to combine.")
    else:
        step_test()

    # ── Comparison table + charts ─────────────────────────────────
    if any(m is not None for m in [speech_metrics, hw_metrics, audio_metrics]):
        print_comparison(speech_metrics, hw_metrics, audio_metrics)

        banner("SAVING VISUALISATIONS")
        plot_radar_comparison(speech_metrics, hw_metrics, audio_metrics)
        plot_metrics_comparison(speech_metrics, hw_metrics, audio_metrics)
        plot_metrics_heatmap(speech_metrics, hw_metrics, audio_metrics)
        plot_simulated_confusion_matrices(speech_metrics, hw_metrics, audio_metrics)
        plot_score_distributions(speech_metrics, hw_metrics, audio_metrics)
        plot_ensemble_weights(speech_metrics, hw_metrics, audio_metrics)
        plot_hw_correlation()

    # ── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print("\n  Checkpoints in result/:")
    print("    speech_model_complete.pt          ← NACC speech + clinical")
    print("    hw_model_complete.pt              ← handwriting")
    print("    speech_audio_model_complete.pt    ← raw audio (needs train/)")
    print("    final_model.pt                    ← FINAL: weighted ensemble of all above")
    print()
    print("  Plots saved to plots/:")
    print("    radar_comparison.png       ← polar metric radar chart")
    print("    metrics_comparison.png     ← grouped bar chart")
    print("    metrics_heatmap.png        ← annotated metric heatmap  [NEW]")
    print("    confusion_matrices.png     ← per-model confusion heatmaps  [NEW]")
    print("    score_distributions.png    ← KDE score distributions  [NEW]")
    print("    ensemble_weights.png       ← AUC-based weight breakdown  [NEW]")
    print("    hw_correlation.png         ← handwriting feature correlations  [NEW]")
    print()
    print("  Re-run without retraining (combine + predict only):")
    print("    python main.py --skip-train")
    print()
    print("  Skip audio only (no train/ folder required):")
    print("    python main.py --skip-audio")
    print()
    print("  NOTE: Research tool only. Always consult a medical professional.\n")


if __name__ == "__main__":
    main()