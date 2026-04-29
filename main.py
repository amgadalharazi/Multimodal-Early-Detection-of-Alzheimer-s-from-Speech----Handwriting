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
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


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

    loaded_meta = []  # {name, feature_names, imputer, scaler, weight}
    sub_model_states = {}  # name → state_dict
    sub_model_networks = []  # nn.Module list (to build ModuleList)
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

            # Use saved AUC as weight so stronger models vote more
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
    total_w = sum(weights)

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
            # Everything test.py needs to run predictions
            "meta": loaded_meta,  # feature_names, imputer, scaler, weight per model
            "sub_model_states": sub_model_states,  # state_dict per model
            # Summary info
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


# ─────────────────────────────────────────────
# RADAR CHART COMPARISON (saved to plots/)
# ─────────────────────────────────────────────
def plot_radar_comparison(speech_metrics, hw_metrics, audio_metrics):
    """Radar chart comparing models across AUC, Accuracy, Precision, Recall, F1."""
    plots_dir = BASE_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_modalities = {
        "Speech+Clinical": speech_metrics,
        "Handwriting": hw_metrics,
        "Raw Audio": audio_metrics,
    }
    active = {name: m for name, m in all_modalities.items() if m is not None}
    if not active:
        print("  ⚠ No metrics available, skipping radar chart.")
        return

    metrics = ["auc", "accuracy", "precision", "recall", "f1"]
    labels = ["AUC", "Accuracy", "Precision", "Recall", "F1"]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]   # close the circle

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color='grey', fontsize=8)
    ax.yaxis.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

    colors = sns.color_palette("Set2", n_colors=len(active))
    for idx, (name, m) in enumerate(active.items()):
        values = [m.get(k, 0.0) for k in metrics]
        values += values[:1]   # close the circle
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
        ax.plot(angles, values, linewidth=2, color=colors[idx], label=name)
        # Annotate each point
        for angle, val in zip(angles[:-1], values[:-1]):
            ax.text(angle, val + 0.05, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=colors[idx], fontweight='bold')

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.title("Model Performance Radar — Multimodal Comparison",
              size=14, fontweight='bold', pad=20)
    plt.tight_layout()
    out_path = plots_dir / "radar_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved radar chart → {out_path}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Alzheimer's Detection Pipeline"
    )
    parser.add_argument(
        "--skip-speech", action="store_true", help="Skip NACC speech+clinical training"
    )
    parser.add_argument(
        "--skip-hw", action="store_true", help="Skip handwriting training"
    )
    parser.add_argument(
        "--skip-audio", action="store_true", help="Skip raw-audio training"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip ALL training — combine + predict only",
    )
    parser.add_argument(
        "--regen-hw",
        action="store_true",
        help="Force-regenerate the handwriting dataset",
    )
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
    hw_metrics = None
    audio_metrics = None

    # ── Step 1: NACC Speech + Clinical ──────────────────────────
    if args.skip_train or args.skip_speech:
        print("\n  Skipping NACC speech training — loading checkpoint …")
        speech_metrics = load_metrics("speech_model_complete.pt", "Speech+Clinical")
        print(
            f"  {'✓ loaded' if speech_metrics else '✗ not found — run without --skip-speech first'}"
        )
    else:
        speech_metrics = step_speech()

    # ── Step 2: Handwriting ──────────────────────────────────────
    if args.skip_train or args.skip_hw:
        print("\n  Skipping handwriting training — loading checkpoint …")
        hw_metrics = load_metrics("hw_model_complete.pt", "Handwriting")
        print(
            f"  {'✓ loaded' if hw_metrics else '✗ not found — run without --skip-hw first'}"
        )
    else:
        hw_metrics = step_hw(regen=args.regen_hw)

    # ── Step 3: Raw Audio ────────────────────────────────────────
    if args.skip_train or args.skip_audio:
        print("\n  Skipping audio training — loading checkpoint …")
        audio_metrics = load_metrics("speech_audio_model_complete.pt", "Raw Audio")
        print(
            f"  {'✓ loaded' if audio_metrics else '✗ not found — run without --skip-audio first'}"
        )
    else:
        audio_metrics = step_audio()

    # ── Step 4: Combine all into final_model.pt ──────────────────
    combined = step_combine()
    if not combined:
        print("\n  ⚠  Skipping predictions — no models available to combine.")
    else:
        # ── Step 5: Run predictions using final_model.pt ─────────
        step_test()

    # ── Comparison table + radar chart ───────────────────────────
    if any(m is not None for m in [speech_metrics, hw_metrics, audio_metrics]):
        print_comparison(speech_metrics, hw_metrics, audio_metrics)
        plot_radar_comparison(speech_metrics, hw_metrics, audio_metrics)

    # ── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print("\n  Checkpoints in result/:")
    print("    speech_model_complete.pt          ← NACC speech + clinical")
    print("    hw_model_complete.pt              ← handwriting")
    print("    speech_audio_model_complete.pt    ← raw audio (needs train/)")
    print(
        "    final_model.pt                    ← FINAL: weighted ensemble of all above"
    )
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