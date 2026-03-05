"""
main.py
=======
Full pipeline orchestrator for Multimodal Alzheimer's Detection.

Steps:
  1. speech.py  → generate speech features (if needed) + train on NACC dataset
  2. hw.py      → generate fake handwriting dataset + train model
  3. test.py    → run predictions on 3 sample patients, compare modalities
  4. Print a side-by-side performance comparison table

Usage:
    python main.py                          # full run
    python main.py --skip-speech            # skip speech training
    python main.py --skip-hw                # skip handwriting training
    python main.py --skip-train             # skip both training steps (predict only)
    python main.py --regen-hw               # force-regenerate handwriting dataset
"""

import argparse
import time
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# BANNER
# ──────────────────────────────────────────────
def banner(text):
    line = "═" * 55
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


# ──────────────────────────────────────────────
# STEP 1 — SPEECH + CLINICAL MODEL
# ──────────────────────────────────────────────
def step_speech():
    banner("STEP 1 ▸ Speech + Clinical  (NACC dataset)")
    from speech import train_speech_model
    return train_speech_model()


# ──────────────────────────────────────────────
# STEP 2 — HANDWRITING MODEL
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# STEP 3 — PREDICTIONS
# ──────────────────────────────────────────────
def step_test():
    banner("STEP 3 ▸ Patient Predictions (all modalities)")
    from test import run_test
    return run_test()


# ──────────────────────────────────────────────
# COMPARISON TABLE
# ──────────────────────────────────────────────
def print_comparison(speech_metrics, hw_metrics):
    banner("MODALITY COMPARISON — TEST SET RESULTS")

    metrics_order = ["auc", "accuracy", "precision", "recall", "f1"]

    col_w   = 22
    label_w = 14
    print(f"\n  {'Metric':<{label_w}} {'Speech + Clinical':>{col_w}} {'Handwriting':>{col_w}}")
    print(f"  {'-'*label_w} {'-'*col_w} {'-'*col_w}")

    for m in metrics_order:
        sp_val = speech_metrics.get(m, float("nan"))
        hw_val = hw_metrics.get(m, float("nan"))

        if sp_val > hw_val:
            sp_str = f"{sp_val:.4f} <"
            hw_str = f"{hw_val:.4f}"
        elif hw_val > sp_val:
            sp_str = f"{sp_val:.4f}"
            hw_str = f"{hw_val:.4f} <"
        else:
            sp_str = f"{sp_val:.4f}"
            hw_str = f"{hw_val:.4f}"

        print(f"  {m.upper():<{label_w}} {sp_str:>{col_w}} {hw_str:>{col_w}}")

    print(f"\n  < = better score for that metric\n")

    sp_auc = speech_metrics.get("auc", 0)
    hw_auc = hw_metrics.get("auc", 0)
    diff   = abs(sp_auc - hw_auc)
    winner = "Speech + Clinical" if sp_auc >= hw_auc else "Handwriting"

    sp_f1  = speech_metrics.get("f1",     0)
    hw_f1  = hw_metrics.get("f1",         0)
    sp_rec = speech_metrics.get("recall", 0)
    hw_rec = hw_metrics.get("recall",     0)

    print(f"  Key observations:")
    print(f"  * Best ROC-AUC : {winner} ({max(sp_auc, hw_auc):.4f})")
    print(f"  * AUC gap      : {diff:.4f} — "
          + ("models perform similarly" if diff < 0.05 else "notable difference"))
    print(f"  * Best F1      : {'Speech + Clinical' if sp_f1 >= hw_f1 else 'Handwriting'} "
          f"({max(sp_f1, hw_f1):.4f})")
    print(f"  * Best Recall  : {'Speech + Clinical' if sp_rec >= hw_rec else 'Handwriting'} "
          f"({max(sp_rec, hw_rec):.4f})")
    print(f"    (recall = sensitivity — critical for early detection)\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Alzheimer's Detection Pipeline")
    parser.add_argument("--skip-speech", action="store_true",
                        help="Skip speech model training")
    parser.add_argument("--skip-hw",     action="store_true",
                        help="Skip handwriting model training")
    parser.add_argument("--skip-train",  action="store_true",
                        help="Skip ALL training steps")
    parser.add_argument("--regen-hw",    action="store_true",
                        help="Force-regenerate the handwriting dataset")
    args = parser.parse_args()

    t0 = time.time()

    banner("MULTIMODAL ALZHEIMER'S DETECTION — FULL PIPELINE")
    print("\n  Modalities:")
    print("    1. Speech + Clinical  (NACC real dataset)")
    print("    2. Handwriting        (synthetic dataset)")
    print("  Architecture: MultimodalNet — 4-layer MLP + BatchNorm + Dropout")

    speech_metrics = None
    hw_metrics     = None

    # ── Step 1: Speech ──
    if args.skip_train or args.skip_speech:
        print("\n  Skipping speech training — loading saved checkpoint ...")
        sp_path = RESULT_DIR / "speech_model_complete.pt"
        if sp_path.exists():
            import torch
            ckpt = torch.load(sp_path, weights_only=False)
            speech_metrics = ckpt.get("metrics", {
                "modality": "Speech + Clinical",
                "auc": float("nan"), "accuracy": float("nan"),
                "precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
            })
            print("  Speech checkpoint found")
        else:
            print("  No speech checkpoint — run without --skip-speech first")
    else:
        speech_metrics = step_speech()

    # ── Step 2: Handwriting ──
    if args.skip_train or args.skip_hw:
        print("\n  Skipping handwriting training — loading saved checkpoint ...")
        hw_path = RESULT_DIR / "hw_model_complete.pt"
        if hw_path.exists():
            import torch
            ckpt = torch.load(hw_path, weights_only=False)
            hw_metrics = ckpt.get("metrics", {
                "modality": "Handwriting",
                "auc": float("nan"), "accuracy": float("nan"),
                "precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
            })
            print("  Handwriting checkpoint found")
        else:
            print("  No handwriting checkpoint — run without --skip-hw first")
    else:
        hw_metrics = step_hw(regen=args.regen_hw)

    # ── Step 3: Predictions ──
    step_test()

    # ── Comparison ──
    if speech_metrics and hw_metrics:
        print_comparison(speech_metrics, hw_metrics)

    elapsed = time.time() - t0
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print("\n  Files created:")
    print("    Datasets/speech_features.csv")
    print("    Datasets/handwriting_dataset.csv")
    print("    result/speech_model_complete.pt")
    print("    result/hw_model_complete.pt")
    print()
    print("  Rerun without retraining:")
    print("    python main.py --skip-train")
    print()
    print("  NOTE: Research tool only. Always consult a medical professional.\n")


if __name__ == "__main__":
    main()