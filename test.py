"""
test.py
=======
Runs predictions using the final ensemble model (result/final_model.pt).
Falls back to loading individual .pt checkpoints if the final model is
not yet built.

Build the final model first (no retraining needed):
    python combine_models.py

Then predict:
    python test.py                     # 3 sample patients
    python test.py /path/to/audio.wav  # real audio file
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"


# ──────────────────────────────────────────────
# SHARED ARCHITECTURE
# ──────────────────────────────────────────────
class MultimodalNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,  32),        nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32,  1),
        )
    def forward(self, x):
        return self.net(x).squeeze()


# ──────────────────────────────────────────────
# LOAD FINAL ENSEMBLE MODEL
# ──────────────────────────────────────────────
def load_final_model():
    """
    Load result/final_model.pt built by combine_models.py.
    Returns a list of dicts:
        [{ name, model, feature_names, imputer, scaler, weight }, ...]
    """
    path = RESULT_DIR / "final_model.pt"
    if not path.exists():
        return None

    ckpt       = torch.load(path, map_location="cpu", weights_only=False)
    meta       = ckpt["meta"]
    sub_states = ckpt["sub_model_states"]

    sub_models = []
    for m in meta:
        model = MultimodalNet(input_dim=len(m["feature_names"]))
        model.load_state_dict(sub_states[m["name"]])
        model.eval()
        sub_models.append({
            "name":          m["name"],
            "model":         model,
            "feature_names": m["feature_names"],
            "imputer":       m["imputer"],
            "scaler":        m["scaler"],
            "weight":        m["weight"],
        })
    return sub_models


# ──────────────────────────────────────────────
# FALLBACK: load individual .pt files
# ──────────────────────────────────────────────
def load_individual_models():
    """Used only if final_model.pt does not exist."""
    checkpoints = [
        ("Speech+Clinical", "speech_model_complete.pt",       1.0),
        ("Handwriting",     "hw_model_complete.pt",           1.0),
        ("Raw Audio",       "speech_audio_model_complete.pt", 1.0),
    ]
    sub_models = []
    for name, fname, weight in checkpoints:
        path = RESULT_DIR / fname
        if not path.exists():
            continue
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        feats = ckpt["feature_names"]
        model = MultimodalNet(input_dim=len(feats))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        w = ckpt.get("metrics", {}).get("auc", weight)
        sub_models.append({
            "name":          name,
            "model":         model,
            "feature_names": feats,
            "imputer":       ckpt["imputer"],
            "scaler":        ckpt["scaler"],
            "weight":        w,
        })
    return sub_models


# ──────────────────────────────────────────────
# SINGLE PATIENT PREDICTION THROUGH ENSEMBLE
# ──────────────────────────────────────────────
def predict_patient(patient_data: dict, sub_models: list) -> dict:
    weighted_sum = 0.0
    weight_total = 0.0
    per_model    = {}

    for m in sub_models:
        try:
            row = np.array(
                [[patient_data.get(f, np.nan) for f in m["feature_names"]]],
                dtype=float,
            )
            row = m["scaler"].transform(m["imputer"].transform(row))
            t   = torch.tensor(row, dtype=torch.float32)
            with torch.no_grad():
                prob = torch.sigmoid(m["model"](t)).item()
            per_model[m["name"]] = prob
            weighted_sum += prob * m["weight"]
            weight_total += m["weight"]
        except Exception as e:
            per_model[m["name"]] = None
            print(f"    ⚠  {m['name']} failed: {e}")

    if weight_total == 0:
        raise RuntimeError("All sub-models failed.")

    final_prob = weighted_sum / weight_total
    return {
        "probability": final_prob,
        "label":       1 if final_prob >= 0.5 else 0,
        "per_model":   per_model,
    }


# ──────────────────────────────────────────────
# SAMPLE PATIENTS
# ──────────────────────────────────────────────
PATIENTS = [
    {
        "name": "Patient A — Likely Healthy",
        "NACCAGE": 65,  "SEX": 1,     "EDUC": 16,
        "MMSELOC": 29,  "CDRGLOB": 0,
        "MEMORY": 0,    "JUDGMENT": 0, "COMMUN": 0, "HOMEHOBB": 0, "PERSCARE": 0,
        "mfcc_mean": 0.1,    "mfcc_std": 1.0,
        "pitch_mean": 155.0, "pitch_std": 18.0,
        "speech_rate": 4.0,  "pause_rate": 0.7,
        "hnr_db": 18.5,      "jitter_pct": 0.45, "shimmer_pct": 2.8,
        "age_education": 65*16, "mmse_cdr": 29*0,
        "pitch_variance": 18/(155+1e-5),
        "pressure_mean": 0.66, "pressure_std": 0.09,
        "velocity_mean": 4.30, "velocity_std": 1.10,
        "acceleration_mean": 2.20, "jerk_mean": 0.75,
        "writing_duration": 17,    "num_pen_lifts": 11,
        "pen_up_time_pct": 0.16,   "inter_stroke_pause": 0.18,
        "letter_size_mean": 0.52,  "letter_size_std": 0.04,
        "word_spacing": 0.57,      "stroke_length": 2.60, "slant_angle": 12,
        "tremor_index": 0.07,      "direction_changes": 6, "pen_pressure_cv": 0.14,
        "clock_radius_std": 0.04,  "clock_digit_err": 0.2,
        "fluency_score": 4.30/(0.75+1e-5),
        "tremor_pressure": 0.07*0.09,
        "size_variability": 0.04/(0.52+1e-5),
        "pause_ratio": 0.16/(1-0.16+1e-5),
    },
    {
        "name": "Patient B — Likely Alzheimer's",
        "NACCAGE": 80,  "SEX": 2,     "EDUC": 8,
        "MMSELOC": 18,  "CDRGLOB": 2,
        "MEMORY": 3,    "JUDGMENT": 3, "COMMUN": 2, "HOMEHOBB": 3, "PERSCARE": 2,
        "mfcc_mean": -1.5,   "mfcc_std": 0.5,
        "pitch_mean": 120.0, "pitch_std": 30.0,
        "speech_rate": 2.0,  "pause_rate": 3.8,
        "hnr_db": 10.5,      "jitter_pct": 1.25, "shimmer_pct": 6.8,
        "age_education": 80*8, "mmse_cdr": 18*2,
        "pitch_variance": 30/(120+1e-5),
        "pressure_mean": 0.45, "pressure_std": 0.22,
        "velocity_mean": 2.00, "velocity_std": 2.10,
        "acceleration_mean": 0.85, "jerk_mean": 2.10,
        "writing_duration": 46,    "num_pen_lifts": 30,
        "pen_up_time_pct": 0.45,   "inter_stroke_pause": 0.80,
        "letter_size_mean": 0.32,  "letter_size_std": 0.14,
        "word_spacing": 0.28,      "stroke_length": 1.30, "slant_angle": 3,
        "tremor_index": 0.42,      "direction_changes": 14, "pen_pressure_cv": 0.40,
        "clock_radius_std": 0.22,  "clock_digit_err": 3.1,
        "fluency_score": 2.00/(2.10+1e-5),
        "tremor_pressure": 0.42*0.22,
        "size_variability": 0.14/(0.32+1e-5),
        "pause_ratio": 0.45/(1-0.45+1e-5),
    },
    {
        "name": "Patient C — Borderline / MCI",
        "NACCAGE": 72,  "SEX": 1,     "EDUC": 12,
        "MMSELOC": 24,  "CDRGLOB": 0.5,
        "MEMORY": 1,    "JUDGMENT": 1, "COMMUN": 1, "HOMEHOBB": 1, "PERSCARE": 0,
        "mfcc_mean": 0.3,    "mfcc_std": 0.9,
        "pitch_mean": 145.0, "pitch_std": 22.0,
        "speech_rate": 3.2,  "pause_rate": 1.6,
        "hnr_db": 14.5,      "jitter_pct": 0.80, "shimmer_pct": 4.5,
        "age_education": 72*12, "mmse_cdr": 24*0.5,
        "pitch_variance": 22/(145+1e-5),
        "pressure_mean": 0.56, "pressure_std": 0.15,
        "velocity_mean": 3.10, "velocity_std": 1.60,
        "acceleration_mean": 1.50, "jerk_mean": 1.30,
        "writing_duration": 28,    "num_pen_lifts": 19,
        "pen_up_time_pct": 0.30,   "inter_stroke_pause": 0.45,
        "letter_size_mean": 0.43,  "letter_size_std": 0.08,
        "word_spacing": 0.42,      "stroke_length": 1.90, "slant_angle": 7,
        "tremor_index": 0.20,      "direction_changes": 10, "pen_pressure_cv": 0.26,
        "clock_radius_std": 0.11,  "clock_digit_err": 1.5,
        "fluency_score": 3.10/(1.30+1e-5),
        "tremor_pressure": 0.20*0.15,
        "size_variability": 0.08/(0.43+1e-5),
        "pause_ratio": 0.30/(1-0.30+1e-5),
    },
]


# ──────────────────────────────────────────────
# RUN PREDICTIONS
# ──────────────────────────────────────────────
def run_test():
    final_path = RESULT_DIR / "final_model.pt"
    if final_path.exists():
        sub_models   = load_final_model()
        source_label = "final_model.pt  (fused ensemble)"
    else:
        sub_models   = load_individual_models()
        source_label = "individual checkpoints  (run combine_models.py to fuse)"

    if not sub_models:
        print("  ✗  No trained models found in result/")
        print("     Run:  python main.py           (to train)")
        print("     Then: python combine_models.py (to fuse into final_model.pt)")
        return []

    model_names = [m["name"] for m in sub_models]
    width       = 70

    print("\n" + "═" * width)
    print(f"  ALZHEIMER'S DETECTION  —  {source_label}")
    print("═" * width)
    print(f"  Active sub-models : {', '.join(model_names)}")
    print("─" * width)

    results = []
    for p in PATIENTS:
        result = predict_patient(p, sub_models)
        prob   = result["probability"]
        label  = result["label"]

        risk = (
            "🔴 HIGH"   if prob >= 0.75 else
            "🟡 MEDIUM" if prob >= 0.50 else
            "🟢 LOW"
        )
        conf = "High" if abs(prob - 0.5) > 0.3 else "Low — borderline"
        flag = "⚠️  Alzheimer's" if label == 1 else "✅ Healthy"

        print(f"\n  {p['name']}")
        for name, sub_prob in result["per_model"].items():
            if sub_prob is not None:
                sub_flag = "AD  " if sub_prob >= 0.5 else "Hlth"
                bar      = "█" * int(sub_prob * 20) + "░" * (20 - int(sub_prob * 20))
                print(f"    {name:<22} {sub_prob*100:5.1f}%  [{bar}]  {sub_flag}")
            else:
                print(f"    {name:<22}  N/A")

        print(f"    {'─'*55}")
        ens_bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        print(f"    {'ENSEMBLE FINAL':<22} {prob*100:5.1f}%  [{ens_bar}]  {flag}")
        print(f"    Risk: {risk}   |   Confidence: {conf}")

        valid_labels = [1 if v >= 0.5 else 0
                        for v in result["per_model"].values() if v is not None]
        if len(set(valid_labels)) > 1:
            print(f"    ⚡ Sub-models disagree — interpret with caution.")

        results.append({
            "patient":   p["name"],
            "per_model": result["per_model"],
            "ensemble":  prob,
            "label":     label,
        })

    print("\n" + "═" * width)
    print("  NOTE: Research tool only. Always consult a medical professional.")
    print("═" * width)
    return results


# ──────────────────────────────────────────────
# REAL AUDIO PREDICTION
# ──────────────────────────────────────────────
def predict_from_audio(audio_path):
    import librosa

    final_path = RESULT_DIR / "final_model.pt"
    sub_models = load_final_model() if final_path.exists() else load_individual_models()

    if not sub_models:
        print("No models found. Run main.py then combine_models.py first.")
        return {}

    y, sr    = librosa.load(audio_path, sr=16000)
    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    pitch    = librosa.yin(y, fmin=50, fmax=300)
    dur      = len(y) / sr
    onsets   = librosa.onset.onset_detect(y=y, sr=sr)
    ivls     = librosa.effects.split(y, top_db=30)
    non_sil  = np.sum(ivls[:, 1] - ivls[:, 0])

    patient_data = {
        "NACCAGE": 70,  "SEX": 1,     "EDUC": 12,
        "MMSELOC": 25,  "CDRGLOB": 0.5,
        "MEMORY": 1,    "JUDGMENT": 1, "COMMUN": 1, "HOMEHOBB": 1, "PERSCARE": 1,
        "mfcc_mean":   float(np.mean(mfcc)),
        "mfcc_std":    float(np.std(mfcc)),
        "pitch_mean":  float(np.mean(pitch)),
        "pitch_std":   float(np.std(pitch)),
        "speech_rate": len(onsets) / (dur + 1e-5),
        "pause_rate":  1 - (non_sil / len(y)),
        "hnr_db":      15.0,
        "jitter_pct":  1.0,
        "shimmer_pct": 4.0,
    }
    patient_data["age_education"]  = patient_data["NACCAGE"] * patient_data["EDUC"]
    patient_data["mmse_cdr"]       = patient_data["MMSELOC"] * patient_data["CDRGLOB"]
    patient_data["pitch_variance"] = patient_data["pitch_std"] / (patient_data["pitch_mean"] + 1e-5)

    result = predict_patient(patient_data, sub_models)

    print("\n" + "═" * 52)
    print("            REAL AUDIO PREDICTION")
    print("═" * 52)
    print(f"  File: {audio_path}\n")
    for name, prob in result["per_model"].items():
        if prob is not None:
            flag = "⚠️  AD likely" if prob >= 0.5 else "✅ Healthy  "
            print(f"  {name:<25} {prob*100:5.1f}%  {flag}")
    print(f"  {'─'*46}")
    ens  = result["probability"]
    flag = "⚠️  Alzheimer's likely" if result["label"] == 1 else "✅ Healthy"
    print(f"  {'ENSEMBLE FINAL':<25} {ens*100:5.1f}%  {flag}")
    print("═" * 52)
    return result


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", nargs="?",
                        help="Path to a .wav file for real audio prediction")
    args = parser.parse_args()

    if args.audio_file:
        predict_from_audio(args.audio_file)
    else:
        run_test()