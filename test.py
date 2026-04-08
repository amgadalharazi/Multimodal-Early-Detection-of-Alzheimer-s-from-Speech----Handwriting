"""
test.py
=======
Loads both trained models (speech+clinical, handwriting) and runs predictions
on either:
  - 3 sample patients (no audio required), or
  - a real audio file (speech+clinical model only).

Usage:
    python test.py                     # run on sample patients
    python test.py /path/to/audio.wav  # predict from real audio
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import librosa

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"


# ──────────────────────────────────────────────
# MODEL DEFINITION (shared)
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# LOAD A CHECKPOINT
# ──────────────────────────────────────────────
def load_model(ckpt_name):
    path = RESULT_DIR / ckpt_name
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\nRun main.py (or train the model) first."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    feature_names = ckpt["feature_names"]
    imputer = ckpt["imputer"]
    scaler = ckpt["scaler"]

    model = MultimodalNet(input_dim=len(feature_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, feature_names, imputer, scaler


# ──────────────────────────────────────────────
# PREDICT — speech/clinical model
# ──────────────────────────────────────────────
def predict_speech(patient_data, model, feature_names, imputer, scaler):
    # Build row in the exact order of feature_names
    row = np.array([[patient_data.get(f, np.nan) for f in feature_names]], dtype=float)

    # Re-compute engineered features (if they exist in feature_names)
    age = patient_data.get("NACCAGE", np.nan)
    educ = patient_data.get("EDUC", np.nan)
    mmse = patient_data.get("MMSELOC", np.nan)
    cdr = patient_data.get("CDRGLOB", np.nan)
    pmean = patient_data.get("pitch_mean", np.nan)
    pstd = patient_data.get("pitch_std", np.nan)

    for feat, val in [
        ("age_education", age * educ),
        ("mmse_cdr", mmse * cdr),
        ("pitch_variance", pstd / (pmean + 1e-5)),
    ]:
        if feat in feature_names:
            row[0][feature_names.index(feat)] = val

    row = scaler.transform(imputer.transform(row))
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(row, dtype=torch.float32))).item()
    return 1 if prob >= 0.5 else 0, prob


# ──────────────────────────────────────────────
# PREDICT — handwriting model
# ──────────────────────────────────────────────
def predict_hw(patient_data, model, feature_names, imputer, scaler):
    row = np.array([[patient_data.get(f, np.nan) for f in feature_names]], dtype=float)

    vel = patient_data.get("velocity_mean", np.nan)
    jerk = patient_data.get("jerk_mean", np.nan)
    trem = patient_data.get("tremor_index", np.nan)
    pstd = patient_data.get("pressure_std", np.nan)
    lsize = patient_data.get("letter_size_mean", np.nan)
    lstd = patient_data.get("letter_size_std", np.nan)
    pup = patient_data.get("pen_up_time_pct", np.nan)

    for feat, val in [
        ("fluency_score", vel / (jerk + 1e-5)),
        ("tremor_pressure", trem * pstd),
        ("size_variability", lstd / (lsize + 1e-5)),
        ("pause_ratio", pup / (1 - pup + 1e-5)),
    ]:
        if feat in feature_names:
            row[0][feature_names.index(feat)] = val

    row = scaler.transform(imputer.transform(row))
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(row, dtype=torch.float32))).item()
    return 1 if prob >= 0.5 else 0, prob


# ──────────────────────────────────────────────
# EXTRACT AUDIO FEATURES (REAL)
# ──────────────────────────────────────────────
def extract_audio_features(file_path):
    """Extract all speech features expected by the model."""
    y, sr = librosa.load(file_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    pitch = librosa.yin(y, fmin=50, fmax=300)

    duration = len(y) / sr
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    speech_rate = len(onsets) / (duration + 1e-5)

    intervals = librosa.effects.split(y, top_db=30)
    non_silent_samples = np.sum(intervals[:, 1] - intervals[:, 0])
    silence_ratio = 1 - (non_silent_samples / len(y))

    # Placeholders for features that require more advanced processing
    # (in a real system you would compute these from the audio signal)
    return {
        "mfcc_mean": np.mean(mfcc),
        "mfcc_std": np.std(mfcc),
        "pitch_mean": np.mean(pitch),
        "pitch_std": np.std(pitch),
        "speech_rate": speech_rate,
        "pause_rate": silence_ratio,
        "hnr_db": 15.0,        # placeholder – improve with real computation
        "jitter_pct": 1.0,     # placeholder
        "shimmer_pct": 4.0,    # placeholder
    }


# ──────────────────────────────────────────────
# SAMPLE PATIENTS (for demonstration)
# ──────────────────────────────────────────────
PATIENTS = [
    {
        "name": "Patient A — Likely Healthy",
        # Clinical
        "NACCAGE": 65,
        "SEX": 1,
        "EDUC": 16,
        "MMSELOC": 29,
        "CDRGLOB": 0,
        "MEMORY": 0,
        "JUDGMENT": 0,
        "COMMUN": 0,
        "HOMEHOBB": 0,
        "PERSCARE": 0,
        # Speech
        "mfcc_mean": 0.1,
        "mfcc_std": 1.0,
        "pitch_mean": 155.0,
        "pitch_std": 18.0,
        "speech_rate": 4.0,
        "pause_rate": 0.7,
        "hnr_db": 18.5,
        "jitter_pct": 0.45,
        "shimmer_pct": 2.8,
        # Handwriting
        "pressure_mean": 0.66,
        "pressure_std": 0.09,
        "velocity_mean": 4.30,
        "velocity_std": 1.10,
        "acceleration_mean": 2.20,
        "jerk_mean": 0.75,
        "writing_duration": 17,
        "num_pen_lifts": 11,
        "pen_up_time_pct": 0.16,
        "inter_stroke_pause": 0.18,
        "letter_size_mean": 0.52,
        "letter_size_std": 0.04,
        "word_spacing": 0.57,
        "stroke_length": 2.60,
        "slant_angle": 12,
        "tremor_index": 0.07,
        "direction_changes": 6,
        "pen_pressure_cv": 0.14,
        "clock_radius_std": 0.04,
        "clock_digit_err": 0.2,
    },
    {
        "name": "Patient B — Likely Alzheimer's",
        "NACCAGE": 80,
        "SEX": 2,
        "EDUC": 8,
        "MMSELOC": 18,
        "CDRGLOB": 2,
        "MEMORY": 3,
        "JUDGMENT": 3,
        "COMMUN": 2,
        "HOMEHOBB": 3,
        "PERSCARE": 2,
        "mfcc_mean": -1.5,
        "mfcc_std": 0.5,
        "pitch_mean": 120.0,
        "pitch_std": 30.0,
        "speech_rate": 2.0,
        "pause_rate": 3.8,
        "hnr_db": 10.5,
        "jitter_pct": 1.25,
        "shimmer_pct": 6.8,
        "pressure_mean": 0.45,
        "pressure_std": 0.22,
        "velocity_mean": 2.00,
        "velocity_std": 2.10,
        "acceleration_mean": 0.85,
        "jerk_mean": 2.10,
        "writing_duration": 46,
        "num_pen_lifts": 30,
        "pen_up_time_pct": 0.45,
        "inter_stroke_pause": 0.80,
        "letter_size_mean": 0.32,
        "letter_size_std": 0.14,
        "word_spacing": 0.28,
        "stroke_length": 1.30,
        "slant_angle": 3,
        "tremor_index": 0.42,
        "direction_changes": 14,
        "pen_pressure_cv": 0.40,
        "clock_radius_std": 0.22,
        "clock_digit_err": 3.1,
    },
    {
        "name": "Patient C — Borderline / MCI",
        "NACCAGE": 72,
        "SEX": 1,
        "EDUC": 12,
        "MMSELOC": 24,
        "CDRGLOB": 0.5,
        "MEMORY": 1,
        "JUDGMENT": 1,
        "COMMUN": 1,
        "HOMEHOBB": 1,
        "PERSCARE": 0,
        "mfcc_mean": 0.3,
        "mfcc_std": 0.9,
        "pitch_mean": 145.0,
        "pitch_std": 22.0,
        "speech_rate": 3.2,
        "pause_rate": 1.6,
        "hnr_db": 14.5,
        "jitter_pct": 0.80,
        "shimmer_pct": 4.5,
        "pressure_mean": 0.56,
        "pressure_std": 0.15,
        "velocity_mean": 3.10,
        "velocity_std": 1.60,
        "acceleration_mean": 1.50,
        "jerk_mean": 1.30,
        "writing_duration": 28,
        "num_pen_lifts": 19,
        "pen_up_time_pct": 0.30,
        "inter_stroke_pause": 0.45,
        "letter_size_mean": 0.43,
        "letter_size_std": 0.08,
        "word_spacing": 0.42,
        "stroke_length": 1.90,
        "slant_angle": 7,
        "tremor_index": 0.20,
        "direction_changes": 10,
        "pen_pressure_cv": 0.26,
        "clock_radius_std": 0.11,
        "clock_digit_err": 1.5,
    },
]


# ──────────────────────────────────────────────
# RUN PREDICTIONS ON SAMPLE PATIENTS
# ──────────────────────────────────────────────
def run_test():
    sp_model, sp_feats, sp_imp, sp_scl = load_model("speech_model_complete.pt")
    hw_model, hw_feats, hw_imp, hw_scl = load_model("hw_model_complete.pt")

    print("\n" + "=" * 65)
    print("         ALZHEIMER'S PREDICTION RESULTS — ALL MODALITIES")
    print("=" * 65)
    print(f"{'Patient':<30} {'Speech+Clinical':^20} {'Handwriting':^15}")
    print("-" * 65)

    results = []
    for p in PATIENTS:
        name = p["name"]
        sp_lbl, sp_prob = predict_speech(p, sp_model, sp_feats, sp_imp, sp_scl)
        hw_lbl, hw_prob = predict_hw(p, hw_model, hw_feats, hw_imp, hw_scl)

        sp_flag = "⚠️  AD" if sp_lbl == 1 else "✅ Hlth"
        hw_flag = "⚠️  AD" if hw_lbl == 1 else "✅ Hlth"

        print(f"\n  {name}")
        print(f"  {'':30} {'Prob':>6}  {'Label':<10}  {'Prob':>6}  {'Label'}")
        print(
            f"  {'':30} {sp_prob*100:5.1f}%  {sp_flag:<10}  {hw_prob*100:5.1f}%  {hw_flag}"
        )

        avg_prob = (sp_prob + hw_prob) / 2
        risk = (
            "🔴 HIGH"
            if avg_prob >= 0.75
            else ("🟡 MEDIUM" if avg_prob >= 0.5 else "🟢 LOW")
        )
        conf = "High" if abs(avg_prob - 0.5) > 0.3 else "Low — borderline"
        print(
            f"  Ensemble average: {avg_prob*100:.1f}%  |  Risk: {risk}  |  Confidence: {conf}"
        )

        results.append(
            {
                "patient": name,
                "sp_prob": sp_prob,
                "hw_prob": hw_prob,
                "avg_prob": avg_prob,
            }
        )

    print("\n" + "=" * 65)
    print("NOTE: Research tool only. Always consult a medical professional.")
    print("=" * 65)
    return results


# ──────────────────────────────────────────────
# REAL AUDIO PREDICTION (speech+clinical only)
# ──────────────────────────────────────────────
def predict_from_audio(audio_path):
    """Load speech model, extract features from audio, and predict."""
    sp_model, sp_feats, sp_imp, sp_scl = load_model("speech_model_complete.pt")

    # Extract audio features
    audio_feats = extract_audio_features(audio_path)

    # Build a complete patient dictionary.
    # The model expects many clinical features. We provide reasonable defaults
    # (taken from the healthy sample patient) for any missing ones.
    default_clinical = {
        "NACCAGE": 70,
        "SEX": 1,
        "EDUC": 12,
        "MMSELOC": 25,
        "CDRGLOB": 0.5,
        "MEMORY": 1,
        "JUDGMENT": 1,
        "COMMUN": 1,
        "HOMEHOBB": 1,
        "PERSCARE": 1,
    }

    # Merge: audio features override defaults, but defaults fill missing keys
    patient_data = {**default_clinical, **audio_feats}

    # Check that all features expected by the model are present.
    # If any are missing, add a default (0.0) and warn.
    missing = [f for f in sp_feats if f not in patient_data]
    if missing:
        print(f"Warning: Model expects {len(missing)} missing features. Using 0.0 as fallback.")
        for f in missing:
            patient_data[f] = 0.0

    label, prob = predict_speech(patient_data, sp_model, sp_feats, sp_imp, sp_scl)

    print("\n" + "=" * 50)
    print("        REAL AUDIO PREDICTION")
    print("=" * 50)
    print(f"Audio file: {audio_path}")
    if label == 1:
        print(f"⚠️  Alzheimer’s Likely   (probability = {prob:.3f})")
    else:
        print(f"✅ Healthy               (probability = {1-prob:.3f})")
    print("=" * 50)

    return {"label": label, "probability": prob}


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alzheimer's prediction from multimodal data")
    parser.add_argument("audio_file", nargs="?", help="Path to a .wav audio file for prediction")
    args = parser.parse_args()

    if args.audio_file:
        # Real audio prediction mode
        predict_from_audio(args.audio_file)
    else:
        # Default: run on the three sample patients
        run_test()