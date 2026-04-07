"""
speech_audio.py
================
Loads trained speech model and predicts Alzheimer's from real audio.

Usage:
    python speech_audio.py path/to/audio.wav
"""

import sys
import numpy as np
import torch
import librosa

# Import your model class
from speech import MultimodalNet


# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
def load_model():
    checkpoint = torch.load("result/speech_model_complete.pt")

    model = MultimodalNet(len(checkpoint["feature_names"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


# ──────────────────────────────────────────────
# AUDIO FEATURE EXTRACTION
# ──────────────────────────────────────────────
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc)
    mfcc_std = np.std(mfcc)

    # Pitch
    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)

    # Speech rate (approx)
    duration = len(y) / sr
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    speech_rate = len(onsets) / (duration + 1e-5)

    # Pause rate (simple)
    intervals = librosa.effects.split(y, top_db=30)
    silence_ratio = 1 - (np.sum(intervals[:,1] - intervals[:,0]) / len(y))
    pause_rate = silence_ratio

    return {
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "speech_rate": speech_rate,
        "pause_rate": pause_rate,
        "hnr_db": 15.0,
        "jitter_pct": 1.0,
        "shimmer_pct": 4.0
    }


# ──────────────────────────────────────────────
# DUMMY CLINICAL DATA
# ──────────────────────────────────────────────
def get_dummy_clinical():
    return {
        "NACCAGE": 70,
        "SEX": 1,
        "EDUC": 12,
        "MMSELOC": 25,
        "CDRGLOB": 0.5,
        "MEMORY": 1,
        "JUDGMENT": 1,
        "COMMUN": 1,
        "HOMEHOBB": 1,
        "PERSCARE": 1
    }


# ──────────────────────────────────────────────
# PREPARE INPUT
# ──────────────────────────────────────────────
def prepare_input(audio_path, feature_names):
    speech = extract_audio_features(audio_path)
    clinical = get_dummy_clinical()

    data = {**clinical, **speech}

    # engineered features (same as training)
    data["age_education"] = data["NACCAGE"] * data["EDUC"]
    data["mmse_cdr"] = data["MMSELOC"] * data["CDRGLOB"]
    data["pitch_variance"] = data["pitch_std"] / (data["pitch_mean"] + 1e-5)

    return np.array([[data[f] for f in feature_names]])


# ──────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────
def predict(audio_path):
    model, checkpoint = load_model()

    X = prepare_input(audio_path, checkpoint["feature_names"])

    # preprocessing
    X = checkpoint["imputer"].transform(X)
    X = checkpoint["scaler"].transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        prob = torch.sigmoid(model(X_tensor)).item()

    print("\n========== RESULT ==========")
    if prob > 0.5:
        print(f"⚠️ Alzheimer’s Likely")
        print(f"Confidence: {prob:.2f}")
    else:
        print(f"✅ Healthy")
        print(f"Confidence: {1 - prob:.2f}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python speech_audio.py audio.wav")
        sys.exit()

    audio_file = sys.argv[1]
    predict(audio_file)