import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# =========================
# MODEL DEFINITION (copied so no import needed)
# =========================
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

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()


# =========================
# LOAD SAVED MODEL
# =========================
BASE_DIR = Path(__file__).resolve().parent

checkpoint = torch.load(
    BASE_DIR / "result/alzheimer_model_complete.pt",
    weights_only=False
)

feature_names = checkpoint["feature_names"]
imputer       = checkpoint["imputer"]
scaler        = checkpoint["scaler"]

model = MultimodalNet(input_dim=len(feature_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"✓ Model loaded successfully")
print(f"✓ Number of features: {len(feature_names)}")
print(f"✓ Features: {feature_names}")


# =========================
# PREDICTION FUNCTION
# =========================
def predict_patient(patient_data):
    row = np.array([[patient_data.get(f, np.nan) for f in feature_names]], dtype=float)

    # Compute engineered features
    age   = patient_data.get("NACCAGE",     np.nan)
    educ  = patient_data.get("EDUC",        np.nan)
    mmse  = patient_data.get("MMSELOC",     np.nan)
    cdr   = patient_data.get("CDRGLOB",     np.nan)
    pmean = patient_data.get("pitch_mean",  np.nan)
    pstd  = patient_data.get("pitch_std",   np.nan)

    row[0][feature_names.index("age_education")]  = age * educ
    row[0][feature_names.index("mmse_cdr")]       = mmse * cdr
    row[0][feature_names.index("pitch_variance")] = pstd / (pmean + 1e-5)

    # Impute missing values then scale
    row = imputer.transform(row)
    row = scaler.transform(row)

    # Run through model
    tensor = torch.tensor(row, dtype=torch.float32)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()

    label = 1 if prob >= 0.5 else 0
    return label, prob


# =========================
# TEST PATIENTS
# =========================
patients = [
    {
        "name":        "Patient A — Likely Healthy",
        "NACCAGE":     65,
        "SEX":          1,      # 1=Male, 2=Female
        "EDUC":        16,      # Years of education
        "MMSELOC":     29,      # MMSE score (max=30, lower = more impaired)
        "CDRGLOB":      0,      # CDR (0=normal, 0.5=MCI, 1=mild, 2=moderate, 3=severe)
        "MEMORY":       0,      # 0=normal, 1=slight, 2=mild, 3=moderate, 4=severe
        "JUDGMENT":     0,
        "COMMUN":       0,
        "HOMEHOBB":     0,
        "PERSCARE":     0,
        "mfcc_mean":    0.1,
        "mfcc_std":     1.0,
        "pitch_mean": 155.0,
        "pitch_std":   18.0,
        "speech_rate":  4.0,
    },
    {
        "name":        "Patient B — Likely Alzheimer's",
        "NACCAGE":     80,
        "SEX":          2,
        "EDUC":         8,
        "MMSELOC":     18,
        "CDRGLOB":      2,
        "MEMORY":       3,
        "JUDGMENT":     3,
        "COMMUN":       2,
        "HOMEHOBB":     3,
        "PERSCARE":     2,
        "mfcc_mean":   -1.5,
        "mfcc_std":     0.5,
        "pitch_mean": 120.0,
        "pitch_std":   30.0,
        "speech_rate":  2.0,
    },
    {
        "name":        "Patient C — Borderline / MCI",
        "NACCAGE":     72,
        "SEX":          1,
        "EDUC":        12,
        "MMSELOC":     24,
        "CDRGLOB":      0.5,
        "MEMORY":       1,
        "JUDGMENT":     1,
        "COMMUN":       1,
        "HOMEHOBB":     1,
        "PERSCARE":     0,
        "mfcc_mean":    0.3,
        "mfcc_std":     0.9,
        "pitch_mean": 145.0,
        "pitch_std":   22.0,
        "speech_rate":  3.2,
    },
]

# =========================
# RUN PREDICTIONS
# =========================
print("\n" + "="*55)
print("       ALZHEIMER'S PREDICTION RESULTS")
print("="*55)

for p in patients:
    name = p.pop("name")
    label, prob = predict_patient(p)

    if prob >= 0.75:
        risk = "🔴 HIGH"
    elif prob >= 0.5:
        risk = "🟡 MEDIUM"
    else:
        risk = "🟢 LOW"

    status = "⚠️  Alzheimer's Detected" if label == 1 else "✅ Healthy"

    print(f"\n{name}")
    print(f"  Probability  : {prob*100:.1f}%")
    print(f"  Prediction   : {status}")
    print(f"  Risk Level   : {risk}")
    print(f"  Confidence   : {'High' if abs(prob - 0.5) > 0.3 else 'Low — borderline case'}")

#warning: This is a simplified demonstration. Real clinical tools require rigorous validation and should not be used for self-diagnosis.
print("\n" + "="*55)
print("NOTE: This tool is for research only.")
print("Always consult a medical professional. and never rely solely on AI for health decisions.")
print("="*55)