"""
Multimodal Alzheimer's Detection — Improved Speech Model
=========================================================
Improvements over baseline:
  1. Richer feature set  (prosodic, articulatory, pause, spectral)
  2. Audio preprocessing (silence trimming, peak normalisation)
  3. Data augmentation  (noise, time-stretch, pitch-shift)
  4. sklearn Pipeline   (scaler + classifier, no data-leakage)
  5. Voting Ensemble    (RF + SVM + GradientBoosting)
  6. Stratified K-Fold cross-validation
  7. Full evaluation    (ROC-AUC, confusion matrix, feature importance)
  8. Artefact saving    (model, scaler, feature names)
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
DATA_PATH = "train"
MODEL_DIR = "model"
SAMPLE_RATE = 16_000  # resample everything to 16 kHz
N_MFCC = 40
AUGMENT = True  # set False to skip augmentation
RANDOM_STATE = 42

LABELS = {"healthy": 0, "Alzheimer": 1}


# ─────────────────────────────────────────────
# 1. AUDIO PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(y: np.ndarray, sr: int) -> np.ndarray:
    """Trim silence, peak-normalise."""
    y, _ = librosa.effects.trim(y, top_db=25)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


# ─────────────────────────────────────────────
# 2. FEATURE EXTRACTION  (60+ features)
# ─────────────────────────────────────────────
def extract_features(file_path: str) -> np.ndarray:
    y_raw, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    y = preprocess(y_raw, sr)

    features = {}

    # — MFCCs (mean + std → 80 values) —————————————————————————
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    features["mfcc_mean"] = np.mean(mfcc, axis=1)  # 40
    features["mfcc_std"] = np.std(mfcc, axis=1)  # 40

    # — Delta MFCCs (capture temporal change) ——————————————————
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features["delta_mean"] = np.mean(delta, axis=1)  # 40
    features["delta2_mean"] = np.mean(delta2, axis=1)  # 40

    # — Chroma ——————————————————————————————————————————————————
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_mean"] = np.mean(chroma, axis=1)  # 12
    features["chroma_std"] = np.std(chroma, axis=1)  # 12

    # — Spectral features ———————————————————————————————————————
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)

    features["contrast_mean"] = np.mean(contrast, axis=1)  # 7
    features["rolloff_mean"] = [np.mean(rolloff)]  # 1
    features["bandwidth_mean"] = [np.mean(bandwidth)]  # 1
    features["centroid_mean"] = [np.mean(centroid)]  # 1
    features["flatness_mean"] = [np.mean(flatness)]  # 1

    # — Zero-crossing rate ——————————————————————————————————————
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = [np.mean(zcr)]
    features["zcr_std"] = [np.std(zcr)]

    # — RMS energy ——————————————————————————————————————————————
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = [np.mean(rms)]
    features["rms_std"] = [np.std(rms)]

    # — Fundamental frequency / Pitch (F0) ————————————————————
    # Alzheimer patients often show reduced pitch variability
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        fill_na=0.0,
    )
    voiced_f0 = f0[voiced_flag == 1] if np.any(voiced_flag) else np.array([0.0])
    features["f0_mean"] = [np.mean(voiced_f0)]
    features["f0_std"] = [np.std(voiced_f0)]  # pitch variability
    features["f0_range"] = [np.ptp(voiced_f0)]  # pitch range
    features["voiced_frac"] = [np.mean(voiced_flag)]  # voicing fraction

    # — Pause / silence analysis ————————————————————————————————
    # Alzheimer speech has more/longer pauses
    frame_length, hop_length = 512, 256
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[
        0
    ]
    silence_threshold = 0.01 * np.max(energy)
    silent_frames = energy < silence_threshold
    pause_count, in_pause = 0, False
    pause_lengths = []
    current_pause = 0
    for sf in silent_frames:
        if sf:
            in_pause = True
            current_pause += 1
        else:
            if in_pause:
                pause_count += 1
                pause_lengths.append(current_pause)
                current_pause = 0
                in_pause = False
    if in_pause:
        pause_count += 1
        pause_lengths.append(current_pause)

    features["pause_count"] = [pause_count]
    features["pause_mean_dur"] = [np.mean(pause_lengths) if pause_lengths else 0]
    features["pause_max_dur"] = [np.max(pause_lengths) if pause_lengths else 0]
    features["silence_ratio"] = [np.mean(silent_frames)]

    # — Jitter approximation (cycle-to-cycle F0 variation) ———
    if len(voiced_f0) > 1:
        jitter = np.mean(np.abs(np.diff(voiced_f0))) / (np.mean(voiced_f0) + 1e-9)
    else:
        jitter = 0.0
    features["jitter"] = [jitter]

    # — Shimmer approximation (amplitude variation) ————————————
    if len(voiced_f0) > 1:
        amp = np.abs(y)
        shimmer = np.mean(np.abs(np.diff(amp))) / (np.mean(amp) + 1e-9)
    else:
        shimmer = 0.0
    features["shimmer"] = [shimmer]

    # — Speaking rate (voiced frames per second) ———————————————
    speaking_rate = np.sum(voiced_flag) / (len(y) / sr + 1e-9)
    features["speaking_rate"] = [speaking_rate]

    # — Concatenate all features ————————————————————————————————
    return np.hstack([v for v in features.values()])


def get_feature_names() -> list[str]:
    """Return feature names matching extract_features() output."""
    names = []
    names += [f"mfcc_mean_{i}" for i in range(N_MFCC)]
    names += [f"mfcc_std_{i}" for i in range(N_MFCC)]
    names += [f"delta_mean_{i}" for i in range(N_MFCC)]
    names += [f"delta2_mean_{i}" for i in range(N_MFCC)]
    names += [f"chroma_mean_{i}" for i in range(12)]
    names += [f"chroma_std_{i}" for i in range(12)]
    names += [f"contrast_{i}" for i in range(7)]
    names += [
        "rolloff",
        "bandwidth",
        "centroid",
        "flatness",
        "zcr_mean",
        "zcr_std",
        "rms_mean",
        "rms_std",
        "f0_mean",
        "f0_std",
        "f0_range",
        "voiced_frac",
        "pause_count",
        "pause_mean_dur",
        "pause_max_dur",
        "silence_ratio",
        "jitter",
        "shimmer",
        "speaking_rate",
    ]
    return names


# ─────────────────────────────────────────────
# 3. DATA AUGMENTATION
# ─────────────────────────────────────────────
def augment_audio(y: np.ndarray, sr: int) -> list[np.ndarray]:
    """Return list of augmented copies (adds noise, stretch, pitch-shift)."""
    augmented = []

    # Gaussian noise
    noise = y + 0.005 * np.random.randn(len(y))
    augmented.append(noise)

    # Time stretch (slow down — mirrors Alzheimer speech)
    stretched = librosa.effects.time_stretch(y, rate=0.9)
    augmented.append(stretched)

    # Pitch shift ±1 semitone
    up = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
    down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1)
    augmented.append(up)
    augmented.append(down)

    return augmented


# ─────────────────────────────────────────────
# 4. LOAD DATASET  (with optional augmentation)
# ─────────────────────────────────────────────
print("=" * 50)
print("Loading dataset …")
print("=" * 50)

X_raw, y_raw, file_list = [], [], []

for label, idx in LABELS.items():
    folder = os.path.join(DATA_PATH, label)
    files = [
        f
        for f in os.listdir(folder)
        if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
    ]
    print(f"  {label}: {len(files)} files")

    for fname in files:
        fpath = os.path.join(folder, fname)
        try:
            feats = extract_features(fpath)
            X_raw.append(feats)
            y_raw.append(idx)
            file_list.append(fpath)

            # Augment Alzheimer class more aggressively (class imbalance helper)
            if AUGMENT and idx == 1:
                audio, sr = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)
                audio = preprocess(audio, sr)
                for aug_audio in augment_audio(audio, sr):
                    try:
                        # Extract from augmented waveform via a temp approach
                        aug_feats = _extract_from_array(aug_audio, sr)
                        X_raw.append(aug_feats)
                        y_raw.append(idx)
                    except Exception:
                        pass

        except Exception as e:
            print(f"    ⚠ Skipped {fpath}: {e}")


def _extract_from_array(y: np.ndarray, sr: int) -> np.ndarray:
    """Same as extract_features() but operates on a numpy array directly."""
    import tempfile, soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y, sr)
        feats = extract_features(tmp.name)
    os.unlink(tmp.name)
    return feats


X = np.array(X_raw, dtype=np.float32)
y = np.array(y_raw, dtype=np.int32)
feat_names = get_feature_names()

print(f"\nFinal dataset — X: {X.shape}, y: {y.shape}")
print(f"Class balance — Healthy: {np.sum(y==0)}, Alzheimer: {np.sum(y==1)}")


# ─────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

# — PCA scatter —
pca = PCA(n_components=2)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

plt.figure(figsize=(7, 5))
colors = ["steelblue", "tomato"]
class_names = ["Healthy", "Alzheimer"]
for i, (name, c) in enumerate(zip(class_names, colors)):
    mask = y == i
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=name,
        alpha=0.6,
        c=c,
        edgecolors="none",
        s=30,
    )
plt.title("PCA — Speech Features (Healthy vs Alzheimer)", fontsize=12)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/pca_scatter.png", dpi=150)
plt.show()

# — MFCC example —
example_file = file_list[0]
y_audio, sr_ex = librosa.load(example_file, sr=SAMPLE_RATE, mono=True)
mfcc_ex = librosa.feature.mfcc(y=y_audio, sr=sr_ex, n_mfcc=N_MFCC)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc_ex, sr=sr_ex, x_axis="time")
plt.colorbar(format="%+2.0f dB")
plt.title("MFCC — Sample Audio", fontsize=12)
plt.tight_layout()
plt.savefig("plots/mfcc_example.png", dpi=150)
plt.show()


# ─────────────────────────────────────────────
# 6. MODEL — VOTING ENSEMBLE INSIDE A PIPELINE
# ─────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=RANDOM_STATE,
)
svm = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=RANDOM_STATE,
)

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb), ("svm", svm)],
    voting="soft",  # soft voting uses predicted probabilities
    weights=[1, 1, 1],
)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("ensemble", ensemble),
    ]
)


# ─────────────────────────────────────────────
# 7. STRATIFIED K-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("Cross-validation (5-fold stratified) …")
print("=" * 50)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring=["accuracy", "f1", "roc_auc"],
    return_train_score=True,
    n_jobs=-1,
)

print(
    f"  Accuracy  : {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}"
)
print(
    f"  F1        : {cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}"
)
print(
    f"  ROC-AUC   : {cv_results['test_roc_auc'].mean():.3f} ± {cv_results['test_roc_auc'].std():.3f}"
)


# ─────────────────────────────────────────────
# 8. FINAL TRAINING ON FULL DATASET
# ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

pipeline.fit(X_train, y_train)

print("\n" + "=" * 50)
print("Test-set Evaluation")
print("=" * 50)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=class_names))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# — Confusion matrix —
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=150)
plt.show()

# — ROC Curve —
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name="Ensemble")
ax.set_title("ROC Curve")
plt.tight_layout()
plt.savefig("plots/roc_curve.png", dpi=150)
plt.show()

# — Feature importance from the RF member —
scaler_ = pipeline.named_steps["scaler"]
ensemble_ = pipeline.named_steps["ensemble"]
rf_ = ensemble_.estimators_[0]  # trained RF

importances = rf_.feature_importances_
top_k = 20
top_idx = np.argsort(importances)[-top_k:][::-1]
top_names = [feat_names[i] if i < len(feat_names) else f"feat_{i}" for i in top_idx]

plt.figure(figsize=(10, 5))
plt.bar(range(top_k), importances[top_idx])
plt.xticks(range(top_k), top_names, rotation=45, ha="right", fontsize=8)
plt.title(f"Top {top_k} Feature Importances (Random Forest member)")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150)
plt.show()


# ─────────────────────────────────────────────
# 9. SAVE ARTEFACTS
# ─────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(pipeline, f"{MODEL_DIR}/alzheimer_pipeline.pkl")
joblib.dump(LABELS, f"{MODEL_DIR}/labels.pkl")
joblib.dump(feat_names, f"{MODEL_DIR}/feature_names.pkl")

print(f"\n✅  Saved pipeline → {MODEL_DIR}/alzheimer_pipeline.pkl")
print(f"✅  Saved labels   → {MODEL_DIR}/labels.pkl")
print(f"✅  Saved features → {MODEL_DIR}/feature_names.pkl")


# ─────────────────────────────────────────────
# 10. INFERENCE HELPER
# ─────────────────────────────────────────────
def predict(file_path: str, threshold: float = 0.5) -> dict:
    """
    Returns:
        {
          "label":       "Healthy" | "Alzheimer",
          "probability": float,          # P(Alzheimer)
          "confidence":  "Low|Med|High"
        }
    """
    feats = extract_features(file_path).reshape(1, -1)
    prob = pipeline.predict_proba(feats)[0, 1]
    label = "Alzheimer" if prob >= threshold else "Healthy"
    if prob < 0.35 or prob > 0.65:
        confidence = "High"
    elif prob < 0.40 or prob > 0.60:
        confidence = "Medium"
    else:
        confidence = "Low"
    return {
        "label": label,
        "probability": round(float(prob), 4),
        "confidence": confidence,
    }

# ─────────────────────────────────────────────
# 11. SAVE PYTORCH .pt MODEL (compatible with test.py)
# ─────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import roc_auc_score as ras

class MultimodalNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,  32),        nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32,  1)
        )
    def forward(self, x):
        return self.net(x).squeeze()

print("\n" + "=" * 50)
print("Training PyTorch MultimodalNet for .pt export ...")
print("=" * 50)

SEED       = 42
BATCH_SIZE = 64
NUM_EPOCHS = 100
PATIENCE   = 15
torch.manual_seed(SEED)

# ── Impute + scale (kept separate so test.py can call imputer/scaler directly) ──
pt_imputer = SimpleImputer(strategy="median")
pt_scaler  = StandardScaler()

X_tr, X_te, y_tr, y_te = tts(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_tr, X_va, y_tr, y_va = tts(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=SEED)

X_tr = pt_scaler.fit_transform(pt_imputer.fit_transform(X_tr))
X_va = pt_scaler.transform(pt_imputer.transform(X_va))
X_te = pt_scaler.transform(pt_imputer.transform(X_te))

to_t = lambda a, dt: torch.tensor(a, dtype=dt)
X_tr_t = to_t(X_tr, torch.float32);  y_tr_t = to_t(y_tr.astype(float), torch.float32)
X_va_t = to_t(X_va, torch.float32);  y_va_t = to_t(y_va.astype(float), torch.float32)
X_te_t = to_t(X_te, torch.float32);  y_te_t = to_t(y_te.astype(float), torch.float32)

loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)

model     = MultimodalNet(X_tr_t.shape[1])
pos_w     = float((y_tr_t == 0).sum() / (y_tr_t == 1).sum())
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

best_auc, no_improve, best_state = 0.0, 0, None

for epoch in range(NUM_EPOCHS):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_va_t)).numpy()
        val_auc   = ras(y_va_t.numpy(), val_probs)
    scheduler.step(val_auc)

    if val_auc > best_auc:
        best_auc   = val_auc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d} | Val AUC: {val_auc:.4f}")

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    test_probs = torch.sigmoid(model(X_te_t)).numpy()
    test_preds = (test_probs >= 0.5).astype(int)

print(f"  Test AUC : {ras(y_te_t.numpy(), test_probs):.4f}")

# ── Save in the exact format test.py expects ──
os.makedirs("result", exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "feature_names":    feat_names,   # list from get_feature_names()
    "imputer":          pt_imputer,
    "scaler":           pt_scaler,
}, "result/speech_audio_model_complete.pt")

print("✅  Saved → result/speech_audio_model_complete.pt")


# Quick smoke-test on the first training file
result = predict(example_file)
print(f"\nSample prediction → {result}")
    