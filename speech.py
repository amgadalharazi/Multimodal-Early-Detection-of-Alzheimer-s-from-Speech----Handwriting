"""
speech.py
=========
Loads the NACC clinical dataset, creates synthetic speech features,
merges them, trains a MultimodalNet, and saves:
    result/speech_model_complete.pt

Run standalone:
    python speech.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)

BASE_DIR   = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
DATA_DIR   = BASE_DIR / "Datasets"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE  = 64
NUM_EPOCHS  = 150
PATIENCE    = 15
SEED        = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# HELPER — per-split normalised distributions
# ──────────────────────────────────────────────
def trnorm(mean, std, low, high, size):
    return np.clip(np.random.normal(mean, std, size), low, high)


# ──────────────────────────────────────────────
# LOAD & PREPARE NACC DATA
# ──────────────────────────────────────────────
def load_nacc(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[df["PACKET"] == "I"]
    df = df[df["NACCUDSD"].isin([1, 3])]
    df["label"] = df["NACCUDSD"].map({1: 0, 3: 1})
    return df


# ──────────────────────────────────────────────
# CREATE / LOAD SPEECH FEATURES
# ──────────────────────────────────────────────
def get_speech_features(df, speech_path):
    """
    Load or generate speech features.
    If the CSV exists but is missing columns (e.g. created by an older version),
    the missing columns are synthesised and the CSV is updated in place.
    """
    REQUIRED_COLS = [
        "NACCID",
        "mfcc_mean", "mfcc_std",
        "pitch_mean", "pitch_std", "speech_rate",
        "pause_rate", "hnr_db", "jitter_pct", "shimmer_pct",
    ]

    # ── Per-label generation helpers ────────────────────────────────
    def _row_healthy():
        return dict(
            mfcc_mean   = trnorm( 0.2,  0.6,  -2,    2,   1)[0],
            mfcc_std    = trnorm( 1.1,  0.2,   0.5,  2,   1)[0],
            pitch_mean  = trnorm(155,   20,  100,  230,   1)[0],
            pitch_std   = trnorm( 18,    5,    5,   40,   1)[0],
            speech_rate = trnorm(  4.0,  0.5,  2,    6,   1)[0],
            pause_rate  = trnorm(  0.8,  0.3,  0,    3,   1)[0],
            hnr_db      = trnorm( 18,    3,   10,   28,   1)[0],
            jitter_pct  = trnorm(  0.5,  0.15, 0.1,  1.2, 1)[0],
            shimmer_pct = trnorm(  3.0,  0.8,  1,    7,   1)[0],
        )

    def _row_ad():
        return dict(
            mfcc_mean   = trnorm(-1.5,  0.9,  -4,    1,   1)[0],
            mfcc_std    = trnorm( 0.55, 0.2,   0.1,  1.2, 1)[0],
            pitch_mean  = trnorm(120,   25,   70,  200,   1)[0],
            pitch_std   = trnorm( 30,    8,   10,   60,   1)[0],
            speech_rate = trnorm(  2.2,  0.7,  0.5,  4,   1)[0],
            pause_rate  = trnorm(  3.5,  1.0,  1,    8,   1)[0],
            hnr_db      = trnorm( 11,    4,    4,   21,   1)[0],
            jitter_pct  = trnorm(  1.2,  0.35, 0.3,  3,   1)[0],
            shimmer_pct = trnorm(  6.5,  1.5,  3,   12,   1)[0],
        )

    # ── Case 1: CSV doesn't exist → generate from scratch ───────────
    if not speech_path.exists():
        print("  Speech features not found → generating synthetic features ...")
        unique_ids = df["NACCID"].unique()
        labels     = df.set_index("NACCID")["label"].to_dict()

        rows = []
        for nid in unique_ids:
            lbl = labels.get(nid, 0)
            row = _row_healthy() if lbl == 0 else _row_ad()
            row["NACCID"] = nid
            rows.append(row)

        speech_df = pd.DataFrame(rows)
        speech_df.to_csv(speech_path, index=False)
        print(f"  ✓ Saved {len(speech_df)} synthetic speech records → {speech_path}")
        return speech_df

    # ── Case 2: CSV exists → load and patch missing columns ─────────
    speech_df = pd.read_csv(speech_path)
    print(f"  ✓ Loaded {len(speech_df)} speech records from {speech_path}")

    missing_cols = [c for c in REQUIRED_COLS if c != "NACCID" and c not in speech_df.columns]
    if missing_cols:
        print(f"  ⚠  Old speech CSV is missing columns: {missing_cols}")
        print(f"  → Adding them synthetically using label information ...")

        # Build label lookup from the NACC dataframe
        labels = df.set_index("NACCID")["label"].to_dict()

        for col in missing_cols:
            new_vals = []
            for nid in speech_df["NACCID"]:
                lbl = labels.get(nid, 0)
                row = _row_healthy() if lbl == 0 else _row_ad()
                new_vals.append(row[col])
            speech_df[col] = new_vals

        speech_df.to_csv(speech_path, index=False)
        print(f"  ✓ Updated speech_features.csv with {len(missing_cols)} new column(s)")

    return speech_df


# ──────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────
def train_speech_model():
    print("\n" + "="*55)
    print("  SPEECH MODEL — TRAINING")
    print("="*55)

    # ── Load NACC ──
    nacc_path = DATA_DIR / "nacc.csv"
    df = load_nacc(nacc_path)
    print(f"  ✓ NACC records (Healthy + AD): {len(df)}")

    # ── Speech features ──
    speech_df = get_speech_features(df, DATA_DIR / "speech_features.csv")
    df = df.merge(speech_df, on="NACCID", how="inner")
    print(f"  ✓ Merged dataset: {len(df)} records")

    # ── Feature lists ──
    clinical_features = [
        "NACCAGE", "SEX", "EDUC",
        "MMSELOC", "CDRGLOB",
        "MEMORY", "JUDGMENT", "COMMUN", "HOMEHOBB", "PERSCARE"
    ]
    speech_features = [
        "mfcc_mean", "mfcc_std",
        "pitch_mean", "pitch_std",
        "speech_rate", "pause_rate",
        "hnr_db", "jitter_pct", "shimmer_pct"
    ]
    raw_cols = clinical_features + speech_features
    df[raw_cols] = df[raw_cols].replace([-4, -9, 88, 99], pd.NA)
    df[raw_cols] = df[raw_cols].apply(pd.to_numeric, errors="coerce")

    # ── Engineered features ──
    df["age_education"]  = df["NACCAGE"]   * df["EDUC"]
    df["mmse_cdr"]       = df["MMSELOC"]   * df["CDRGLOB"]
    df["pitch_variance"] = df["pitch_std"] / (df["pitch_mean"] + 1e-5)

    all_features = clinical_features + speech_features + \
                   ["age_education", "mmse_cdr", "pitch_variance"]

    X = df[all_features].values.astype(float)
    y = df["label"].values.astype(float)

    # ── Split ──
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.2, stratify=y_tmp, random_state=SEED)

    # ── Impute & scale (fit on train only) ──
    imputer = KNNImputer(n_neighbors=5)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_val   = scaler.transform(imputer.transform(X_val))
    X_test  = scaler.transform(imputer.transform(X_test))

    print(f"  Train: {len(y_train)}  |  Val: {len(y_val)}  |  Test: {len(y_test)}")

    # ── Tensors / loaders ──
    to_t  = lambda a: torch.tensor(a, dtype=torch.float32)
    X_tr_t, y_tr_t = to_t(X_train), to_t(y_train)
    X_vl_t, y_vl_t = to_t(X_val),   to_t(y_val)
    X_ts_t, y_ts_t = to_t(X_test),  to_t(y_test)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=BATCH_SIZE, shuffle=True)

    # ── Model ──
    model     = MultimodalNet(X_tr_t.shape[1])
    pos_w     = float((y_tr_t == 0).sum() / (y_tr_t == 1).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=5)

    # ── Training loop ──
    best_auc, no_improve = 0.0, 0
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
            val_probs = torch.sigmoid(model(X_vl_t)).numpy()
            val_auc   = roc_auc_score(y_vl_t.numpy(), val_probs)
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

    # ── Test evaluation ──
    model.eval()
    with torch.no_grad():
        test_probs = torch.sigmoid(model(X_ts_t)).numpy()
        test_preds = (test_probs >= 0.5).astype(int)
    y_np = y_ts_t.numpy()

    metrics = {
        "modality":  "Speech + Clinical",
        "auc":       roc_auc_score(y_np, test_probs),
        "accuracy":  accuracy_score(y_np, test_preds),
        "precision": precision_score(y_np, test_preds, zero_division=0),
        "recall":    recall_score(y_np, test_preds, zero_division=0),
        "f1":        f1_score(y_np, test_preds, zero_division=0),
    }

    print(f"\n  ── Speech Model Test Results ──────────────────")
    for k, v in metrics.items():
        if k != "modality":
            print(f"  {k.capitalize():12s}: {v:.4f}")
    print(classification_report(y_np, test_preds,
                                   target_names=["Healthy", "AD"]))

    # ── Save ──
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_names":    all_features,
        "imputer":          imputer,
        "scaler":           scaler,
    }, RESULT_DIR / "speech_model_complete.pt")
    print(f"  ✓ Saved → result/speech_model_complete.pt")

    return metrics


if __name__ == "__main__":
    train_speech_model()