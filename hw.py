"""
hw.py
=====
Generates a synthetic handwriting dataset and trains a MultimodalNet on it.
Saves:
    Datasets/handwriting_dataset.csv
    result/hw_model_complete.pt

Run standalone:
    python hw.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

BASE_DIR   = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
DATA_DIR   = BASE_DIR / "Datasets"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
NUM_EPOCHS = 100
PATIENCE   = 12
SEED       = 42
N_HEALTHY  = 500
N_AD       = 500
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
# HELPER
# ──────────────────────────────────────────────
def trnorm(mean, std, low, high, size):
    return np.clip(np.random.normal(mean, std, size), low, high)


# ──────────────────────────────────────────────
# SYNTHETIC HANDWRITING DATA
# ──────────────────────────────────────────────
def gen_healthy(n):
    return {
        # Pen kinematics
        "pressure_mean":     trnorm(0.65, 0.10, 0.30, 1.00, n),
        "pressure_std":      trnorm(0.10, 0.03, 0.02, 0.25, n),
        "velocity_mean":     trnorm(4.20, 0.60, 2.00, 7.00, n),
        "velocity_std":      trnorm(1.20, 0.25, 0.30, 2.50, n),
        "acceleration_mean": trnorm(2.10, 0.40, 0.80, 4.00, n),
        "jerk_mean":         trnorm(0.80, 0.15, 0.30, 1.50, n),
        # Temporal
        "writing_duration":  trnorm(18,   3.0,  10,   35,   n),
        "num_pen_lifts":     trnorm(12,   3.0,   5,   25,   n).astype(int),
        "pen_up_time_pct":   trnorm(0.18, 0.05, 0.05, 0.40, n),
        "inter_stroke_pause":trnorm(0.20, 0.05, 0.05, 0.60, n),
        # Spatial
        "letter_size_mean":  trnorm(0.50, 0.07, 0.30, 0.80, n),
        "letter_size_std":   trnorm(0.04, 0.01, 0.01, 0.10, n),
        "word_spacing":      trnorm(0.55, 0.08, 0.30, 0.90, n),
        "stroke_length":     trnorm(2.50, 0.40, 1.00, 4.50, n),
        "slant_angle":       trnorm(10,   5.0, -15,   35,   n),
        # Tremor / noise
        "tremor_index":      trnorm(0.08, 0.03, 0.01, 0.20, n),
        "direction_changes": trnorm(6.5,  1.0,  3,    12,   n),
        "pen_pressure_cv":   trnorm(0.15, 0.04, 0.04, 0.35, n),
        # Clock-drawing
        "clock_radius_std":  trnorm(0.05, 0.02, 0.01, 0.15, n),
        "clock_digit_err":   trnorm(0.3,  0.3,  0,    2,    n),
    }


def gen_ad(n):
    return {
        "pressure_mean":     trnorm(0.48, 0.14, 0.15, 0.85, n),
        "pressure_std":      trnorm(0.20, 0.06, 0.05, 0.40, n),
        "velocity_mean":     trnorm(2.20, 0.80, 0.50, 4.50, n),
        "velocity_std":      trnorm(2.00, 0.50, 0.50, 4.00, n),
        "acceleration_mean": trnorm(0.90, 0.40, 0.20, 2.50, n),
        "jerk_mean":         trnorm(1.90, 0.35, 0.80, 3.50, n),
        "writing_duration":  trnorm(42,  10.0,  20,   80,   n),
        "num_pen_lifts":     trnorm(28,   6.0,  12,   55,   n).astype(int),
        "pen_up_time_pct":   trnorm(0.42, 0.10, 0.15, 0.72, n),
        "inter_stroke_pause":trnorm(0.70, 0.20, 0.15, 1.50, n),
        "letter_size_mean":  trnorm(0.35, 0.12, 0.10, 0.65, n),
        "letter_size_std":   trnorm(0.12, 0.04, 0.03, 0.30, n),
        "word_spacing":      trnorm(0.32, 0.12, 0.05, 0.65, n),
        "stroke_length":     trnorm(1.40, 0.50, 0.40, 3.00, n),
        "slant_angle":       trnorm(5,   10.0, -30,   40,   n),
        "tremor_index":      trnorm(0.38, 0.12, 0.10, 0.75, n),
        "direction_changes": trnorm(12,   2.5,  6,    22,   n),
        "pen_pressure_cv":   trnorm(0.38, 0.08, 0.12, 0.65, n),
        "clock_radius_std":  trnorm(0.18, 0.06, 0.05, 0.40, n),
        "clock_digit_err":   trnorm(2.5,  0.9,  0.5,  5.5,  n),
    }


def add_engineered(df):
    df["fluency_score"]    = df["velocity_mean"]   / (df["jerk_mean"]        + 1e-5)
    df["tremor_pressure"]  = df["tremor_index"]    * df["pressure_std"]
    df["size_variability"] = df["letter_size_std"] / (df["letter_size_mean"] + 1e-5)
    df["pause_ratio"]      = df["pen_up_time_pct"] / (1 - df["pen_up_time_pct"] + 1e-5)
    return df


def inject_missing(df, rate=0.06):
    mask          = np.random.rand(*df.shape) < rate
    df_m          = df.mask(mask)
    df_m["label"] = df["label"]
    return df_m


# ──────────────────────────────────────────────
# GENERATE DATASET
# ──────────────────────────────────────────────
def generate_handwriting_dataset(save=True):
    healthy         = pd.DataFrame(gen_healthy(N_HEALTHY)); healthy["label"] = 0
    ad              = pd.DataFrame(gen_ad(N_AD));           ad["label"]      = 1
    df              = pd.concat([healthy, ad], ignore_index=True)
    df              = add_engineered(df)
    df              = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df              = inject_missing(df)

    if save:
        out = DATA_DIR / "handwriting_dataset.csv"
        df.to_csv(out, index=False)
        print(f"  ✓ Saved handwriting dataset → {out}  [{len(df)} rows × {len(df.columns)} cols]")

    return df


# ──────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────
def train_hw_model():
    print("\n" + "="*55)
    print("  HANDWRITING MODEL — DATASET GENERATION + TRAINING")
    print("="*55)

    # ── Generate / load dataset ──
    hw_path = DATA_DIR / "handwriting_dataset.csv"
    if hw_path.exists():
        df = pd.read_csv(hw_path)
        print(f"  ✓ Loaded existing handwriting dataset: {len(df)} rows")
    else:
        df = generate_handwriting_dataset(save=True)

    feature_names = [c for c in df.columns if c != "label"]
    X = df[feature_names].values.astype(float)
    y = df["label"].values.astype(float)

    # ── Split ──
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15, stratify=y_tmp, random_state=SEED)

    # ── Impute & scale ──
    imputer = KNNImputer(n_neighbors=5)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_val   = scaler.transform(imputer.transform(X_val))
    X_test  = scaler.transform(imputer.transform(X_test))

    print(f"  Train: {len(y_train)}  |  Val: {len(y_val)}  |  Test: {len(y_test)}")

    # ── Tensors ──
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=5)

    best_auc, no_improve = 0.0, 0
    best_state = None

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
        "modality":  "Handwriting",
        "auc":       roc_auc_score(y_np, test_probs),
        "accuracy":  accuracy_score(y_np, test_preds),
        "precision": precision_score(y_np, test_preds, zero_division=0),
        "recall":    recall_score(y_np, test_preds, zero_division=0),
        "f1":        f1_score(y_np, test_preds, zero_division=0),
    }

    print(f"\n  ── Handwriting Model Test Results ─────────────")
    for k, v in metrics.items():
        if k != "modality":
            print(f"  {k.capitalize():12s}: {v:.4f}")
    print(classification_report(y_np, test_preds,
                                   target_names=["Healthy", "AD"]))

    # ── Save ──
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_names":    feature_names,
        "imputer":          imputer,
        "scaler":           scaler,
    }, RESULT_DIR / "hw_model_complete.pt")
    print(f"  ✓ Saved → result/hw_model_complete.pt")

    return metrics


if __name__ == "__main__":
    generate_handwriting_dataset(save=True)
    train_hw_model()