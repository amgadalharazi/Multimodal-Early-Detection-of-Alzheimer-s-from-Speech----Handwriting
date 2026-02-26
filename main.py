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

# =========================
# LOAD DATA
# =========================
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "Datasets/nacc.csv", low_memory=False)

# Ensure output directory exists (FIX #7)
(BASE_DIR / "result").mkdir(parents=True, exist_ok=True)

# =========================
# CREATE SPEECH FEATURES IF NOT EXISTS
# =========================
speech_file = BASE_DIR / "Datasets/speech_features.csv"
if not speech_file.exists():
    print("Speech features file not found. Creating synthetic speech features...")
    unique_ids = df["NACCID"].unique()
    np.random.seed(42)
    speech_df = pd.DataFrame({
        "NACCID":      unique_ids,
        "mfcc_mean":   np.random.normal(0,   1,   len(unique_ids)),
        "mfcc_std":    np.random.normal(1,   0.3, len(unique_ids)),
        "pitch_mean":  np.random.normal(150, 30,  len(unique_ids)),
        "pitch_std":   np.random.normal(20,  5,   len(unique_ids)),
        "speech_rate": np.random.normal(3.5, 0.8, len(unique_ids))
    })
    speech_df.to_csv(speech_file, index=False)
    print(f"✓ Created {len(speech_df)} synthetic speech feature records")
else:
    speech_df = pd.read_csv(speech_file)
    print(f"✓ Loaded {len(speech_df)} speech feature records")

# =========================
# FILTER DATA
# =========================
df = df[df["PACKET"] == "I"]
df = df[df["NACCUDSD"].isin([1, 3])]
df["label"] = df["NACCUDSD"].map({1: 0, 3: 1})

# =========================
# MERGE SPEECH
# =========================
df = df.merge(speech_df, on="NACCID", how="inner")
print(f"✓ Merged dataset has {len(df)} records")

# =========================
# FEATURES
# =========================
clinical_features = [
    "NACCAGE", "SEX", "EDUC",
    "MMSELOC", "CDRGLOB",
    "MEMORY", "JUDGMENT",
    "COMMUN", "HOMEHOBB", "PERSCARE"
]
speech_features = [
    "mfcc_mean", "mfcc_std",
    "pitch_mean", "pitch_std",
    "speech_rate"
]

# =========================
# CLEAN DATA FIRST (FIX #3 — clean BEFORE feature engineering)
# =========================
# Replace sentinel/invalid values with NaN on raw columns only
raw_cols = clinical_features + speech_features
df[raw_cols] = df[raw_cols].replace([-4, -9, 88, 99], pd.NA)
df[raw_cols] = df[raw_cols].apply(pd.to_numeric, errors="coerce")

# =========================
# FEATURE ENGINEERING (FIX #3 — now done on clean values)
# =========================
df["age_education"]  = df["NACCAGE"]  * df["EDUC"]
df["mmse_cdr"]       = df["MMSELOC"]  * df["CDRGLOB"]
df["pitch_variance"] = df["pitch_std"] / (df["pitch_mean"] + 1e-5)

engineered_features = ["age_education", "mmse_cdr", "pitch_variance"]
all_features = clinical_features + speech_features + engineered_features

X = df[all_features].values
y = df["label"].values

# =========================
# TRAIN / VAL / TEST SPLIT  (FIX #1 — split BEFORE imputation/scaling)
# =========================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42
)

# =========================
# IMPUTATION — fit on train only (FIX #1)
# =========================
# KNN imputation fills missing values using the k nearest neighbours.
# Fitting only on the training set prevents test-set information from
# leaking into preprocessing.
imputer = KNNImputer(n_neighbors=5)
X_train = imputer.fit_transform(X_train)
X_val   = imputer.transform(X_val)
X_test  = imputer.transform(X_test)

# =========================
# NORMALIZATION — fit on train only (FIX #1)
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# =========================
# CONVERT TO TENSORS
# =========================
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

print(f"\nDataset splits:")
print(f"  Training:   {len(X_train_t)} samples")
print(f"  Validation: {len(X_val_t)} samples")
print(f"  Test:       {len(X_test_t)} samples")

# =========================
# DATALOADER (FIX #8 — mini-batch training for proper BatchNorm behaviour)
# =========================
BATCH_SIZE = 64
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =========================
# MODEL
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

model = MultimodalNet(X_train_t.shape[1])
print(f"✓ Model initialized with {X_train_t.shape[1]} input features")

# =========================
# HANDLE CLASS IMBALANCE
# =========================
pos_weight_val = float((y_train_t == 0).sum() / (y_train_t == 1).sum())
print(f"✓ Class imbalance ratio: {pos_weight_val:.2f}")
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val))  # FIX #4

# =========================
# OPTIMIZER & SCHEDULER
# =========================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)

# =========================
# TRAINING WITH EARLY STOPPING
# =========================
print("\n" + "="*50)
print("TRAINING")
print("="*50)

best_val_auc    = 0.0
patience        = 15
patience_counter = 0
num_epochs      = 150

for epoch in range(num_epochs):
    # --- Training phase (mini-batch) ---
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        # Gradient clipping to prevent exploding gradients (FIX #5)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    # --- Validation phase ---
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_probs  = torch.sigmoid(val_logits).numpy()
        val_auc    = roc_auc_score(y_val_t.numpy(), val_probs)

    # Learning rate scheduling
    old_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_auc)
    new_lr = optimizer.param_groups[0]["lr"]
    if old_lr != new_lr:
        print(f"  LR reduced: {old_lr:.6f} → {new_lr:.6f}")

    # Early stopping & checkpoint
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), BASE_DIR / "result/best_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

    if epoch % 10 == 0:
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

print(f"\n✓ Best Validation AUC: {best_val_auc:.4f}")

# =========================
# LOAD BEST MODEL (FIX #6 — weights_only=True)
# =========================
model.load_state_dict(
    torch.load(BASE_DIR / "result/best_model.pt", weights_only=True)
)

# =========================
# TEST SET EVALUATION
# =========================
print("\n" + "="*50)
print("TEST SET EVALUATION")
print("="*50)

model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_probs  = torch.sigmoid(test_logits).numpy()
    test_preds  = (test_probs >= 0.5).astype(int)

y_test_np = y_test_t.numpy()

print(f"\nROC-AUC Score:    {roc_auc_score(y_test_np, test_probs):.4f}")
print(f"Accuracy:         {accuracy_score(y_test_np, test_preds):.4f}")
print(f"Precision:        {precision_score(y_test_np, test_preds):.4f}")
print(f"Recall:           {recall_score(y_test_np, test_preds):.4f}")
print(f"F1-Score:         {f1_score(y_test_np, test_preds):.4f}")

print("\n" + "-"*50)
print("Confusion Matrix:")
print("-"*50)
cm = confusion_matrix(y_test_np, test_preds)
print(cm)
print(f"\nTrue Negatives:  {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives:  {cm[1,1]}")

print("\n" + "-"*50)
print("Classification Report:")
print("-"*50)
print(classification_report(y_test_np, test_preds,
                            target_names=["Healthy", "Alzheimer's"]))

# =========================
# CROSS-VALIDATION (FIX #2 — impute & scale per fold on raw X/y to prevent leakage,)
#                  (FIX #9 — early stopping inside each fold)
# =========================
print("\n" + "="*50)
print("CROSS-VALIDATION")
print("="*50)

# Use the raw (un-imputed, un-scaled) feature matrix so each fold
# preprocesses only its own training data, preventing leakage.
X_raw = df[all_features].values   # still contains NaNs from sentinel replacement
y_raw = df["label"].values

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_raw, y_raw)):
    print(f"\nFold {fold + 1}/5 ...")

    X_tr_raw, X_vl_raw = X_raw[train_idx], X_raw[val_idx]
    y_tr,     y_vl     = y_raw[train_idx],  y_raw[val_idx]

    # --- Per-fold imputation (fit on fold train only) ---
    fold_imputer = KNNImputer(n_neighbors=5)
    X_tr_imp = fold_imputer.fit_transform(X_tr_raw)
    X_vl_imp = fold_imputer.transform(X_vl_raw)

    # --- Per-fold scaling (fit on fold train only) ---
    fold_scaler = StandardScaler()
    X_tr_sc = fold_scaler.fit_transform(X_tr_imp)
    X_vl_sc = fold_scaler.transform(X_vl_imp)

    X_tr_t = torch.tensor(X_tr_sc, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr,    dtype=torch.float32)
    X_vl_t = torch.tensor(X_vl_sc, dtype=torch.float32)
    y_vl_t = torch.tensor(y_vl,    dtype=torch.float32)

    fold_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    fold_model     = MultimodalNet(X_tr_t.shape[1])
    fold_optimizer = torch.optim.Adam(fold_model.parameters(), lr=1e-3, weight_decay=1e-5)

    fold_pos_weight = float((y_tr_t == 0).sum() / (y_tr_t == 1).sum())
    fold_criterion  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(fold_pos_weight))

    # Early stopping inside CV fold (FIX #9)
    best_fold_auc    = 0.0
    fold_patience    = 10
    fold_patience_ctr = 0

    for epoch in range(150):
        fold_model.train()
        for X_batch, y_batch in fold_loader:
            fold_optimizer.zero_grad()
            logits = fold_model(X_batch)
            loss   = fold_criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fold_model.parameters(), max_norm=1.0)
            fold_optimizer.step()

        fold_model.eval()
        with torch.no_grad():
            vl_probs  = torch.sigmoid(fold_model(X_vl_t)).numpy()
            fold_auc  = roc_auc_score(y_vl_t.numpy(), vl_probs)

        if fold_auc > best_fold_auc:
            best_fold_auc    = fold_auc
            fold_patience_ctr = 0
        else:
            fold_patience_ctr += 1

        if fold_patience_ctr >= fold_patience:
            break

    cv_auc_scores.append(best_fold_auc)
    print(f"  Fold {fold + 1} Best AUC: {best_fold_auc:.4f}  (stopped at epoch {epoch})")

print("\n" + "-"*50)
print(f"Cross-Validation Mean AUC: {np.mean(cv_auc_scores):.4f} ± {np.std(cv_auc_scores):.4f}")
print("="*50)

# =========================
# SAVE FINAL MODEL & PREPROCESSING OBJECTS
# =========================
torch.save({
    "model_state_dict": model.state_dict(),
    "scaler":           scaler,
    "imputer":          imputer,
    "feature_names":    all_features
}, BASE_DIR / "result/alzheimer_model_complete.pt")

print("\n✓ Model saved successfully!")
print(f"✓ Location: {BASE_DIR / 'result/alzheimer_model_complete.pt'}")