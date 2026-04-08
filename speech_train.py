import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
import librosa

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATASET_PATH = "Datasets/healthy/dev-other"
RESULT_PATH = "result/speech_model_real.pt"
SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

np.random.seed(SEED)
torch.manual_seed(SEED)


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze()


# ──────────────────────────────────────────────
# FEATURE EXTRACTION (REAL)
# ──────────────────────────────────────────────
def extract_features(path):
    y, sr = librosa.load(path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    pitch = librosa.yin(y, fmin=50, fmax=300)

    duration = len(y) / sr
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    speech_rate = len(onsets) / (duration + 1e-5)

    intervals = librosa.effects.split(y, top_db=30)
    silence_ratio = 1 - (np.sum(intervals[:, 1] - intervals[:, 0]) / len(y))

    return [
        np.mean(mfcc),
        np.std(mfcc),
        np.mean(pitch),
        np.std(pitch),
        speech_rate,
        silence_ratio,
    ]


FEATURE_NAMES = [
    "mfcc_mean",
    "mfcc_std",
    "pitch_mean",
    "pitch_std",
    "speech_rate",
    "pause_rate",
]


# ──────────────────────────────────────────────
# LOAD DATASET
# ──────────────────────────────────────────────
def load_dataset(root):
    X = []

    for speaker in os.listdir(root):
        spk_path = os.path.join(root, speaker)

        # ✅ Skip non-directories
        if not os.path.isdir(spk_path):
            continue

        for chapter in os.listdir(spk_path):
            chap_path = os.path.join(spk_path, chapter)

            # ✅ Skip non-directories again
            if not os.path.isdir(chap_path):
                continue

            for file in os.listdir(chap_path):
                if not file.endswith(".flac"):
                    continue

                path = os.path.join(chap_path, file)

                try:
                    features = extract_features(path)
                    X.append(features)
                except Exception as e:
                    print(f"Skipping {path}: {e}")

    return np.array(X)


# ──────────────────────────────────────────────
# CREATE ANOMALY LABELS (TEMPORARY)
# ──────────────────────────────────────────────
def create_labels(X):
    # label 0 = normal
    # label 1 = artificial anomaly
    noise = np.random.normal(0, 0.5, X.shape)
    X_anomaly = X + noise

    X_full = np.vstack([X, X_anomaly])
    y = np.array([0] * len(X) + [1] * len(X_anomaly))

    return X_full, y


# ──────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────
def train():
    print("Loading dataset...")
    X = load_dataset(DATASET_PATH)

    print("Creating anomaly labels...")
    X, y = create_labels(X)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # preprocessing
    imputer = KNNImputer()
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test = scaler.transform(imputer.transform(X_test))

    # model
    model = Net(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    # training loop
    for epoch in range(EPOCHS):
        model.train()

        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # evaluation
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32)))
        preds = (preds.numpy() > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    print("Test Accuracy:", acc)

    # save
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "imputer": imputer,
            "scaler": scaler,
        },
        RESULT_PATH,
    )

    print("Model saved!")


if __name__ == "__main__":
    train()
