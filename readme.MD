# 🧠 Multimodal Alzheimer's Detection

A research-grade machine learning pipeline that detects Alzheimer's Disease (AD) from **two independent modalities** — speech/clinical data and digital handwriting — using deep neural networks and a classical voting ensemble. Designed to be modular, reproducible, and extensible.

> ⚠️ **Medical Disclaimer**: This is a research tool only. It is not a diagnostic device and must not be used as a substitute for professional medical evaluation. Always consult a qualified healthcare provider.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Modalities](#modalities)
  - [Modality 1: Speech + Clinical (NACC)](#modality-1-speech--clinical-nacc)
  - [Modality 2: Digital Handwriting](#modality-2-digital-handwriting)
  - [Modality 3: Raw Audio (speech_train.py)](#modality-3-raw-audio-speech_trainpy)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
  - [Full Pipeline](#full-pipeline)
  - [Individual Modules](#individual-modules)
  - [Predict from Real Audio](#predict-from-real-audio)
  - [CLI Flags](#cli-flags)
- [Results & Metrics](#results--metrics)
- [Feature Engineering](#feature-engineering)
- [Output Files](#output-files)
- [Limitations](#limitations)
- [Citation](#citation)

---

## Overview

Alzheimer's Disease leaves measurable traces in how people **speak** and **write** — long before a formal clinical diagnosis. This project explores whether those signals can be reliably detected by combining:

- **Clinical biomarkers** (age, education, MMSE score, CDR ratings) from the NACC dataset
- **Synthetic speech proxies** (pitch, MFCCs, pause rate, jitter, shimmer) merged onto NACC subjects
- **Digital handwriting kinematics** (pressure, velocity, tremor, pen lifts) from a synthetic dataset
- **Raw audio features** (60+ acoustic features extracted with `librosa`) for an improved standalone speech classifier

All three approaches produce independent probability estimates that can be ensembled into a final risk score.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multimodal AD Detection                      │
│                                                                 │
│  ┌──────────────────────┐    ┌──────────────────────────────┐   │
│  │  Speech + Clinical   │    │   Digital Handwriting        │   │
│  │  (NACC + synthetic   │    │   (200,000-row synthetic     │   │
│  │   speech features)   │    │    dataset)                  │   │
│  │                      │    │                              │   │
│  │  MultimodalNet       │    │  MultimodalNet               │   │
│  │  (4-layer MLP)       │    │  (4-layer MLP)               │   │
│  │  ──────────────      │    │  ──────────────              │   │
│  │  Linear(→128)        │    │  Linear(→128)                │   │
│  │  BatchNorm + ReLU    │    │  BatchNorm + ReLU            │   │
│  │  Dropout(0.4)        │    │  Dropout(0.4)                │   │
│  │  Linear(→64)         │    │  Linear(→64)                 │   │
│  │  Linear(→32)         │    │  Linear(→32)                 │   │
│  │  Linear(→1)          │    │  Linear(→1)                  │   │
│  │  BCEWithLogitsLoss   │    │  BCEWithLogitsLoss           │   │
│  └──────────┬───────────┘    └─────────────┬────────────────┘   │
│             │  P(AD)                        │  P(AD)             │
│             └───────────────┬───────────────┘                    │
│                             ▼                                   │
│                    Ensemble Average                             │
│                    Risk Score + Label                           │
└─────────────────────────────────────────────────────────────────┘

Separate:  speech_train.py
           VotingClassifier (RF + SVM + GradientBoosting)
           60+ acoustic features via librosa
           Stratified K-Fold cross-validation
```

**MultimodalNet** (shared architecture for both MLP models):
| Layer | Size | Extras |
|-------|------|--------|
| Input | `d` features | — |
| Hidden 1 | 128 | BatchNorm, ReLU, Dropout(0.4) |
| Hidden 2 | 64 | BatchNorm, ReLU, Dropout(0.3) |
| Hidden 3 | 32 | BatchNorm, ReLU, Dropout(0.2) |
| Output | 1 | BCEWithLogitsLoss, sigmoid at inference |

---

## Project Structure

```
.
├── main.py               # Pipeline orchestrator — run this first
├── speech.py             # NACC clinical + synthetic speech → MultimodalNet
├── hw.py                 # Synthetic handwriting dataset + MultimodalNet
├── test.py               # Load saved models, predict on 3 sample patients
├── speech_train.py       # Standalone improved speech classifier (raw audio)
│
├── Datasets/
│   ├── nacc.csv          # NACC dataset (you must supply this)
│   ├── speech_features.csv   # Auto-generated on first run
│   └── handwriting_dataset.csv  # Auto-generated on first run
│
├── result/
│   ├── speech_model_complete.pt  # Saved speech+clinical checkpoint
│   └── hw_model_complete.pt      # Saved handwriting checkpoint
│
├── model/                # Outputs from speech_train.py
│   ├── alzheimer_pipeline.pkl
│   ├── labels.pkl
│   └── feature_names.pkl
│
├── train/                # Required by speech_train.py
│   ├── healthy/          # .wav / .mp3 / .flac audio files
│   └── Alzheimer/        # .wav / .mp3 / .flac audio files
│
└── plots/                # Auto-generated diagnostic plots
    ├── pca_scatter.png
    ├── mfcc_example.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
```

---

## Modalities

### Modality 1: Speech + Clinical (NACC)

**File:** `speech.py`

Combines structured clinical data from the [NACC Uniform Data Set](https://naccdata.org/) with synthetic speech proxies generated per subject.

**Clinical features used:**

| Feature | Description |
|---------|-------------|
| `NACCAGE` | Age at visit |
| `SEX` | Biological sex |
| `EDUC` | Years of education |
| `MMSELOC` | MMSE total score |
| `CDRGLOB` | Clinical Dementia Rating (global) |
| `MEMORY` | CDR memory subscale |
| `JUDGMENT` | CDR judgment subscale |
| `COMMUN` | CDR community affairs |
| `HOMEHOBB` | CDR home & hobbies |
| `PERSCARE` | CDR personal care |

**Synthetic speech features:**

| Feature | Healthy range | AD range |
|---------|--------------|----------|
| `mfcc_mean` | ~0.2 | ~−1.5 |
| `pitch_mean` (Hz) | ~155 | ~120 |
| `speech_rate` (syll/s) | ~4.0 | ~2.2 |
| `pause_rate` | ~0.8 | ~3.5 |
| `hnr_db` | ~18 | ~11 |
| `jitter_pct` | ~0.5% | ~1.2% |
| `shimmer_pct` | ~3.0% | ~6.5% |

**Engineered features:** `age_education`, `mmse_cdr`, `pitch_variance`

**Training details:**
- KNN Imputer (k=5) → StandardScaler → MultimodalNet
- Adam optimiser, lr=1e-3, weight_decay=1e-5
- ReduceLROnPlateau scheduler (patience=5)
- Early stopping (patience=15 epochs)
- Class-weighted BCE loss

---

### Modality 2: Digital Handwriting

**File:** `hw.py`

Generates a balanced 200,000-row synthetic dataset simulating pen-tablet writing tasks. Features are drawn from clinically-grounded distributions.

**Feature categories and key signals:**

| Category | Features | Key AD signal |
|----------|----------|---------------|
| Pen kinematics | `pressure_mean/std`, `velocity_mean/std`, `acceleration_mean`, `jerk_mean` | Lower velocity, higher jerk |
| Temporal | `writing_duration`, `num_pen_lifts`, `pen_up_time_pct`, `inter_stroke_pause` | Longer pauses, more lifts |
| Spatial | `letter_size_mean/std`, `word_spacing`, `stroke_length`, `slant_angle` | Smaller, more variable letters |
| Tremor | `tremor_index`, `direction_changes`, `pen_pressure_cv` | Higher tremor index |
| Clock-drawing | `clock_radius_std`, `clock_digit_err` | Irregular radius, more errors |

**Engineered features:** `fluency_score`, `tremor_pressure`, `size_variability`, `pause_ratio`

**Training details:**
- SimpleImputer (median strategy) → StandardScaler → MultimodalNet
- Batch size 256, Adam lr=1e-3, weight_decay=1e-4
- Early stopping (patience=12 epochs)
- ~6% missing value injection for realism

---

### Modality 3: Raw Audio (`speech_train.py`)

**File:** `speech_train.py`

A standalone, improved speech classifier for use when raw `.wav`/`.mp3`/`.flac` audio recordings are available. Uses `librosa` for feature extraction and a sklearn voting ensemble.

**60+ acoustic features extracted:**

- **MFCCs**: 40 mean + 40 std (80 values)
- **Delta & Delta² MFCCs**: 40 + 40 temporal derivatives
- **Chroma**: 12 mean + 12 std
- **Spectral**: contrast (×7), rolloff, bandwidth, centroid, flatness
- **Prosodic**: F0 mean/std/range, voiced fraction, speaking rate
- **Voice quality**: jitter, shimmer
- **Pause analysis**: pause count, mean/max duration, silence ratio
- **Energy**: ZCR mean/std, RMS mean/std

**Ensemble classifier:**
```
StandardScaler →
  VotingClassifier (soft vote, equal weights):
    ├── RandomForestClassifier     (300 trees, balanced)
    ├── GradientBoostingClassifier (200 estimators, lr=0.05)
    └── SVC                        (RBF kernel, C=10, balanced)
```

**Data augmentation applied during loading:**
- Gaussian noise injection
- Time stretching (0.9×, mirroring slower AD speech)
- Pitch shifting (±1 semitone)

**Evaluation:** 5-fold stratified cross-validation + held-out test set + ROC curve + confusion matrix + feature importance plot

---

## Installation

### Requirements

- Python 3.10+
- PyTorch ≥ 2.0
- scikit-learn ≥ 1.3
- librosa ≥ 0.10
- pandas, numpy, soundfile, joblib, seaborn, matplotlib

### Install

```bash
# Clone the repo
git clone https://github.com/your-username/multimodal-alzheimers-detection.git
cd multimodal-alzheimers-detection

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install torch scikit-learn librosa pandas numpy soundfile joblib seaborn matplotlib
```

---

## Dataset Setup

### NACC Dataset (required for `speech.py` / `main.py`)

1. Register at [https://naccdata.org](https://naccdata.org) and request access to the Uniform Data Set (UDS).
2. Download the combined CSV and place it at:
   ```
   Datasets/nacc.csv
   ```
3. The pipeline uses only `PACKET == "I"` (initial visit) rows where `NACCUDSD ∈ {1, 3}` (Normal cognition = 1, AD = 3).

### Handwriting Dataset

Generated automatically on first run. No external data needed.
```
Datasets/handwriting_dataset.csv   ← created by hw.py
```

### Audio Dataset (for `speech_train.py` only)

Organise audio files into two folders:
```
train/
├── healthy/        ← recordings from healthy controls
│   ├── subject001.wav
│   └── ...
└── Alzheimer/      ← recordings from AD patients
    ├── subject101.wav
    └── ...
```
Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`

Public audio datasets you can use:
- [DementiaBank Pitt Corpus](https://dementia.talkbank.org/) (requires registration)
- [ADReSS Challenge](https://luzs.gitlab.io/adress/) (INTERSPEECH 2020)

---

## Usage

### Full Pipeline

```bash
python main.py
```

This runs all three steps in sequence:
1. Train the Speech + Clinical model on NACC data
2. Generate the handwriting dataset and train the handwriting model
3. Run predictions on 3 demo patients
4. Print a side-by-side modality comparison table

### Individual Modules

```bash
# Train only the speech+clinical model
python speech.py

# Generate handwriting data + train handwriting model
python hw.py

# Run predictions on 3 sample patients (requires trained models)
python test.py

# Train the raw-audio ensemble model
python speech_train.py
```

### Predict from Real Audio

```bash
# Predict AD vs Healthy from a .wav file
python test.py /path/to/recording.wav
```

The model will extract acoustic features from the audio file and return a probability and label. Note: clinical features will be filled with sensible defaults — for best results, supplement with actual clinical values in the source code.

### CLI Flags

```bash
# Skip training, use existing checkpoints (fast re-run)
python main.py --skip-train

# Skip only speech training
python main.py --skip-speech

# Skip only handwriting training
python main.py --skip-hw

# Force-regenerate the handwriting dataset
python main.py --regen-hw
```

---

## Results & Metrics

The pipeline prints a comparison table like the one below after training. Actual values will vary by NACC dataset version and random seed.

```
══════════════════════════════════════════════════════
  MODALITY COMPARISON — TEST SET RESULTS
══════════════════════════════════════════════════════

  Metric         Speech + Clinical       Handwriting
  -------------- ---------------------- --------------
  AUC                         0.97xx <       0.96xx
  ACCURACY                    0.93xx         0.94xx <
  PRECISION                   0.92xx         0.94xx <
  RECALL                      0.95xx <       0.93xx
  F1                          0.93xx         0.94xx <

  < = better score for that metric
```

**What to look for:**
- **AUC (ROC)** — primary metric; measures overall discriminative ability
- **Recall/Sensitivity** — most important for clinical screening; high recall = fewer missed cases
- **F1** — harmonic mean of precision and recall; balances both

The handwriting model benefits from the large balanced synthetic dataset (200k rows). The speech+clinical model reflects real-world clinical feature distributions from NACC.

---

## Feature Engineering

Both models use domain-informed engineered features computed before training:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `age_education` | `NACCAGE × EDUC` | Interaction term; higher education is protective |
| `mmse_cdr` | `MMSELOC × CDRGLOB` | Joint cognitive/functional severity |
| `pitch_variance` | `pitch_std / pitch_mean` | Normalised pitch variability |
| `fluency_score` | `velocity_mean / jerk_mean` | Smooth vs. fragmented writing motion |
| `tremor_pressure` | `tremor_index × pressure_std` | Combined instability measure |
| `size_variability` | `letter_size_std / letter_size_mean` | Normalised letter inconsistency |
| `pause_ratio` | `pen_up_pct / (1 − pen_up_pct)` | Logit-like pen-up ratio |

---

## Output Files

| File | Created by | Contents |
|------|-----------|---------|
| `Datasets/nacc.csv` | You | Raw NACC dataset |
| `Datasets/speech_features.csv` | `speech.py` | Per-subject synthetic speech features |
| `Datasets/handwriting_dataset.csv` | `hw.py` | 200k-row synthetic handwriting dataset |
| `result/speech_model_complete.pt` | `speech.py` | Model weights + imputer + scaler + feature names |
| `result/hw_model_complete.pt` | `hw.py` | Model weights + imputer + scaler + feature names |
| `model/alzheimer_pipeline.pkl` | `speech_train.py` | Full sklearn pipeline (scaler + ensemble) |
| `model/feature_names.pkl` | `speech_train.py` | Feature name list for inference |
| `plots/pca_scatter.png` | `speech_train.py` | 2D PCA of speech features |
| `plots/confusion_matrix.png` | `speech_train.py` | Test-set confusion matrix |
| `plots/roc_curve.png` | `speech_train.py` | ROC curve with AUC |
| `plots/feature_importance.png` | `speech_train.py` | Top-20 RF feature importances |

---

## Limitations

- **Synthetic data**: The handwriting dataset is entirely synthetic. Real-world pen-tablet data would yield more clinically valid models.
- **Speech proxies**: The speech features in `speech.py` are synthetically assigned to NACC subjects — not extracted from real recordings. Use `speech_train.py` with real audio for genuine acoustic modelling.
- **NACC access**: The NACC dataset requires institutional registration and is not redistributable.
- **No temporal modelling**: The MLP architecture does not capture longitudinal progression. A recurrent or transformer-based model could improve detection in follow-up visit data.
- **Demographic bias**: Models trained on NACC reflect the demographic distribution of that cohort. Performance may vary across different populations.
- **Not validated clinically**: These models have not undergone clinical validation and should not be used for diagnosis.

---

## Citation

If you use this code in your research, please cite appropriately and acknowledge the NACC dataset:

```bibtex
@dataset{nacc_uds,
  title     = {National Alzheimer's Coordinating Center — Uniform Data Set},
  author    = {NACC},
  url       = {https://naccdata.org},
  note      = {Data collected through funding from NIA/NIH Grant U24 AG072122}
}
```

---

## License

MIT License. See `LICENSE` for details.

---

*Built with PyTorch, scikit-learn, librosa, and the NACC Uniform Data Set.*