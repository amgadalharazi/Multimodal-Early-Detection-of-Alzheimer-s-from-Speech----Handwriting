# 🧠 Multimodal Alzheimer's Detection

A research-grade machine learning pipeline that detects Alzheimer's Disease (AD) from **three independent modalities** — speech/clinical data, digital handwriting, and raw audio — using deep neural networks and a classical voting ensemble, fused into a single weighted ensemble model. Designed to be modular, reproducible, and extensible.

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
- [Ensemble & Fusion](#ensemble--fusion)
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

All three approaches produce independent probability estimates that are fused into a final AUC-weighted ensemble risk score stored in `result/final_model.pt`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Multimodal AD Detection                          │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Speech+Clinical  │  │ Digital          │  │ Raw Audio        │  │
│  │ (NACC + synth.   │  │ Handwriting      │  │ (librosa 60+     │  │
│  │  speech feats)   │  │ (200k synthetic) │  │  features)       │  │
│  │                  │  │                  │  │                  │  │
│  │  MultimodalNet   │  │  MultimodalNet   │  │  MultimodalNet   │  │
│  │  (4-layer MLP)   │  │  (4-layer MLP)   │  │  (4-layer MLP)   │  │
│  │  ────────────    │  │  ────────────    │  │  ────────────    │  │
│  │  Linear(→128)    │  │  Linear(→128)    │  │  Linear(→128)    │  │
│  │  BatchNorm+ReLU  │  │  BatchNorm+ReLU  │  │  BatchNorm+ReLU  │  │
│  │  Dropout(0.4)    │  │  Dropout(0.4)    │  │  Dropout(0.4)    │  │
│  │  Linear(→64)     │  │  Linear(→64)     │  │  Linear(→64)     │  │
│  │  Linear(→32)     │  │  Linear(→32)     │  │  Linear(→32)     │  │
│  │  Linear(→1)      │  │  Linear(→1)      │  │  Linear(→1)      │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
│           │  P(AD)              │  P(AD)              │  P(AD)     │
│           └─────────────────────┴──────────┬──────────┘            │
│                                            ▼                       │
│                              AUC-Weighted Ensemble                  │
│                              result/final_model.pt                 │
│                              Risk Score + Label                    │
└─────────────────────────────────────────────────────────────────────┘

Separate (speech_train.py):
  VotingClassifier (RF + SVM + GradientBoosting)
  60+ acoustic features via librosa
  Stratified K-Fold cross-validation
  → also exports a MultimodalNet .pt for fusion
```

**MultimodalNet** (shared architecture across all three MLP models):

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
│   ├── nacc.csv                    # NACC dataset (you must supply this)
│   ├── speech_features.csv         # Auto-generated on first run
│   └── handwriting_dataset.csv     # Auto-generated on first run
│
├── result/
│   ├── speech_model_complete.pt          # Speech + clinical checkpoint
│   ├── hw_model_complete.pt              # Handwriting checkpoint
│   ├── speech_audio_model_complete.pt    # Raw audio checkpoint
│   └── final_model.pt                    # Fused weighted ensemble
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

Combines structured clinical data from the [NACC Uniform Data Set](https://naccdata.org/) with synthetic speech proxies generated per subject. Only `PACKET == "I"` (initial visit) rows where `NACCUDSD ∈ {1, 3}` (Normal = 1, AD = 3) are used.

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

| Feature | Healthy range | AD range | Clinical signal |
|---------|--------------|----------|-----------------|
| `mfcc_mean` | ~0.2 | ~−1.5 | Timbral quality |
| `pitch_mean` (Hz) | ~155 | ~120 | Reduced prosody |
| `speech_rate` (syll/s) | ~4.0 | ~2.2 | Processing speed |
| `pause_rate` | ~0.8 | ~3.5 | Word-finding difficulty |
| `hnr_db` | ~18 | ~11 | Vocal breathiness |
| `jitter_pct` | ~0.5% | ~1.2% | Vocal cord instability |
| `shimmer_pct` | ~3.0% | ~6.5% | Amplitude variability |

**Engineered features:** `age_education`, `mmse_cdr`, `pitch_variance`

**Training details:**
- KNN Imputer (k=5) → StandardScaler → MultimodalNet
- Adam, lr=1e-3, weight_decay=1e-5
- ReduceLROnPlateau scheduler (patience=5)
- Early stopping (patience=15 epochs on val AUC)
- Class-weighted BCE loss

---

### Modality 2: Digital Handwriting

**File:** `hw.py`

Generates a balanced 200,000-row synthetic dataset (100k healthy / 100k AD) simulating pen-tablet writing tasks. Features are drawn from clinically-grounded distributions with ~6% MCAR missing values injected for realism.

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
- SimpleImputer (median) → StandardScaler → MultimodalNet
- Batch size 256, Adam lr=1e-3, weight_decay=1e-4
- Early stopping (patience=12 epochs)

> KNN imputation is intentionally skipped here — it would take many minutes on 200k rows; median imputation is equivalent on this synthetic dataset.

---

### Modality 3: Raw Audio (`speech_train.py`)

**File:** `speech_train.py`

A standalone, improved speech classifier for use when real `.wav`/`.mp3`/`.flac` recordings are available. Uses `librosa` for feature extraction and a sklearn voting ensemble, then also exports a compatible `MultimodalNet` `.pt` file for inclusion in `final_model.pt`.

**60+ acoustic features extracted:**

| Group | Count | Description |
|-------|-------|-------------|
| MFCCs (mean + std) | 80 | Timbral texture |
| Delta & Delta² MFCCs | 80 | Temporal dynamics |
| Chroma (mean + std) | 24 | Harmonic content |
| Spectral (contrast, rolloff, bandwidth, centroid, flatness) | 11 | Spectral shape |
| ZCR, RMS (mean + std) | 4 | Energy / noisiness |
| F0 (mean, std, range, voiced fraction) | 4 | Pitch characteristics |
| Pause analysis (count, mean/max duration, silence ratio) | 4 | Hesitation patterns |
| Jitter, Shimmer, Speaking rate | 3 | Voice quality |

**Audio preprocessing:** Silence trimming (`top_db=25`) + peak normalisation

**Data augmentation (optional, `AUGMENT=True`):** Gaussian noise, time-stretch (×0.9), pitch-shift (±1 semitone) — 4 augmented copies per file

**sklearn pipeline:** `StandardScaler → VotingClassifier(RF + SVM + GradientBoosting)`  
**Evaluation:** Stratified 5-Fold cross-validation + held-out test set

---

## Ensemble & Fusion

`main.py` (Step 4) loads all available `.pt` checkpoints and fuses them into `result/final_model.pt`. Models with higher ROC-AUC automatically receive a larger vote:

```
final_prob = Σ (prob_i × AUC_i) / Σ AUC_i
```

If a checkpoint is missing (e.g. no `train/` audio folder), it is silently skipped — the ensemble still runs with 1 or 2 models. Each sub-model's preprocessing objects (imputer, scaler) are saved inside `final_model.pt` so inference requires only that single file.

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
3. Required columns: `PACKET`, `NACCUDSD`, `NACCID`, `NACCAGE`, `SEX`, `EDUC`, `MMSELOC`, `CDRGLOB`, `MEMORY`, `JUDGMENT`, `COMMUN`, `HOMEHOBB`, `PERSCARE`

### Handwriting Dataset

Generated automatically on first run — no external data needed.
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

Runs all five steps in sequence:
1. Train the Speech + Clinical model on NACC data
2. Generate the handwriting dataset and train the handwriting model
3. Train the raw audio model (skipped automatically if `train/` is absent)
4. Fuse all checkpoints into `result/final_model.pt`
5. Run predictions on 3 demo patients and print a modality comparison table

### Individual Modules

```bash
# Train only the speech + clinical model
python speech.py

# Generate handwriting data + train handwriting model
python hw.py

# Train the raw-audio ensemble model (needs train/)
python speech_train.py

# Run predictions on 3 sample patients (needs trained models)
python test.py
```

### Predict from Real Audio

```bash
python test.py /path/to/recording.wav
```

The model will extract acoustic features from the audio file and return a probability and label. Clinical features are filled with sensible defaults — for best results, supplement with actual clinical values in the source code.

### CLI Flags

```bash
# Skip ALL training — combine existing checkpoints + predict only
python main.py --skip-train

# Skip only NACC speech training
python main.py --skip-speech

# Skip only handwriting training
python main.py --skip-hw

# Skip only audio training (no train/ folder required)
python main.py --skip-audio

# Force-regenerate the handwriting dataset from scratch
python main.py --regen-hw
```

---

## Results & Metrics

The pipeline prints a side-by-side comparison table after training. Actual values will vary by NACC dataset version and random seed.

```
══════════════════════════════════════════════════════════════════
  MODALITY COMPARISON — TEST SET RESULTS
══════════════════════════════════════════════════════════════════

  Metric         Speech+Clinical      Handwriting       Raw Audio
  -------------- -------------------- ---------------- ----------
  AUC                      0.97xx <         0.96xx        varies
  ACCURACY                 0.93xx           0.94xx <      varies
  PRECISION                0.92xx           0.94xx <      varies
  RECALL                   0.95xx <         0.93xx        varies
  F1                       0.93xx           0.94xx <      varies

  < = best score for that metric
```

**What to look for:**
- **AUC (ROC)** — primary metric; measures overall discriminative ability
- **Recall / Sensitivity** — most important for clinical screening; high recall = fewer missed cases
- **F1** — harmonic mean of precision and recall; balances both

The handwriting model benefits from the large balanced synthetic dataset (200k rows). The speech + clinical model reflects real-world clinical feature distributions from NACC. Raw audio performance is entirely dataset-dependent.

---

## Feature Engineering

Both MLP models use domain-informed engineered features computed before training:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `age_education` | `NACCAGE × EDUC` | Interaction term; higher education is protective |
| `mmse_cdr` | `MMSELOC × CDRGLOB` | Joint cognitive/functional severity |
| `pitch_variance` | `pitch_std / pitch_mean` | Normalised pitch variability |
| `fluency_score` | `velocity_mean / jerk_mean` | Smooth vs. fragmented writing motion |
| `tremor_pressure` | `tremor_index × pressure_std` | Combined motor instability measure |
| `size_variability` | `letter_size_std / letter_size_mean` | Normalised letter inconsistency |
| `pause_ratio` | `pen_up_pct / (1 − pen_up_pct)` | Logit-like pen-up ratio |

---

## Output Files

| File | Created by | Contents |
|------|-----------|---------|
| `Datasets/nacc.csv` | You | Raw NACC UDS dataset |
| `Datasets/speech_features.csv` | `speech.py` | Per-subject synthetic speech features |
| `Datasets/handwriting_dataset.csv` | `hw.py` | 200k-row synthetic handwriting dataset |
| `result/speech_model_complete.pt` | `speech.py` | Weights + imputer + scaler + feature names |
| `result/hw_model_complete.pt` | `hw.py` | Weights + imputer + scaler + feature names |
| `result/speech_audio_model_complete.pt` | `speech_train.py` | Weights + imputer + scaler + feature names |
| `result/final_model.pt` | `main.py` | Fused ensemble: all sub-model states + AUC weights |
| `model/alzheimer_pipeline.pkl` | `speech_train.py` | Full sklearn pipeline (scaler + RF+SVM+GBM) |
| `model/feature_names.pkl` | `speech_train.py` | Feature name list for inference |
| `plots/confusion_matrix.png` | `speech_train.py` | Test-set confusion matrix |
| `plots/roc_curve.png` | `speech_train.py` | ROC curve with AUC |
| `plots/feature_importance.png` | `speech_train.py` | Top-20 RF feature importances |
| `plots/pca_scatter.png` | `speech_train.py` | 2D PCA of speech features |

---

## Limitations

- **Synthetic data**: The handwriting dataset and speech features in `speech.py` are synthetically generated. Real-world pen-tablet and audio data would yield more clinically valid models.
- **NACC access**: The NACC dataset requires institutional registration and is not redistributable.
- **Audio model**: Performance depends entirely on the quality and size of the `.wav` dataset you provide.
- **No temporal modelling**: The MLP architecture does not capture longitudinal progression. A recurrent or transformer-based model could improve detection across follow-up visits.
- **Demographic bias**: Models trained on NACC reflect the demographic distribution of that cohort. Performance may vary across different populations.
- **Binary classification only**: The pipeline distinguishes Healthy vs. AD. It does not classify MCI subtypes or other dementia forms.
- **Not clinically validated**: These models have not undergone clinical validation and must not be used for diagnosis.

---

## Citation

If you use this code in your research, please cite appropriately and acknowledge the NACC dataset:

```bibtex
@dataset{nacc_uds,
  title  = {National Alzheimer's Coordinating Center — Uniform Data Set},
  author = {NACC},
  url    = {https://naccdata.org},
  note   = {Data collected through funding from NIA/NIH Grant U24 AG072122}
}
```

Related work this pipeline draws on:

- Luz et al., *ADReSS Challenge: Speech-based Automatic AD Detection*, INTERSPEECH 2020
- Impedovo & Pirlo, *Dynamic Handwriting Analysis for the Assessment of Neurodegenerative Diseases*, IEEE Reviews in Biomedical Engineering, 2019

---

## License

MIT License. See `LICENSE` for details.

---

*Built with PyTorch, scikit-learn, librosa, and the NACC Uniform Data Set.*