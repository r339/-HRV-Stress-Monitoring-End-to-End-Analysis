# 🫀 HRV Stress Monitoring — End-to-End Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: MIT-BIH NSR DB](https://img.shields.io/badge/Dataset-MIT--BIH%20NSR%20DB-red)](https://physionet.org/content/nsrdb/1.0.0/)

> **HRV-based physiological stress monitoring using signal processing, statistical analysis, ARIMA time-series modelling, and LSTM deep learning — applied to 18 subjects from the MIT-BIH Normal Sinus Rhythm Database.**

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Results](#-key-results)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results & Findings](#-results--findings)
- [Evaluation: Baseline vs Full Model](#-evaluation-baseline-vs-full-model)
- [References](#-references)

---

## 🔬 Project Overview

This project performs a comprehensive Heart Rate Variability (HRV) analysis to detect physiological stress states from ECG recordings. It covers the full pipeline:

1. **Signal Acquisition** — ECG download from PhysioNet via `wfdb`, R-peak detection, RR-interval extraction
2. **HRV Feature Engineering** — 10 HRV features (time-domain, frequency-domain, nonlinear)
3. **Exploratory Analysis** — Tachograms, Power Spectral Density, Poincaré plots, rolling HRV
4. **Statistical Analysis** — ANOVA, Tukey HSD, Mann-Whitney U, Cohen's d across age groups and genders
5. **Time-Series Modelling** — AR(2) baseline, then full ARIMA with AIC/BIC grid search, residual diagnostics (Ljung-Box)
6. **Classification** — Rule-based autonomic state classifier + 2-layer LSTM with demographic features
7. **Feature Importance** — Random Forest permutation importance
8. **Evaluation** — Baseline (Eval 1) vs full multi-subject model (Eval 2) comparison

---

## 🏆 Key Results

| Metric | Eval 1 (Baseline) | Eval 2 (Full Model) |
|--------|:-----------------:|:-------------------:|
| Subjects | 1 | 18 |
| Features | 10 HRV | 12 (HRV + demographics) |
| Labeling | Fixed SI=20 threshold | Per-subject median |
| Accuracy | ~95% *(misleading)* | **76.6%** |
| AUC | 0.52 *(random)* | **0.842** |
| Macro F1 | ~0.5 | **0.766** |

> **Key Insight:** Eval 1 had 97% "Stress" labels → model always predicted Stress → 95% accuracy but AUC = 0.52 (useless classifier). Per-subject median labelling in Eval 2 guaranteed class balance → AUC improved from 0.52 → **0.842**.

---

## 📁 Repository Structure

```
hrv-stress-monitoring/
│
├── notebooks/
│   └── HRV_Complete_Project.ipynb   # Full Colab-compatible notebook (all 10 sections)
│
├── src/
│   ├── features.py                  # HRV feature extraction functions
│   ├── models.py                    # LSTM model builder
│   ├── analysis.py                  # Statistical analysis utilities
│   └── arima_utils.py               # ARIMA grid search & diagnostics
│
├── docs/
│   └── methodology.md               # Detailed methodology notes
│
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | [PhysioNet MIT-BIH NSR DB](https://physionet.org/content/nsrdb/1.0.0/) |
| Subjects | 18 (9M / 9F) |
| Age range | 23 – 71 years |
| Sampling rate | 128 Hz |
| Duration per subject | 30 minutes |
| Age groups | Young (20–35): 6 · Middle (36–55): 8 · Older (56+): 4 |

Data is downloaded **automatically at runtime** via the `wfdb` library — no manual download required.

---

## 🔧 Methodology

### HRV Features (12 total)

| Domain | Feature | Description |
|--------|---------|-------------|
| Time | `mean_rr` | Mean RR interval (ms) |
| Time | `sdnn` | Standard deviation of NN intervals |
| Time | `rmssd` | Root mean square of successive differences |
| Time | `pnn50` | % of successive differences > 50 ms |
| Time | `cv` | Coefficient of variation |
| Frequency | `lf` | Low-frequency power (0.04–0.15 Hz) |
| Frequency | `hf` | High-frequency power (0.15–0.40 Hz) |
| Frequency | `lf_hf` | LF/HF ratio (sympathovagal balance) |
| Nonlinear | `sd1` | Poincaré SD1 (short-term variability) |
| Nonlinear | `sd2` | Poincaré SD2 (long-term variability) |
| Nonlinear | `si` | Baevsky Stress Index |
| Demographic | `age`, `gender_enc` | Subject demographics |

### Windowing Strategy
- **Window:** 60 beats (sliding)
- **Step:** 20 beats (66% overlap)
- **Labeling:** Per-subject median SI threshold → balanced binary Stress / Non-Stress labels

### LSTM Architecture
```
Input (60 timesteps × 12 features)
  └─► LSTM(64) + Dropout(0.3)
  └─► BatchNormalization
  └─► LSTM(32) + Dropout(0.3)
  └─► Dense(16, relu)
  └─► Dense(1, sigmoid)
```
- Optimizer: Adam | Loss: Binary Crossentropy
- Callbacks: EarlyStopping (patience=10), ReduceLROnPlateau

---

## 🛠 Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/hrv-stress-monitoring.git
cd hrv-stress-monitoring

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Or run directly in **Google Colab** (no setup needed):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/hrv-stress-monitoring/blob/main/notebooks/HRV_Complete_Project.ipynb)

---

## 🚀 Usage

### Run the full notebook

Open `notebooks/HRV_Complete_Project.ipynb` in Jupyter or Google Colab and run all cells top-to-bottom. The notebook:

- Automatically downloads ECG data from PhysioNet
- Runs all 10 analytical sections end-to-end
- Saves `hrv_dashboard.png` to the working directory

### Section guide

| Cell group | Section | What it does |
|---|---|---|
| 0.x | Setup | Install deps, imports, seeds |
| 1.x | Demographics | Subject registry, age/gender distribution |
| 2.x | Signal Processing | ECG download, R-peak detection, RR extraction |
| 3.x | Single-Subject EDA | Tachogram, PSD, Poincaré, rolling HRV |
| 4.x | HRV Dashboard | Multi-panel visualization saved to PNG |
| 5.x | ACF/PACF | Stationarity check, ACF/PACF plots |
| 6.x | Feature Extraction | 12-feature matrix across all 18 subjects |
| 7.x | Statistical Analysis | ANOVA, Tukey HSD, Mann-Whitney U, Cohen's d |
| 8.x | State Classifier | Rule-based autonomic state classifier |
| 9.x | LSTM Model | Training, evaluation, ROC, confusion matrix |
| 10.x | ARIMA & Summary | AR(2), full ARIMA, Eval 1 vs Eval 2 comparison |

---

## 📈 Results & Findings

### Demographic HRV Findings

- **SDNN declines ~31%** from Young → Older (autonomic nervous system aging)
- **RMSSD drops ~47%** from Young → Older (reduced vagal protection)
- **LF/HF doubles** Young → Older (sympathetic dominance with age)
- **Females** show higher RMSSD and HF power (stronger vagal tone vs males)
- ANOVA confirms significant age-group differences (p < 0.05) for SDNN, RMSSD, LF/HF

### Top Stress Biomarkers (Permutation Importance)

1. **lf_hf** — sympathovagal imbalance
2. **sd1** — reduced short-term variability
3. **hf** — reduced parasympathetic tone
4. **rmssd** — reduced vagal activity
5. **sdnn** — overall HRV reduction

### ARIMA Findings

- Best order selected by AIC/BIC grid search
- Residuals pass Ljung-Box test (white noise confirmed → model well-specified)
- AR φ₁ coefficient **decreases with age** → shorter HRV memory in older adults

---

## ⚖️ Evaluation: Baseline vs Full Model

```
=================================================================
  EVAL 1 vs EVAL 2 — COMPARISON SUMMARY
=================================================================
                 Eval 1 (Baseline)             Eval 2 (Full)
Subjects                 1 (16265)               18 subjects
Features                    10 HRV        12 (10+age+gender)
Labeling               Fixed SI=20        Per-subject median
Accuracy         ~95% (misleading)                     76.6%
AUC                 0.52 (random!)                     0.842
F1                            ~0.5                     0.766
Age Analysis                    NO         YES ANOVA + Tukey
Gender Analysis                 NO        YES Mann-Whitney U
ARIMA                   AR(2) only  YES Full ARIMA + AIC/BIC
Statistics                      NO             YES Cohen's d
```

---

## 📚 References

1. Task Force of the European Society of Cardiology (1996). *Heart rate variability: standards of measurement, physiological interpretation and clinical use.* Circulation, 93(5), 1043–1065.
2. Baevsky, R.M. (2002). *Analysis of Heart Rate Variability in Space Medicine.* Human Physiology.
3. PhysioNet / MIT-BIH Normal Sinus Rhythm Database. https://physionet.org/content/nsrdb/1.0.0/
4. Goldberger, A.L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23).

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*Built with Python · wfdb · SciPy · statsmodels · scikit-learn · TensorFlow · Matplotlib · Seaborn*
