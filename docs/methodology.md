# Methodology Notes — HRV Stress Monitoring

## 1. Data Acquisition

ECG recordings are streamed from [PhysioNet](https://physionet.org/content/nsrdb/1.0.0/) using the `wfdb` Python library. No local download is required. Each record provides a 30-minute, 128 Hz ECG.

**Subjects:** 18 (9M / 9F), ages 23–71 years.

---

## 2. R-Peak Detection & RR Intervals

```
ECG signal (128 Hz)
    │
    ▼
scipy.signal.find_peaks(distance=50, height=threshold)
    │
    ▼
R-peak timestamps (samples) → RR intervals (ms)
    │
    ▼
Artifact rejection: keep RR in [300, 2000] ms
```

---

## 3. HRV Feature Engineering

### Time Domain
| Feature | Formula |
|---------|---------|
| mean_rr | mean(RR) |
| SDNN | std(RR) |
| RMSSD | √(mean(ΔRR²)) |
| pNN50 | % \|ΔRR\| > 50 ms |
| CV | SDNN / mean_rr × 100 |

### Frequency Domain (Welch PSD)
RR series → cubic-spline interpolation at 4 Hz → Welch PSD (Hann window, 50% overlap)

| Band | Range | Feature |
|------|-------|---------|
| VLF | 0.003–0.04 Hz | (informational) |
| LF | 0.04–0.15 Hz | `lf` |
| HF | 0.15–0.40 Hz | `hf` |
| Ratio | — | `lf_hf` |

### Nonlinear (Poincaré)
- **SD1** = std((RR[n+1] − RR[n]) / √2) → short-term variability
- **SD2** = std((RR[n+1] + RR[n]) / √2) → long-term variability
- **SI (Baevsky)** = AMo / (2 × Mo × MxDMn) → stress index

---

## 4. Labelling Strategy

### Eval 1 (Baseline — flawed)
- Single subject (16265)
- Fixed threshold: SI > 20 → Stress
- Result: 97% Stress labels → degenerate classifier (AUC = 0.52)

### Eval 2 (Full — valid)
- All 18 subjects
- **Per-subject median SI** as threshold → guaranteed 50/50 class balance
- Result: AUC = 0.842, F1 = 0.766

---

## 5. LSTM Architecture

```
Input (batch, 1, 12)
  └─► Reshape to (batch, window, features) via windowing
  └─► LSTM(64, return_sequences=True)
  └─► Dropout(0.3)
  └─► BatchNormalization
  └─► LSTM(32)
  └─► Dropout(0.3)
  └─► Dense(16, relu)
  └─► Dense(1, sigmoid)

Loss      : Binary Crossentropy
Optimizer : Adam (lr=1e-3)
Callbacks : EarlyStopping(patience=10), ReduceLROnPlateau(patience=5)
Split     : 80% train / 20% test (stratified)
```

---

## 6. Statistical Analysis

| Test | Purpose |
|------|---------|
| One-way ANOVA | Age group differences in HRV metrics |
| Tukey HSD post-hoc | Pairwise group comparisons |
| Mann-Whitney U | Gender differences (non-parametric) |
| Cohen's d | Effect size quantification |

---

## 7. ARIMA Modelling

**Baseline:** AR(2) fit per age group → compare φ₁ coefficients

**Full pipeline:**
1. Stationarity check (ADF test)
2. ACF/PACF inspection for order hints
3. Grid search over (p ∈ 0–3) × (d ∈ 0–1) × (q ∈ 0–3)
4. Select best order by AIC (BIC for comparison)
5. Residual diagnostics: Ljung-Box test (H₀: white noise)

**Finding:** AR φ₁ coefficient decreases with age → shorter HRV autocorrelation memory in older adults, consistent with autonomic decline.

---

## 8. Rule-Based State Classifier

Scoring scheme (validated against ESC 1996 guidelines):

| Feature | Threshold | Score |
|---------|-----------|-------|
| SI | < 10 | −2 |
| SI | 10–30 | 0 |
| SI | 30–70 | +1 |
| SI | > 70 | +2 |
| RMSSD | > 35 ms | −2 |
| RMSSD | > 20 ms | −1 |
| RMSSD | < 15 ms | +2 |
| LF/HF | < 0.8 | −2 |
| LF/HF | 0.8–1.5 | 0 |
| LF/HF | 1.5–3.0 | +1 |
| LF/HF | > 3.0 | +2 |
| HR | > 80 bpm | +1 |

**State mapping:**  score ≤ −3 → Deep Sleep/Recovery · ≤ −1 → Rest · ≤ 1 → Mild Stress · > 1 → High Stress

---

## References

1. Task Force, ESC (1996). *Heart Rate Variability.* Circulation 93(5).
2. Baevsky R.M. (2002). *Analysis of HRV in Space Medicine.* Human Physiology.
3. PhysioNet MIT-BIH NSR DB. https://physionet.org/content/nsrdb/1.0.0/
