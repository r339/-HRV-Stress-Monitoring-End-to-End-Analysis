"""
hrv_stress_monitoring/src/features.py
--------------------------------------
HRV feature extraction utilities extracted from the main notebook.

Features computed per 60-beat window:
  Time domain   : mean_rr, sdnn, rmssd, pnn50, cv
  Frequency     : lf, hf, lf_hf  (Welch PSD, cubic-spline interpolated)
  Nonlinear     : sd1, sd2, si (Baevsky Stress Index)
  Demographic   : age, gender_enc (added at the DataFrame level)
"""

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

# ── Window parameters ──────────────────────────────────────────────────────────
WINDOW = 60   # beats per window
STEP   = 20   # stride (66% overlap)

# ── Baevsky Stress Index ───────────────────────────────────────────────────────
def baevsky_si(rr: np.ndarray) -> float:
    """
    Compute the Baevsky Stress Index (SI).

    SI = AMo / (2 * Mo * MxDMn)
      AMo  — Amplitude Mode  (% of most common RR bin)
      Mo   — Mode            (most common RR value, ms)
      MxDMn — Variation range (max - min RR)

    Parameters
    ----------
    rr : np.ndarray
        RR intervals in milliseconds.

    Returns
    -------
    float
        Stress Index; higher values indicate higher sympathetic dominance.
    """
    if len(rr) < 10:
        return 0.0
    hist, edges = np.histogram(rr, bins=min(50, len(rr) // 2))
    bin_centers  = (edges[:-1] + edges[1:]) / 2
    mo           = float(bin_centers[np.argmax(hist)])
    amo          = float(hist.max()) / len(rr) * 100.0
    var_range    = float(rr.max() - rr.min())
    if mo == 0 or var_range == 0:
        return 0.0
    return amo / (2.0 * mo * var_range)


# ── Per-window feature extraction ─────────────────────────────────────────────
def extract_features_window(rr_win, fs_interp: float = 4.0) -> dict | None:
    """
    Extract 10 HRV features from a single 60-beat RR window.

    Parameters
    ----------
    rr_win : array-like
        RR intervals (ms) for one window (ideally 60 beats).
    fs_interp : float
        Interpolation frequency for Welch PSD (default 4 Hz).

    Returns
    -------
    dict or None
        Dictionary of feature names → values, or None if window too short.
    """
    rr = np.array(rr_win, dtype=float)
    if len(rr) < 20:
        return None

    d       = np.diff(rr)
    mean_rr = np.mean(rr)

    # ── Time domain ──────────────────────────────────────────────────────────
    sdnn  = float(np.std(rr, ddof=1))
    rmssd = float(np.sqrt(np.mean(d ** 2)))
    pnn50 = float(100 * np.sum(np.abs(d) > 50) / len(d))
    cv    = sdnn / mean_rr * 100.0

    # ── Frequency domain (Welch PSD on cubic-spline interpolated signal) ─────
    try:
        t_rr   = np.cumsum(rr) / 1000.0
        t_uni  = np.arange(t_rr[0], t_rr[-1], 1.0 / fs_interp)
        rr_uni = interp1d(t_rr, rr, kind='cubic', fill_value='extrapolate')(t_uni)
        nperseg = min(64, max(8, len(rr_uni) // 4))
        freqs, psd = signal.welch(
            rr_uni, fs=fs_interp, nperseg=nperseg,
            window='hann', detrend='linear'
        )

        def bandpower(lo: float, hi: float) -> float:
            idx = (freqs >= lo) & (freqs <= hi)
            return float(np.trapezoid(psd[idx], freqs[idx])) if idx.sum() > 1 else 0.0

        lf    = bandpower(0.04, 0.15)
        hf    = bandpower(0.15, 0.40)
        lf_hf = lf / hf if hf > 0 else 0.0
    except Exception:
        lf, hf, lf_hf = 0.0, 0.0, 0.0

    # ── Nonlinear ─────────────────────────────────────────────────────────────
    rr1, rr2 = rr[:-1], rr[1:]
    sd1 = float(np.std((rr2 - rr1) / np.sqrt(2), ddof=1))
    sd2 = float(np.std((rr2 + rr1) / np.sqrt(2), ddof=1))
    si  = baevsky_si(rr)

    return {
        'mean_rr': mean_rr,
        'sdnn':    sdnn,
        'rmssd':   rmssd,
        'pnn50':   pnn50,
        'cv':      cv,
        'lf':      lf,
        'hf':      hf,
        'lf_hf':   lf_hf,
        'sd1':     sd1,
        'sd2':     sd2,
        'si':      si,
    }


# ── Sliding-window feature matrix for one subject ─────────────────────────────
def build_feature_matrix(rr_series: np.ndarray,
                         window: int = WINDOW,
                         step: int   = STEP) -> list[dict]:
    """
    Apply sliding windows over a full RR series and extract features.

    Parameters
    ----------
    rr_series : np.ndarray
        Full RR interval series (ms) for one subject.
    window : int
        Window size in beats.
    step : int
        Stride in beats.

    Returns
    -------
    list[dict]
        List of feature dicts (one per window).
    """
    rows = []
    for start in range(0, len(rr_series) - window + 1, step):
        feat = extract_features_window(rr_series[start: start + window])
        if feat is not None:
            rows.append(feat)
    return rows
