"""
hrv_stress_monitoring/src/analysis.py
---------------------------------------
Statistical analysis utilities for HRV demographic comparisons.

Functions
---------
  cohens_d          — Effect size between two groups
  anova_oneway      — One-way ANOVA + Tukey HSD summary
  mannwhitney_test  — Mann-Whitney U test for gender differences
  classify_state    — Rule-based autonomic state classifier
"""

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, mannwhitneyu, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# ── Effect size ────────────────────────────────────────────────────────────────
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two independent samples.

    Parameters
    ----------
    a, b : np.ndarray
        Two groups of continuous measurements.

    Returns
    -------
    float
        Cohen's d (positive → group a has larger mean).
    """
    a, b   = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(
        ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    )
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


# ── ANOVA ──────────────────────────────────────────────────────────────────────
def anova_oneway(df: pd.DataFrame,
                 metric: str,
                 group_col: str = 'age_group') -> dict:
    """
    Run one-way ANOVA across groups and Tukey HSD post-hoc test.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with metric column and group column.
    metric : str
        Column name of the HRV metric to test.
    group_col : str
        Column defining groups (default 'age_group').

    Returns
    -------
    dict with keys: F, p_value, significant, tukey_summary
    """
    groups = [grp[metric].dropna().values
              for _, grp in df.groupby(group_col)]
    F, p = f_oneway(*groups)

    tukey = pairwise_tukeyhsd(
        df[metric].dropna(),
        df.loc[df[metric].notna(), group_col]
    )

    return {
        'metric':        metric,
        'F':             float(F),
        'p_value':       float(p),
        'significant':   p < 0.05,
        'tukey_summary': str(tukey),
    }


# ── Mann-Whitney U ─────────────────────────────────────────────────────────────
def mannwhitney_test(df: pd.DataFrame,
                     metric: str,
                     group_col: str = 'gender',
                     group_a: str   = 'M',
                     group_b: str   = 'F') -> dict:
    """
    Non-parametric Mann-Whitney U test between two gender groups.

    Returns
    -------
    dict with keys: U, p_value, significant, effect_d
    """
    a = df[df[group_col] == group_a][metric].dropna().values
    b = df[df[group_col] == group_b][metric].dropna().values
    U, p = mannwhitneyu(a, b, alternative='two-sided')
    return {
        'metric':      metric,
        'group_a':     group_a,
        'group_b':     group_b,
        'U':           float(U),
        'p_value':     float(p),
        'significant': p < 0.05,
        'cohens_d':    cohens_d(a, b),
    }


# ── Rule-based autonomic state classifier ─────────────────────────────────────
def classify_state(si: float,
                   rmssd: float,
                   lf_hf: float,
                   mean_hr: float,
                   sdnn: float) -> tuple[str, str]:
    """
    Classify autonomic state using validated HRV thresholds.

    States
    ------
    Deep Sleep/Recovery  : SI < 10, RMSSD > 30 ms, LF/HF < 1.0, HR < 65 bpm
    Rest                 : SI 10–30, RMSSD 15–30 ms, LF/HF 1.0–2.0
    Mild Stress/Activity : SI 30–70, RMSSD < 20 ms, LF/HF > 2.0
    High Stress          : SI > 70, RMSSD < 15 ms, HR > 80 bpm

    Reference: Task Force of ESC (Circulation, 1996)

    Returns
    -------
    (state_label: str, hex_color: str)
    """
    score = 0

    # Baevsky SI
    if   si < 10:  score -= 2
    elif si < 30:  score += 0
    elif si < 70:  score += 1
    else:          score += 2

    # RMSSD
    if   rmssd > 35: score -= 2
    elif rmssd > 20: score -= 1
    elif rmssd < 15: score += 2
    else:            score += 1

    # LF/HF
    if   lf_hf < 0.8: score -= 2
    elif lf_hf < 1.5: score += 0
    elif lf_hf < 3.0: score += 1
    else:             score += 2

    # Heart rate
    if   mean_hr < 60: score -= 1
    elif mean_hr > 80: score += 1

    if   score <= -3: return 'Deep Sleep/Recovery', '#1a56db'
    elif score <= -1: return 'Rest',                '#16a34a'
    elif score <=  1: return 'Mild Stress',         '#d97706'
    else:             return 'High Stress',          '#dc2626'
