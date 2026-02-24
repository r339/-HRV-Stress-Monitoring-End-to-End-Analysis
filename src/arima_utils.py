"""
hrv_stress_monitoring/src/arima_utils.py
-----------------------------------------
ARIMA utilities for HRV time-series modelling.

Functions
---------
  arima_grid_search   — AIC/BIC grid search over (p, d, q) orders
  fit_arima           — Fit best ARIMA, return results + Ljung-Box diagnostics
  ar2_forecast        — Simple AR(2) forecast (baseline method)
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings('ignore')


# ── AIC/BIC grid search ───────────────────────────────────────────────────────
def arima_grid_search(series: np.ndarray,
                      p_range: range = range(0, 4),
                      d_range: range = range(0, 2),
                      q_range: range = range(0, 4),
                      criterion: str = 'aic') -> pd.DataFrame:
    """
    Grid search over ARIMA(p, d, q) orders and rank by AIC or BIC.

    Parameters
    ----------
    series : np.ndarray
        Univariate time series (RR intervals in ms).
    p_range, d_range, q_range : range
        Search ranges for AR, integration, and MA orders.
    criterion : str
        'aic' or 'bic'.

    Returns
    -------
    pd.DataFrame
        Sorted results with columns [p, d, q, aic, bic, converged].
    """
    results = []
    for p in p_range:
        for d in d_range:
            for q in q_range:
                if p == 0 and q == 0:
                    continue
                try:
                    res = ARIMA(series, order=(p, d, q)).fit(method='innovations_mle')
                    results.append({
                        'p': p, 'd': d, 'q': q,
                        'aic': res.aic, 'bic': res.bic,
                        'converged': True,
                    })
                except Exception:
                    results.append({'p': p, 'd': d, 'q': q,
                                    'aic': np.inf, 'bic': np.inf,
                                    'converged': False})

    df = pd.DataFrame(results)
    return df.sort_values(criterion).reset_index(drop=True)


# ── Fit best ARIMA + diagnostics ──────────────────────────────────────────────
def fit_arima(series: np.ndarray, order: tuple) -> dict:
    """
    Fit ARIMA with the given order and run Ljung-Box residual test.

    Parameters
    ----------
    series : np.ndarray
        Univariate time series.
    order : tuple
        (p, d, q) ARIMA order.

    Returns
    -------
    dict with keys: model_result, aic, bic, ljungbox_p, white_noise
    """
    res = ARIMA(series, order=order).fit()
    lb  = acorr_ljungbox(res.resid, lags=[10], return_df=True)
    lb_p = float(lb['lb_pvalue'].iloc[0])
    return {
        'model_result': res,
        'order':        order,
        'aic':          res.aic,
        'bic':          res.bic,
        'ljungbox_p':   lb_p,
        'white_noise':  lb_p > 0.05,   # fail to reject → residuals are white noise
    }


# ── AR(2) baseline ────────────────────────────────────────────────────────────
def ar2_forecast(series: np.ndarray, n_ahead: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit AR(2) model and generate an n-step ahead forecast.

    Parameters
    ----------
    series : np.ndarray
        Training RR series.
    n_ahead : int
        Number of steps to forecast.

    Returns
    -------
    (phi, forecast) : (np.ndarray shape (2,), np.ndarray shape (n_ahead,))
        AR coefficients and forecast values (as RR level, not differences).
    """
    res  = ARIMA(series, order=(2, 0, 0)).fit()
    phi  = np.array(res.arparams)  # [phi_1, phi_2]

    hist = list(series[-2:])
    fc   = []
    for _ in range(n_ahead):
        nxt = phi[0] * hist[-1] + phi[1] * hist[-2]
        fc.append(nxt)
        hist.append(nxt)

    # Convert increments to level (cumsum from last known value)
    fc_level = series[-1] + np.cumsum(fc)
    return phi, fc_level
