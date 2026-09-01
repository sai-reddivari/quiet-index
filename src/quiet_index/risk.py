"""Covariance estimators and VaR/ES risk decomposition.

Formulas are the CQF E1 set:
    sigma_p = sqrt(w' Sigma w)
    MCR_i   = (Sigma w)_i / sigma_p                    (marginal contribution)
    CR_i    = w_i * MCR_i;  sum_i CR_i == sigma_p      (Euler — unit-tested)
    dVaR/dw_i = mu_i + z_a * MCR_i
    dES/dw_i  = mu_i - phi(z_a)/(1-c) * MCR_i
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cov_full(returns: pd.DataFrame) -> pd.DataFrame:
    """Full-sample covariance (annualize where reported; keep daily internally)."""
    raise NotImplementedError


def cov_rolling(returns: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Trailing-window covariance as of the last date."""
    raise NotImplementedError


def cov_ewma(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """RiskMetrics EWMA covariance as of the last date:
        S_t = lam * S_{t-1} + (1 - lam) * r_{t-1} r_{t-1}'
    initialized at the sample covariance. Must match a naive loop (tested).
    """
    raise NotImplementedError


def portfolio_vol(w: np.ndarray, sigma: np.ndarray) -> float:
    raise NotImplementedError


def marginal_contrib(w: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """(Sigma w) / sigma_p."""
    raise NotImplementedError


def component_risk(w: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """w_i * MCR_i. Sums to sigma_p (Euler)."""
    raise NotImplementedError


def var_es_sensitivities(
    w: np.ndarray, sigma: np.ndarray, mu: np.ndarray | None = None, c: float = 0.99
) -> pd.DataFrame:
    """Per-asset dVaR/dw_i and dES/dw_i at confidence c (E1 Task 2 formulas).

    mu=None means mu=0 (risk-only view). Returns tidy DataFrame:
    [weight, mcr, component_share, dvar, des].
    """
    raise NotImplementedError


def parametric_var_es(sigma_p: float, c: float = 0.99, horizon_days: int = 10, mu: float = 0.0) -> tuple[float, float]:
    """Normal VaR and ES scaled by sqrt(horizon) (additivity of variance)."""
    raise NotImplementedError
