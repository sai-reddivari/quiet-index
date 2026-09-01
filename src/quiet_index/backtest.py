"""VaR backtesting pipeline (E1 Tasks 3-4, upgraded).

Conventions (E1): 99% confidence, 10-day horizon, log returns,
forward return r10 = ln(S_{t+10}/S_t), breach when r10 < VaR_t (both negative),
VaR_t = z_alpha * sigma_t * sqrt(10).

Upgrades over E1: calibrated-lambda EWMA, GARCH(1,1), Kupiec POF,
Christoffersen independence/conditional coverage, capital-cost metric,
non-overlapping robustness variant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REGIMES = {
    "covid": ("2020-02-01", "2020-06-30"),
    "hiking_2022": ("2022-01-01", "2022-12-31"),
    "liberation_day": ("2025-02-10", "2025-06-30"),
    "the_calm": ("2026-01-01", None),
}


def vol_rolling(returns: pd.Series, window: int = 21) -> pd.Series:
    """Trailing-window daily SD (E1 Task 3)."""
    raise NotImplementedError


def vol_ewma(returns: pd.Series, lam: float = 0.72) -> pd.Series:
    """EWMA daily vol, initialized at full-sample variance (E1 Task 4).
    sigma2_{t+1|t} = lam * sigma2_{t|t-1} + (1-lam) * r_t^2
    """
    raise NotImplementedError


def calibrate_lambda(returns: pd.Series, grid: np.ndarray | None = None) -> float:
    """Choose lambda minimizing QLIKE (or Gaussian NLL) of 1-step variance
    forecasts vs realized r_t^2. Answers the E1 self-critique: lambda is a
    tuning lever, not a constant of nature. Report the loss curve, not just argmin.
    """
    raise NotImplementedError


def vol_garch(returns: pd.Series, dist: str = "normal") -> pd.Series:
    """GARCH(1,1) conditional vol via `arch` (stretch: dist='t')."""
    raise NotImplementedError


def var_series(vol_daily: pd.Series, c: float = 0.99, horizon: int = 10) -> pd.Series:
    """VaR_t = Phi^-1(1-c) * vol_t * sqrt(horizon)."""
    raise NotImplementedError


def breaches(returns: pd.Series, var_t: pd.Series, horizon: int = 10, overlapping: bool = True) -> pd.DataFrame:
    """Forward h-day return vs VaR_t; returns [fwd_ret, var, breach] frame.
    overlapping=False keeps every h-th observation (robustness variant).
    Vectorized result must match a naive loop (tested).
    """
    raise NotImplementedError


@dataclass
class BacktestVerdict:
    n: int
    n_breach: int
    breach_pct: float
    zone: str            # green / yellow / red (binomial percentiles, E1 method)
    kupiec_p: float      # POF likelihood ratio test
    christoffersen_p: float  # independence test
    cc_p: float          # conditional coverage (joint)
    mean_abs_var: float  # capital-cost proxy: average reserved VaR over window


def evaluate(bt: pd.DataFrame, c: float = 0.99) -> BacktestVerdict:
    """Grade one model on one window. Traffic-light cutoffs from
    Binomial(n, 1-c) percentiles (0.95 / 0.9999) — note in write-up that the
    official Basel light is defined on 250 one-day comparisons.
    """
    raise NotImplementedError


def run_matrix(returns: pd.Series, models: dict[str, pd.Series]) -> pd.DataFrame:
    """models: name -> daily vol series. Evaluate every model on every REGIMES
    window; tidy DataFrame for the scoreboard figure (breaches vs mean |VaR|).
    """
    raise NotImplementedError
