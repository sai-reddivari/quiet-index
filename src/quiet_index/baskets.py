"""Bespoke basket construction: five non-overlapping US-equity baskets.

Design (PROJECT_SPEC v1.1, section 3A):
    M7 carved out first, then the ex-M7 S&P 100 double-sorted 2x2 on
    AI-beta x 12-1 momentum (median splits), cap-weighted within basket,
    reformed quarterly with factors measured as of each quarter-end and
    applied FORWARD (no lookahead).

Factors are price-only:
    AI-beta:  two-stage regression per stock over a trailing 252d window —
              (1) strip the market: e_i = resid of r_i ~ r_SPY
              (2) load on theme:    AI-beta_i = slope of e_i ~ e_proxy,
                  where e_proxy = resid of r_SMH ~ r_SPY  (NVDA variant for robustness)
    Momentum: 12-1 (trailing 252d cumulative return excluding the last 21d)

Distinctness checks (Scheme C as validator, not slicer):
    pairwise basket correlations; hierarchical + k-means clusters on the
    name-level correlation matrix vs the rule labels (confusion matrix,
    adjusted Rand index). Failure to recover is a reported finding, not a bug.

Time-box: if this module costs > ~3h, fall back to the v1.0 ETF sleeves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASKETS = ["M7", "AI_HI_MOM_HI", "AI_HI_MOM_LO", "AI_LO_MOM_HI", "AI_LO_MOM_LO"]
BASKET_LABELS = {
    "M7": "The Anchor",
    "AI_HI_MOM_HI": "The Rush",
    "AI_HI_MOM_LO": "Left Behind by the Theme",
    "AI_LO_MOM_HI": "The Quiet Winners",
    "AI_LO_MOM_LO": "The Residual Economy",
}


def load_sp100_members(path: str = "data/reference/sp100_members.csv") -> list[str]:
    """Current S&P 100 tickers (hand-collected snapshot; date + source in the CSV).
    Survivorship caveat lives in README Limitations."""
    raise NotImplementedError


def factor_ai_beta(
    returns: pd.DataFrame, market: pd.Series, proxy: pd.Series, window: int = 252
) -> pd.DataFrame:
    """Rolling AI-beta per name (two-stage regression above).

    Returns a date x ticker frame of betas (NaN until `window` obs).
    Implementation hint: both stages are OLS slopes — vectorize via
    rolling cov/var of residuals rather than statsmodels-per-day.
    """
    raise NotImplementedError


def factor_momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-1 momentum: P_{t-skip} / P_{t-lookback} - 1, per name."""
    raise NotImplementedError


def form_baskets(
    ai_beta: pd.DataFrame, momentum: pd.DataFrame, mag7: list[str], rebalance: str = "Q"
) -> pd.DataFrame:
    """Assign every name to exactly one basket per rebalance date.

    Median splits computed within the ex-M7 universe at each quarter-end,
    applied to the FOLLOWING quarter. Returns a long frame
    [date, ticker, basket]. Invariants (unit-tested): the five baskets
    partition the universe — no overlaps, no orphans.
    """
    raise NotImplementedError


def basket_returns(
    returns: pd.DataFrame, assignments: pd.DataFrame, market_caps: pd.DataFrame
) -> pd.DataFrame:
    """Cap-weighted daily log-return series per basket (columns = BASKETS).
    Weights renormalized within basket at each rebalance."""
    raise NotImplementedError


def basket_weights_in_slice(assignments: pd.DataFrame, market_caps: pd.DataFrame) -> pd.DataFrame:
    """Each basket's cap-weight share of the S&P 100 slice through time —
    the 'capital weight' side of the headline weight-vs-risk figure."""
    raise NotImplementedError


def distinctness_report(basket_rets: pd.DataFrame) -> pd.DataFrame:
    """Pairwise basket correlation matrix + flags for any pair > 0.95
    (threshold from spec; a breach is a finding, not a bug)."""
    raise NotImplementedError


def cluster_check(
    name_returns: pd.DataFrame, assignments_now: pd.Series, k: int = 5
) -> dict:
    """Scheme C validation: hierarchical (correlation distance) and k-means
    clusters on the current-window name-level returns vs rule-based labels.

    Returns {"confusion": DataFrame, "adjusted_rand_hier": float,
    "adjusted_rand_kmeans": float, "linkage": ndarray (for the dendrogram fig)}.
    """
    raise NotImplementedError
