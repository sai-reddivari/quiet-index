"""Mean-variance machinery: closed-form (E1 Task 1) + constrained numerical.

Closed-form frontier scalars:
    A = 1' S^-1 1,  B = mu' S^-1 1,  C = mu' S^-1 mu,  D = A*C - B^2
    lambda = (A*m - B)/D,  gamma = (C - B*m)/D
    w*(m)  = S^-1 (lambda*mu + gamma*1)
    frontier variance: sigma^2(m) = (A*m^2 - 2*B*m + C) / D
Global min-var: w = S^-1 1 / A.  Tangency: w ∝ S^-1 (mu - r_f*1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frontier:
    A: float
    B: float
    C: float
    D: float
    mu: np.ndarray
    sigma: np.ndarray  # covariance matrix

    def weights(self, m: float) -> np.ndarray:
        """Closed-form min-variance weights for target return m."""
        raise NotImplementedError

    def frontier_sigma(self, m: float) -> float:
        """sqrt((A m^2 - 2 B m + C) / D). Must match portfolio_vol(weights(m)) — tested."""
        raise NotImplementedError


def make_frontier(mu: np.ndarray, sigma: np.ndarray) -> Frontier:
    raise NotImplementedError


def global_min_var(sigma: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def tangency(mu: np.ndarray, sigma: np.ndarray, rf: float) -> np.ndarray:
    raise NotImplementedError


def long_only_target_vol(mu: np.ndarray, sigma: np.ndarray, target_vol: float) -> np.ndarray:
    """max mu'w  s.t. w>=0, sum w = 1, sqrt(w'Sw) <= target_vol (scipy SLSQP).

    This is the 'what does the optimizer want at SPY's risk budget' portfolio (H2).
    """
    raise NotImplementedError


def long_only_min_var(sigma: np.ndarray) -> np.ndarray:
    """E1 Task 2 extension, generalized to n assets."""
    raise NotImplementedError


def shrink_mu(sample_mu: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Shrink sample means toward the grand mean; document strength choice.
    One of three mu scenarios (naive / shrunk / mu-agnostic) — never present one alone.
    """
    raise NotImplementedError
