"""Sanity tests — the E1 'verify independently' ethos, productized.

Run: pytest
Every test uses small synthetic inputs with hand-checkable answers;
none touch the network.
"""
import numpy as np
import pytest

# TODO(sai): un-skip as each src function lands.


@pytest.mark.skip(reason="pending implementation")
def test_euler_decomposition_sums_to_portfolio_vol():
    """sum_i w_i * MCR_i == sigma_p, for a random PSD Sigma and random long-only w."""


@pytest.mark.skip(reason="pending implementation")
def test_closed_form_frontier_matches_numerical_optimizer():
    """Frontier.weights(m) vol == frontier_sigma(m) == scipy equality-constrained solve.
    Use the E1 exam universe (mu, sigma, R from Task 1) — expected:
    A=423.61, B=6.807, C=0.9065, D=337.68, w=(78.5, 5.4, 13.4, 2.7)%, sigma_p=5.84%."""


@pytest.mark.skip(reason="pending implementation")
def test_ewma_recursion_matches_naive_loop():
    """cov_ewma / vol_ewma vectorized == explicit python loop, to 1e-12."""


@pytest.mark.skip(reason="pending implementation")
def test_breach_count_matches_naive_loop():
    """Vectorized breaches() == day-by-day loop on a toy series with known breaches."""


@pytest.mark.skip(reason="pending implementation")
def test_var_es_sensitivities_match_e1_task2():
    """E1 Task 2 universe: sigma=(30,20,15)%, w=(50,20,30)%, rho=(.8,.5,.3), c=99%.
    Expected dVaR = (-0.684, -0.387, -0.221); |dES| = (0.783, 0.443, 0.253).
    (Note: 0.253, not 0.443 — fixes the index bug in the original exam notebook.)"""


@pytest.mark.skip(reason="pending implementation")
def test_baskets_partition_universe():
    """form_baskets: every ex-M7 name lands in exactly one 2x2 basket per
    rebalance date — no overlaps, no orphans; M7 names never appear in the
    factor baskets. Use a toy 12-name universe with hand-set factors."""


@pytest.mark.skip(reason="pending implementation")
def test_basket_no_lookahead():
    """Factor values as of quarter-end t drive assignments for t+1..t+63 only.
    Shift the toy factor series by one day and assert assignments change at
    the NEXT rebalance, not retroactively."""


@pytest.mark.skip(reason="pending implementation")
def test_basket_returns_aggregate():
    """Cap-weighted basket returns on a toy panel match a hand computation;
    weights renormalize within basket at rebalance."""


@pytest.mark.skip(reason="pending implementation")
def test_traffic_light_cutoffs():
    """Binomial(260, 0.01): green<=5, yellow<=10 (matches E1); and Binomial(250, 0.01)
    reproduces the exam's 'yellow 5-9' example."""
