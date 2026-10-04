"""Closed-form cases with known answers. Every implementation is checked against these before it touches market data."""

import numpy as np
import pandas as pd

from quiet_index import backtest, frontier, risk


def test_four_asset_minimum_variance():
    mu = np.array([0.02, 0.07, 0.15, 0.20]) #returns vector
    sd = np.array([0.05, 0.12, 0.17, 0.25]) #volatility vector
    R = np.array([[1, 0.3, 0.3, 0.3],
                  [0.3, 1, 0.6, 0.6],
                  [0.3, 0.6, 1, 0.6],
                  [0.3, 0.6, 0.6, 1]]) #0.3 against the first asset, 0.6 elsewhere
    Diag = np.diag(sd)
    Sigma = Diag @ R @ Diag

    weights = frontier.minimum_variance_weights(mu, Sigma, target_return=0.045)
    portfolio_risk = np.sqrt(weights @ Sigma @ weights)

    assert np.isclose(weights.sum(), 1) #budget constraint
    assert np.isclose(weights @ mu, 0.045) #target return constraint
    assert np.allclose(weights, [0.7851, 0.0539, 0.1336, 0.0275], atol=5e-5)
    assert np.isclose(portfolio_risk, 0.0584, atol=5e-5)


def test_closed_form_matches_optimizer():
    mu = np.array([0.02, 0.07, 0.15, 0.20])
    sd = np.array([0.05, 0.12, 0.17, 0.25])
    R = np.full((4, 4), 0.6); R[0, :] = R[:, 0] = 0.3; np.fill_diagonal(R, 1)
    Sigma = np.diag(sd) @ R @ np.diag(sd)

    for target_return in [0.03, 0.045, 0.10, 0.18]:
        closed_form = frontier.minimum_variance_weights(mu, Sigma, target_return)
        numerical = frontier.numerical_minimum_variance(mu, Sigma, target_return)
        assert np.allclose(closed_form, numerical, atol=1e-5)


def test_three_asset_var_es_sensitivities():
    vol = np.array([0.30, 0.20, 0.15]) #volatility vector
    allocations = pd.Series([0.5, 0.2, 0.3]) #weights vector
    correlations = np.array([[1, 0.8, 0.5],
                             [0.8, 1, 0.3],
                             [0.5, 0.3, 1]])
    var_covar = pd.DataFrame(np.diag(vol) @ correlations @ np.diag(vol))

    sensitivities = risk.var_es_sensitivities(allocations, var_covar, confidence=0.99)

    assert np.allclose(sensitivities["var_sensitivity"], [-0.684, -0.387, -0.221], atol=5e-4)
    assert np.allclose(sensitivities["es_sensitivity"].abs(), [0.783, 0.443, 0.253], atol=5e-4)


def test_tail_coefficients_at_99():
    z_alpha, es_coefficient = risk.tail_coefficients(0.99)
    assert np.isclose(z_alpha, -2.326, atol=5e-4)
    assert np.isclose(es_coefficient, 2.665, atol=5e-4)


def test_euler_components_sum_to_portfolio_risk():
    vol = np.array([0.30, 0.20, 0.15])
    correlations = np.array([[1, 0.8, 0.5], [0.8, 1, 0.3], [0.5, 0.3, 1]])
    var_covar = pd.DataFrame(np.diag(vol) @ correlations @ np.diag(vol))
    allocations = pd.Series([0.5, 0.2, 0.3])

    decomposition = risk.risk_decomposition(allocations, var_covar)

    assert np.isclose(decomposition["component_risk"].sum(), risk.portfolio_volatility(allocations, var_covar))
    assert np.isclose(decomposition["risk_share"].sum(), 1)


def test_zone_cutoffs_for_260_observations():
    green_cutoff, yellow_cutoff = backtest.zone_cutoffs(260) #Binomial(260, 0.01) at the 95th and 99.99th percentiles
    assert (green_cutoff, yellow_cutoff) == (5, 10)
    assert backtest.zone(5, 260) == "Green"
    assert backtest.zone(10, 260) == "Yellow"
    assert backtest.zone(11, 260) == "Red"
