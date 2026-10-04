"""Checks on the estimators themselves, run on simulated data where the right answer is known."""

import numpy as np
import pandas as pd

from quiet_index import backtest, baskets, covariance, factors


def simulated_returns(seed=0, days=400):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=days)
    spy = pd.Series(rng.normal(0, 0.01, days), index=dates)
    proxy = 1.5 * spy + pd.Series(rng.normal(0, 0.012, days), index=dates) #an AI proxy with a market beta of 1.5
    stock = 0.9 * spy + 0.4 * (proxy - 1.5 * spy) + pd.Series(rng.normal(0, 0.008, days), index=dates) #true beta 0.9, true lambda 0.4
    return spy, proxy, stock


def test_ols_residuals_are_uncorrelated_with_the_regressor():
    spy, proxy, stock = simulated_returns()
    fit = factors.ols_regression(stock, spy)
    assert abs(fit["resid"].cov(spy)) < 1e-15 #zero to machine precision
    assert abs(fit["resid"].mean()) < 1e-15


def test_ols_recovers_a_known_beta():
    spy, proxy, stock = simulated_returns(days=5000)
    fit = factors.ols_regression(stock, spy)
    assert np.isclose(fit["beta"], 0.9, atol=0.03)


def test_ols_needs_minimum_observations():
    spy, proxy, stock = simulated_returns(days=150)
    assert factors.ols_regression(stock, spy) is None #fewer than 200 joint observations


def test_two_stage_lambda_equals_joint_regression():
    spy, proxy, stock = simulated_returns()
    resid_proxy = factors.ols_regression(proxy, spy)["resid"]
    resid_stock = factors.ols_regression(stock, spy)["resid"]
    two_stage_lambda = factors.ols_regression(resid_stock, resid_proxy)["beta"]
    joint_lambda = factors.joint_regression_ai_beta(stock, spy, proxy)
    assert np.isclose(two_stage_lambda, joint_lambda, atol=1e-12) #Frisch-Waugh-Lovell


def test_formation_dates_skip_an_unfinished_quarter():
    dates = pd.bdate_range("2018-01-01", "2019-11-15") #the data stops in the middle of the fourth quarter
    members_returns = pd.DataFrame({"A": 0.0}, index=dates)
    formation_dates = factors.formation_quarter_ends(members_returns)
    assert formation_dates[-1] == pd.Timestamp("2019-09-30")
    assert all(members_returns.index.get_loc(date) >= 252 for date in formation_dates) #every formation has a year of prices behind it
    assert pd.Timestamp("2018-03-30") not in formation_dates #too early, not enough history


def test_baskets_use_the_median_split_and_carve_out_mag7():
    tickers = ["AAPL", "NVDA"] + [f"S{i}" for i in range(8)]
    loadings = pd.DataFrame({"ai_beta": [9, 9, 1, 2, 3, 4, 5, 6, 7, 8],
                             "momentum": [9, 9, 8, 7, 6, 5, 4, 3, 2, 1]}, index=tickers)
    assignment = baskets.assign_baskets(loadings)
    assert set(assignment[assignment == "M7"].index) == {"AAPL", "NVDA"}
    assert assignment["S7"] == "AI_HI_MOM_LO" #highest lambda, lowest momentum outside the Mag-7
    assert assignment["S0"] == "AI_LO_MOM_HI"
    assert assignment.drop(["AAPL", "NVDA"]).value_counts().sum() == 8


def test_basket_return_reweights_when_a_member_is_missing():
    dates = pd.bdate_range("2024-01-01", periods=2)
    member_returns = pd.DataFrame({"A": [np.log(1.10), np.log(1.10)], "B": [np.log(0.90), np.nan]}, index=dates)
    member_weights = pd.Series({"A": 0.75, "B": 0.25})
    basket_return = baskets.weighted_basket_return(member_returns, member_weights)
    assert np.isclose(np.expm1(basket_return.iloc[0]), 0.75 * 0.10 + 0.25 * -0.10) #weighted average of simple returns
    assert np.isclose(np.expm1(basket_return.iloc[1]), 0.10) #only A has a return, so A carries all the weight


def test_ewma_covariance_matches_explicit_weights():
    rng = np.random.default_rng(1)
    daily_returns = pd.DataFrame(rng.normal(0, 0.01, (300, 2)), columns=["X", "Y"])
    lambda_, seed_window = 0.94, 252
    recursive = covariance.ewma_covariance(daily_returns, lambda_).to_numpy() / 252

    later_returns = daily_returns.iloc[seed_window:].to_numpy()
    ages = np.arange(len(later_returns))[::-1] #0 for the most recent day
    explicit = (lambda_ ** len(later_returns)) * daily_returns.iloc[:seed_window].cov().to_numpy()
    for age, row in zip(ages, later_returns):
        explicit = explicit + (1 - lambda_) * lambda_ ** age * np.outer(row, row)
    assert np.allclose(recursive, explicit)
    assert np.all(np.linalg.eigvalsh(recursive) > 0) #a covariance matrix has to be positive definite


def test_ewma_half_life():
    assert np.isclose(covariance.half_life(0.94), 11.2, atol=0.05)
    assert np.isclose(covariance.half_life(0.72), 2.11, atol=0.01)


def test_kupiec_accepts_the_expected_breach_rate_and_rejects_a_high_one():
    assert backtest.kupiec_test(breaches=3, observations=260)[1] > 0.05 #about 1%
    assert backtest.kupiec_test(breaches=11, observations=260)[1] < 0.01 #about 4%
    assert np.isclose(backtest.kupiec_test(breaches=0, observations=177)[0], -2 * 177 * np.log(0.99)) #no breaches still has a defined statistic


def test_christoffersen_flags_clustered_breaches():
    clustered = [0] * 100 + [1] * 5 + [0] * 100 #five breaches in a row
    spread_out = ([0] * 40 + [1]) * 5 #five breaches, far apart
    assert backtest.christoffersen_test(clustered)[1] < 0.01
    assert backtest.christoffersen_test(spread_out)[1] > 0.05
    assert np.isnan(backtest.christoffersen_test([0] * 50)[1]) #no breaches, nothing to test


def test_var_uses_only_information_up_to_today():
    spy, proxy, stock = simulated_returns(days=600)
    full_history = backtest.ewma_volatility(spy, 0.94)
    truncated = backtest.ewma_volatility(spy.iloc[:400], 0.94)
    assert np.allclose(full_history.iloc[:400], truncated) #later returns do not change earlier estimates
