"""Mean-variance context: closed-form frontier, numerical check, long-only and tangency portfolios.

The closed form allows short positions. The long-only portfolios have no
closed form, so those are solved numerically with the same inputs.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config as cfg


def frontier_constants(mu, Sigma):   #the four scalars A, B, C, D that describe the whole frontier
    ones = np.ones(len(mu))
    Sigma_inv = np.linalg.inv(Sigma)
    A = ones.T @ Sigma_inv @ ones
    B = mu.T @ Sigma_inv @ ones
    C = mu.T @ Sigma_inv @ mu
    D = A * C - B**2
    return Sigma_inv, ones, A, B, C, D


def minimum_variance_weights(mu, Sigma, target_return):   #lowest-variance portfolio that earns the target return, shorts allowed
    Sigma_inv, ones, A, B, C, D = frontier_constants(mu, Sigma)
    lambda_multiplier = (A * target_return - B) / D   #multiplier on the target return constraint
    gamma_multiplier = (C - B * target_return) / D    #multiplier on the budget constraint
    return Sigma_inv @ (lambda_multiplier * mu + gamma_multiplier * ones)


def global_minimum_variance_weights(Sigma):   #the left-most point of the frontier, needs no expected returns
    ones = np.ones(len(Sigma))
    Sigma_inv = np.linalg.inv(Sigma)
    return (Sigma_inv @ ones) / (ones.T @ Sigma_inv @ ones)


def tangency_weights(mu, Sigma, risk_free_rate):   #highest Sharpe ratio portfolio, shorts allowed
    Sigma_inv, ones, A, B, C, D = frontier_constants(mu, Sigma)
    return (Sigma_inv @ (mu - risk_free_rate * ones)) / (B - A * risk_free_rate)


def frontier_volatility(mu, Sigma, target_returns):   #sigma at each target return: sqrt((A m^2 - 2 B m + C) / D)
    Sigma_inv, ones, A, B, C, D = frontier_constants(mu, Sigma)
    return np.sqrt((A * target_returns**2 - 2 * B * target_returns + C) / D)


def portfolio_statistics(weights, mu, Sigma, risk_free_rate):   #return, risk and Sharpe ratio of one portfolio
    portfolio_return = weights @ mu
    portfolio_risk = np.sqrt(weights @ Sigma @ weights)
    return {"expected_return": portfolio_return, "volatility": portfolio_risk,
            "sharpe_ratio": (portfolio_return - risk_free_rate) / portfolio_risk}


def numerical_minimum_variance(mu, Sigma, target_return=None, long_only=False):   #the same problem handed to an optimizer
    n = len(mu)
    constraints = [{"type": "eq", "fun": lambda weights: weights.sum() - 1}]   #budget constraint
    if target_return is not None:
        constraints.append({"type": "eq", "fun": lambda weights: weights @ mu - target_return})   #target return constraint
    bounds = [(0, 1)] * n if long_only else None
    solution = minimize(lambda weights: weights @ Sigma @ weights, x0=np.ones(n) / n, method="SLSQP",
                        bounds=bounds, constraints=constraints, options={"ftol": 1e-14, "maxiter": 1000})
    assert solution.success, solution.message   #stop here rather than report weights from a failed optimization
    return solution.x


def long_only_at_volatility_target(mu, Sigma, vol_target):   #highest expected return with no shorts and volatility capped at the target
    n = len(mu)
    constraints = [{"type": "eq", "fun": lambda weights: weights.sum() - 1},
                   {"type": "ineq", "fun": lambda weights: vol_target**2 - weights @ Sigma @ weights}]   #variance at or below the target
    starting_weights = numerical_minimum_variance(mu, Sigma, long_only=True)   #start from the lowest-risk long-only portfolio
    if np.sqrt(starting_weights @ Sigma @ starting_weights) > vol_target:
        return None   #the target is below anything a long-only portfolio can reach
    solution = minimize(lambda weights: -(weights @ mu), x0=starting_weights, method="SLSQP",
                        bounds=[(0, 1)] * n, constraints=constraints, options={"ftol": 1e-14, "maxiter": 1000})
    assert solution.success, solution.message
    return solution.x


def long_only_tangency(mu, Sigma, risk_free_rate):   #highest Sharpe ratio with no shorts
    n = len(mu)
    constraints = [{"type": "eq", "fun": lambda weights: weights.sum() - 1}]
    negative_sharpe = lambda weights: -(weights @ mu - risk_free_rate) / np.sqrt(weights @ Sigma @ weights)
    solution = minimize(negative_sharpe, x0=np.ones(n) / n, method="SLSQP",
                        bounds=[(0, 1)] * n, constraints=constraints, options={"ftol": 1e-14, "maxiter": 1000})
    assert solution.success, solution.message
    return solution.x


def long_only_frontier(mu, Sigma, points=40):   #lowest volatility at each target return when shorts are not allowed
    lowest_risk_weights = numerical_minimum_variance(mu, Sigma, long_only=True)
    target_returns = np.linspace(lowest_risk_weights @ mu, mu.max(), points)   #from the minimum-variance return up to the best single asset
    frontier_rows = []
    for target_return in target_returns[:-1]:   #the last point is 100% in one asset and needs no optimizer
        weights = numerical_minimum_variance(mu, Sigma, target_return, long_only=True)
        frontier_rows.append({"expected_return": target_return, "volatility": np.sqrt(weights @ Sigma @ weights)})
    frontier_rows.append({"expected_return": mu.max(), "volatility": np.sqrt(Sigma[mu.argmax(), mu.argmax()])})
    return pd.DataFrame(frontier_rows)


def frontier_inputs(basket_daily_returns, benchmark_returns):   #expected returns and covariance for the eight assets
    asset_log_returns = basket_daily_returns.join(benchmark_returns[cfg.DIVERSIFIERS], how="inner").dropna()
    asset_simple_returns = np.expm1(asset_log_returns)   #a portfolio's return is linear in simple returns, so the optimizer gets those
    sample_means = asset_simple_returns.mean() * cfg.TRADING_DAYS
    var_covar_matrix = asset_simple_returns.cov() * cfg.TRADING_DAYS
    return sample_means, var_covar_matrix


def expected_return_scenarios(sample_means):   #three views on expected returns, from most to least trusting of history
    grand_mean = sample_means.mean()
    return {
        "Sample means": sample_means,
        "50% shrinkage": (1 - cfg.SHRINKAGE) * sample_means + cfg.SHRINKAGE * grand_mean,   #halfway between history and "all assets earn the same"
        "No views": pd.Series(grand_mean, index=sample_means.index),                        #every asset earns the same, so only risk matters
    }


def closed_form_versus_optimizer(mu, Sigma, target_returns):   #largest weight gap between the formula and the optimizer
    check_rows = {}
    for target_return in target_returns:
        closed_form = minimum_variance_weights(mu, Sigma, target_return)
        numerical = numerical_minimum_variance(mu, Sigma, target_return)
        check_rows[target_return] = {"closed_form_volatility": np.sqrt(closed_form @ Sigma @ closed_form),
                                     "optimizer_volatility": np.sqrt(numerical @ Sigma @ numerical),
                                     "max_weight_gap": np.abs(closed_form - numerical).max()}
    return pd.DataFrame.from_dict(check_rows, orient="index").rename_axis("target_return")


def scenario_portfolios(sample_means, var_covar_matrix, risk_free_rate):   #every portfolio in Section 3, one column each
    assets = sample_means.index
    Sigma = var_covar_matrix.to_numpy()
    weights_table, statistics_table = {}, {}

    for scenario_name, scenario_means in expected_return_scenarios(sample_means).items():
        mu = scenario_means.to_numpy()
        portfolios = {}
        if scenario_name == "No views":   #with equal expected returns every efficient portfolio collapses to minimum variance
            portfolios["Minimum variance"] = global_minimum_variance_weights(Sigma)
            portfolios["Minimum variance, long-only"] = numerical_minimum_variance(mu, Sigma, long_only=True)
        else:
            portfolios["Tangency"] = tangency_weights(mu, Sigma, risk_free_rate)
            portfolios["Tangency, long-only"] = long_only_tangency(mu, Sigma, risk_free_rate)
            for vol_target in cfg.VOL_TARGETS:
                portfolios[f"Long-only, {vol_target:.0%} vol"] = long_only_at_volatility_target(mu, Sigma, vol_target)

        for portfolio_name, weights in portfolios.items():
            if weights is None:
                continue
            weights_table[(scenario_name, portfolio_name)] = pd.Series(weights, index=assets)
            statistics = portfolio_statistics(weights, mu, Sigma, risk_free_rate)
            if scenario_name == "No views":   #with no return view, only the volatility of the portfolio means anything
                statistics["expected_return"], statistics["sharpe_ratio"] = np.nan, np.nan
            statistics_table[(scenario_name, portfolio_name)] = statistics

    return pd.DataFrame(weights_table), pd.DataFrame(statistics_table)
