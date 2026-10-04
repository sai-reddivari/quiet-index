"""Covariance estimators: full sample, trailing one year, and EWMA.

All three return an annualized covariance matrix with basket labels kept on
both axes, so the risk decomposition can be run on any of them unchanged.
"""

import numpy as np
import pandas as pd

from . import config as cfg


def covariance_from_returns(daily_returns):   #annualized covariance built as D x R x D
    risk_vector = daily_returns.std() * np.sqrt(cfg.TRADING_DAYS) #annualized vol of each basket over this sample
    corr_matrix = daily_returns.corr()
    Diag = np.diag(risk_vector) #diagonal matrix with the annualized vols on the diagonal
    return pd.DataFrame(Diag @ corr_matrix.values @ Diag,
                        index=corr_matrix.index, columns=corr_matrix.columns)


def full_sample_covariance(daily_returns):   #every day in the history carries the same weight
    return covariance_from_returns(daily_returns)


def trailing_covariance(daily_returns, window=cfg.ESTIMATION_WINDOW):   #only the most recent year, equally weighted
    return covariance_from_returns(daily_returns.tail(window))


def ewma_covariance_path(daily_returns, lambda_=cfg.EWMA_LAMBDA, seed_window=cfg.ESTIMATION_WINDOW):   #RiskMetrics recursion, one matrix per day
    returns_array = daily_returns.to_numpy()
    ewma_covar = daily_returns.iloc[:seed_window].cov().to_numpy() #seed the recursion with the first year's covariance

    covariance_by_date = {}
    for day in range(seed_window, len(returns_array)):
        return_today = returns_array[day].reshape(-1, 1) #column vector of today's basket returns
        ewma_covar = lambda_ * ewma_covar + (1 - lambda_) * (return_today @ return_today.T) #yesterday's estimate decays, today's squared returns come in
        covariance_by_date[daily_returns.index[day]] = ewma_covar * cfg.TRADING_DAYS #annualized
    return covariance_by_date


def ewma_covariance(daily_returns, lambda_=cfg.EWMA_LAMBDA):   #the EWMA estimate as of the last day in the history
    covariance_by_date = ewma_covariance_path(daily_returns, lambda_)
    latest_date = max(covariance_by_date)
    return pd.DataFrame(covariance_by_date[latest_date],
                        index=daily_returns.columns, columns=daily_returns.columns)


def all_estimators(daily_returns):   #the three estimators side by side
    return {
        "Full sample": full_sample_covariance(daily_returns),
        "Trailing 252d": trailing_covariance(daily_returns),
        "EWMA 0.94": ewma_covariance(daily_returns),
    }


def half_life(lambda_):   #days for an observation's weight to fall by half under EWMA
    return np.log(0.5) / np.log(lambda_)
