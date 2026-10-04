"""Factor estimation: market beta, AI loading (lambda) and 12-1 momentum.

Two-factor model (market + AI), estimated in two stages over a trailing
252-day window at every formation date, using only data up to that date.
"""

import pandas as pd

from . import config as cfg


def ols_regression(y: pd.Series, x: pd.Series, min_obs: int = cfg.MIN_OBS):#   OLS of y on x. to be used for fitting  residuals of individual stock returns, AI-proxy returns (SMH), and market returns (SPY)
    df = pd.concat({"y": y, "x": x}, axis=1, sort=True).dropna()
    if len(df) < min_obs:
        return None
    yv, xv = df["y"], df["x"]

    beta  = yv.cov(xv) / xv.var()
    alpha = yv.mean() - beta * xv.mean()
    resid = yv - alpha - beta * xv
    r2    = beta**2 * xv.var() / yv.var()

    return {"alpha": alpha, "beta": beta, "resid": resid, "n": len(df), "r2": r2}


def formation_quarter_ends(members_returns):   #last trading day of each completed calendar quarter
    quarter_ends = members_returns.index.to_series().groupby(members_returns.index.to_period("Q")).max() #create quarter-ends in a series , this is a list of dates in the index that represent quarter ends

    formation_dates = []
    for quarter, qe in quarter_ends.items():
        pos = members_returns.index.get_loc(qe)                 # integer position of this quarter-end
        if pos < cfg.ESTIMATION_WINDOW:                         # need 252 prices behind the formation date for momentum
            continue
        if (quarter.end_time.normalize() - qe).days > 4:        # the data stops mid-quarter, so this is not a real quarter-end
            continue
        formation_dates.append(qe)
    return formation_dates


def momentum_at_quarter_end(tickers_data, quarter_end_position):   #12-1 momentum for every stock at one formation date
    return (tickers_data.iloc[quarter_end_position - cfg.MOMENTUM_SKIP]
            / tickers_data.iloc[quarter_end_position - cfg.ESTIMATION_WINDOW] - 1) #momentum is defined as the stocks 12 month return (t-21 price / t-252 price)


def loadings_at_quarter_end(members_returns, spy_returns, proxy_returns, qe):   #beta and lambda for every stock at one formation date
    pos = members_returns.index.get_loc(qe)   # integer position of this quarter-end

    members_window = members_returns.iloc[pos - cfg.ESTIMATION_WINDOW + 1 : pos + 1]   # trailing 252 daily returns
    spy_window     = spy_returns.reindex(members_window.index)                          # SPY aligned to the window's dates

    # regress the AI proxy's returns on SPY returns and keep the residual: the proxy's return with the
    # market component removed. This series is the AI factor for this window.
    resid_proxy = ols_regression(proxy_returns.reindex(members_window.index), spy_window)["resid"]

    # two regressions per stock: (1) returns on SPY returns -> market beta + residual;
    # (2) that residual on the proxy residual -> AI beta (lambda)
    rows = {}
    for ticker in members_window.columns:
        fit1 = ols_regression(members_window[ticker], spy_window)
        if fit1 is None: continue
        fit2 = ols_regression(fit1["resid"], resid_proxy)
        if fit2 is None: continue
        rows[ticker] = {"beta_mkt": fit1["beta"],   # market beta
                        "ai_beta":  fit2["beta"],   # AI factor loading (lambda)
                        "n_obs":    fit1["n"]}      # observations used in the fit (<= 252)

    return pd.DataFrame.from_dict(rows, orient="index")


def loadings_at_every_quarter_end(members_returns, spy_returns, proxy_returns, tickers_data):   #walk-forward loadings, no foresight
    loadings_by_qe = {}
    for qe in formation_quarter_ends(members_returns):   # loadings at each basket formation date
        loadings = loadings_at_quarter_end(members_returns, spy_returns, proxy_returns, qe)
        pos = members_returns.index.get_loc(qe)
        loadings["momentum"] = momentum_at_quarter_end(tickers_data, pos)   #add momentum column for each stock to the loadings table
        loadings_by_qe[qe] = loadings
    return loadings_by_qe


def joint_regression_ai_beta(stock_returns, spy_returns, proxy_returns):   #AI beta from one regression on SPY and the proxy together
    import numpy as np

    df = pd.concat({"y": stock_returns, "spy": spy_returns, "proxy": proxy_returns}, axis=1, sort=True).dropna()
    regressors = np.column_stack([np.ones(len(df)), df["spy"], df["proxy"]])   #intercept, market, AI proxy
    coefficients = np.linalg.lstsq(regressors, df["y"].to_numpy(), rcond=None)[0]
    return coefficients[2]   #used to confirm the two-stage estimate (Frisch-Waugh-Lovell)
