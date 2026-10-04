"""VaR backtesting on SPY: volatility models, breach counts, and the formal tests.

Convention: the volatility estimate on day t uses returns up to and including
day t. VaR_t is compared with the forward return ln(S_{t+10} / S_t), and a
breach is a forward return below VaR_t.
"""

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binom, chi2, norm

from . import config as cfg


def forward_returns(spy_data, horizon_days=cfg.HORIZON_DAYS):   #realized return over the next 10 trading days
    return np.log(spy_data.shift(-horizon_days) / spy_data)


def rolling_volatility(spy_returns, window=cfg.ROLLING_WINDOW):   #equal-weighted standard deviation of the last 21 daily returns
    return spy_returns.rolling(window).std()


def ewma_variance(spy_returns, lambda_, seed_window=cfg.TRADING_DAYS):   #EWMA variance, updated with each day's squared return
    ret_array = spy_returns.to_numpy()
    ewma_var = np.full(len(ret_array), np.nan)
    init_var = ret_array[:seed_window].var()   #start from the first year's variance
    ewma_var[0] = init_var
    for i in range(1, len(ret_array)):
        ewma_var[i] = lambda_ * ewma_var[i - 1] + (1 - lambda_) * ret_array[i] ** 2   #yesterday's estimate decays, today's squared return comes in
    return pd.Series(ewma_var, index=spy_returns.index)


def ewma_volatility(spy_returns, lambda_):
    return np.sqrt(ewma_variance(spy_returns, lambda_))


def lambda_loss_curve(spy_returns, lambda_grid=cfg.LAMBDA_GRID, calibration_end=cfg.CALIBRATION_END):   #one-step variance forecast loss at each lambda
    calibration_returns = spy_returns.loc[:calibration_end]
    loss_rows = {}
    for lambda_ in lambda_grid:
        variance_forecast = ewma_variance(calibration_returns, lambda_).shift(1)   #today's estimate is the forecast for tomorrow
        squared_return = calibration_returns ** 2                                  #tomorrow's squared return is what the forecast is judged against
        scored = pd.concat({"forecast": variance_forecast, "realized": squared_return}, axis=1).iloc[cfg.TRADING_DAYS:]   #skip the first year while the recursion settles
        loss_rows[lambda_] = {
            "qlike": (np.log(scored["forecast"]) + scored["realized"] / scored["forecast"]).mean(),
            "rmse": np.sqrt(((scored["realized"] - scored["forecast"]) ** 2).mean()),
        }
    return pd.DataFrame.from_dict(loss_rows, orient="index").rename_axis("lambda")


def calibrate_lambda(loss_curve, loss="qlike"):   #the lambda with the lowest loss
    return float(loss_curve[loss].idxmin())


def garch_variance_forecasts(spy_returns, refit_every=cfg.GARCH_REFIT_EVERY, horizon_days=cfg.HORIZON_DAYS,
                             backtest_start=cfg.BACKTEST_START):   #GARCH(1,1) re-estimated on an expanding window, never on future data
    from arch import arch_model

    ret_array = spy_returns.to_numpy() * 100   #arch works better on percent returns
    first_fit = spy_returns.index.searchsorted(backtest_start) - 1   #the first fit uses everything before the backtest starts
    one_day_variance = np.full(len(ret_array), np.nan)
    horizon_variance = np.full(len(ret_array), np.nan)
    parameter_rows = {}

    for refit_day in range(first_fit, len(ret_array), refit_every):
        fitted = arch_model(ret_array[:refit_day + 1], mean="Zero", vol="GARCH", p=1, q=1).fit(disp="off", show_warning=False)
        omega, alpha, beta = fitted.params["omega"], fitted.params["alpha[1]"], fitted.params["beta[1]"]
        persistence = alpha + beta
        parameter_rows[spy_returns.index[refit_day]] = {"omega": omega, "alpha": alpha, "beta": beta, "persistence": persistence}

        conditional_var = np.empty(len(ret_array))   #run the variance recursion over the history with these parameters
        conditional_var[0] = ret_array[:first_fit].var()
        last_day = min(refit_day + refit_every, len(ret_array))
        for i in range(1, last_day):
            conditional_var[i] = omega + alpha * ret_array[i - 1] ** 2 + beta * conditional_var[i - 1]

        long_run_var = omega / (1 - persistence) if persistence < 1 else np.nan
        for i in range(refit_day, last_day):   #these parameters are used until the next refit
            next_day_var = omega + alpha * ret_array[i] ** 2 + beta * conditional_var[i]   #forecast for day i+1 made on day i
            one_day_variance[i] = next_day_var
            if persistence < 1:   #each further day mean-reverts toward the long-run variance at rate alpha + beta
                steps = np.arange(horizon_days)
                horizon_variance[i] = (long_run_var + persistence ** steps * (next_day_var - long_run_var)).sum()
            else:
                horizon_variance[i] = horizon_days * next_day_var

    forecasts = pd.DataFrame({"one_day_volatility": np.sqrt(one_day_variance) / 100,
                              "horizon_volatility": np.sqrt(horizon_variance) / 100}, index=spy_returns.index)
    parameters = pd.DataFrame.from_dict(parameter_rows, orient="index").rename_axis("refit_date")
    return forecasts, parameters


def value_at_risk(daily_volatility, confidence=cfg.CONFIDENCE, horizon_days=cfg.HORIZON_DAYS):   #VaR_t = z * sigma_t * sqrt(10)
    z_factor = norm.ppf(1 - confidence)
    return z_factor * daily_volatility * np.sqrt(horizon_days)


def var_models(spy_returns, calibrated_lambda, garch_forecasts):   #the four models' 10-day 99% VaR on every day
    z_factor = norm.ppf(1 - cfg.CONFIDENCE)
    return pd.DataFrame({
        "Rolling 21d": value_at_risk(rolling_volatility(spy_returns)),
        f"EWMA {cfg.EWMA_LAMBDA_FAST}": value_at_risk(ewma_volatility(spy_returns, cfg.EWMA_LAMBDA_FAST)),
        f"EWMA {calibrated_lambda} (calibrated)": value_at_risk(ewma_volatility(spy_returns, calibrated_lambda)),
        "GARCH(1,1)": z_factor * garch_forecasts["horizon_volatility"],   #GARCH supplies its own 10-day volatility
    })


def zone_cutoffs(observations, confidence=cfg.CONFIDENCE):   #breach counts that separate green, yellow and red
    green_cutoff = binom.ppf(cfg.ZONE_PERCENTILES[0], observations, 1 - confidence)
    yellow_cutoff = binom.ppf(cfg.ZONE_PERCENTILES[1], observations, 1 - confidence)
    return int(green_cutoff), int(yellow_cutoff)


def zone(breaches, observations):   #green up to the first cutoff, yellow up to the second, red beyond
    green_cutoff, yellow_cutoff = zone_cutoffs(observations)
    if breaches <= green_cutoff:
        return "Green"
    if breaches <= yellow_cutoff:
        return "Yellow"
    return "Red"


def kupiec_test(breaches, observations, confidence=cfg.CONFIDENCE):   #is the breach rate consistent with 1%
    expected_rate = 1 - confidence
    observed_rate = breaches / observations
    log_likelihood_expected = xlogy(observations - breaches, 1 - expected_rate) + xlogy(breaches, expected_rate)
    log_likelihood_observed = xlogy(observations - breaches, 1 - observed_rate) + xlogy(breaches, observed_rate)
    likelihood_ratio = -2 * (log_likelihood_expected - log_likelihood_observed)
    return likelihood_ratio, chi2.sf(likelihood_ratio, df=1)


def christoffersen_test(breach_series):   #do breaches arrive independently, or does one breach make the next more likely
    breach_array = np.asarray(breach_series, dtype=int)
    yesterday, today = breach_array[:-1], breach_array[1:]
    n00 = int(((yesterday == 0) & (today == 0)).sum())   #no breach followed by no breach
    n01 = int(((yesterday == 0) & (today == 1)).sum())   #no breach followed by a breach
    n10 = int(((yesterday == 1) & (today == 0)).sum())
    n11 = int(((yesterday == 1) & (today == 1)).sum())   #a breach followed by another breach
    if n01 + n11 == 0 or n00 + n01 == 0 or n10 + n11 == 0:
        return np.nan, np.nan   #with no breaches, or no day after a breach, the test cannot be computed

    rate_after_no_breach = n01 / (n00 + n01)
    rate_after_breach = n11 / (n10 + n11)
    rate_overall = (n01 + n11) / (n00 + n01 + n10 + n11)
    log_likelihood_independent = xlogy(n00 + n10, 1 - rate_overall) + xlogy(n01 + n11, rate_overall)
    log_likelihood_dependent = (xlogy(n00, 1 - rate_after_no_breach) + xlogy(n01, rate_after_no_breach)
                                + xlogy(n10, 1 - rate_after_breach) + xlogy(n11, rate_after_breach))
    likelihood_ratio = -2 * (log_likelihood_independent - log_likelihood_dependent)
    return likelihood_ratio, chi2.sf(likelihood_ratio, df=1)


def backtest_table(var_by_model, ret_10d_forward, start=cfg.BACKTEST_START):   #VaR, forward return and breach flag on every backtest day
    backtest_data = var_by_model.join(ret_10d_forward.rename("Forward 10D Return")).loc[start:].dropna()
    breach_data = backtest_data[var_by_model.columns].gt(backtest_data["Forward 10D Return"], axis=0).astype(int)   #breach when the forward return is below VaR
    return backtest_data, breach_data


def grade_window(backtest_data, breach_data, window_start, window_end):   #one regime window, one row per model
    window_var = backtest_data.loc[window_start:window_end]
    window_breaches = breach_data.loc[window_start:window_end]
    observations = len(window_breaches)
    green_cutoff, yellow_cutoff = zone_cutoffs(observations)

    grade_rows = {}
    for model in breach_data.columns:
        breaches = int(window_breaches[model].sum())
        kupiec_lr, kupiec_p = kupiec_test(breaches, observations)
        christoffersen_lr, christoffersen_p = christoffersen_test(window_breaches[model])
        grade_rows[model] = {
            "observations": observations,
            "breaches": breaches,
            "breach_pct": breaches / observations,
            "green_cutoff": green_cutoff,
            "yellow_cutoff": yellow_cutoff,
            "zone": zone(breaches, observations),
            "kupiec_p": kupiec_p,
            "christoffersen_p": christoffersen_p,
            "mean_var": window_var[model].mean(),   #average VaR reserved over the window, the capital cost of the model
        }
    return pd.DataFrame.from_dict(grade_rows, orient="index").rename_axis("model")


def grade_all_windows(backtest_data, breach_data, regime_windows=cfg.REGIME_WINDOWS):   #every model in every regime window, plus the full period
    windows = dict(regime_windows)
    windows["Full period"] = (backtest_data.index[0], backtest_data.index[-1])
    return pd.concat({window_name: grade_window(backtest_data, breach_data, window_start, window_end)
                      for window_name, (window_start, window_end) in windows.items()}, names=["window"])


def non_overlapping_check(backtest_data, breach_data, horizon_days=cfg.HORIZON_DAYS, regime_windows=cfg.REGIME_WINDOWS):   #the same backtest on 10-day windows that do not overlap
    windows = dict(regime_windows)
    windows["Full period"] = (backtest_data.index[0], backtest_data.index[-1])

    check_rows = []
    for window_name, (window_start, window_end) in windows.items():
        window_breaches = breach_data.loc[window_start:window_end]
        for model in breach_data.columns:
            breaches_by_offset, observations_by_offset, kupiec_p_by_offset = [], [], []
            for offset in range(horizon_days):   #ten ways to pick every tenth day, each a valid non-overlapping sample
                sampled = window_breaches[model].iloc[offset::horizon_days]
                breaches_by_offset.append(int(sampled.sum()))
                observations_by_offset.append(len(sampled))
                kupiec_p_by_offset.append(kupiec_test(int(sampled.sum()), len(sampled))[1])
            check_rows.append({"window": window_name, "model": model,
                               "observations": float(np.mean(observations_by_offset)),
                               "mean_breaches": float(np.mean(breaches_by_offset)),
                               "min_breaches": min(breaches_by_offset),
                               "max_breaches": max(breaches_by_offset),
                               "breach_pct": float(np.sum(breaches_by_offset) / np.sum(observations_by_offset)),
                               "offsets_passing_kupiec": int((np.array(kupiec_p_by_offset) > 0.05).sum())})   #out of ten
    return pd.DataFrame(check_rows).set_index(["window", "model"])


def current_var_by_half_life(spy_returns, calibrated_lambda, garch_forecasts, garch_parameters):   #today's VaR from estimators ordered by how fast they forget
    z_factor = norm.ppf(1 - cfg.CONFIDENCE)
    latest_garch = garch_parameters.iloc[-1]
    estimators = {
        f"EWMA {cfg.EWMA_LAMBDA_FAST}": (np.log(0.5) / np.log(cfg.EWMA_LAMBDA_FAST), ewma_volatility(spy_returns, cfg.EWMA_LAMBDA_FAST).iloc[-1]),
        "Rolling 21d": (cfg.ROLLING_WINDOW / 2, rolling_volatility(spy_returns).iloc[-1]),   #half the weight sits in the most recent half of the window
        f"EWMA {calibrated_lambda} (calibrated)": (np.log(0.5) / np.log(calibrated_lambda), ewma_volatility(spy_returns, calibrated_lambda).iloc[-1]),
        f"EWMA {cfg.EWMA_LAMBDA}": (np.log(0.5) / np.log(cfg.EWMA_LAMBDA), ewma_volatility(spy_returns, cfg.EWMA_LAMBDA).iloc[-1]),
        "GARCH(1,1)": (np.log(0.5) / np.log(latest_garch["persistence"]), garch_forecasts["horizon_volatility"].iloc[-1] / np.sqrt(cfg.HORIZON_DAYS)),
        "Rolling 252d": (cfg.TRADING_DAYS / 2, spy_returns.rolling(cfg.TRADING_DAYS).std().iloc[-1]),
        "Full sample since 2013": (len(spy_returns) / 2, spy_returns.std()),
    }
    ranking_rows = {name: {"half_life_days": half_life,
                           "annualized_volatility": daily_volatility * np.sqrt(cfg.TRADING_DAYS),
                           "var_99_10d": z_factor * daily_volatility * np.sqrt(cfg.HORIZON_DAYS)}
                    for name, (half_life, daily_volatility) in estimators.items()}
    return pd.DataFrame.from_dict(ranking_rows, orient="index").rename_axis("estimator").sort_values("half_life_days")
