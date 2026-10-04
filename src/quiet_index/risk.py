"""Risk decomposition: portfolio volatility, marginal and component risk, VaR and ES.

Everything here is parametric-normal. The inputs are a weight vector and an
annualized covariance matrix, so the same functions serve baskets and stocks.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as cfg
from . import covariance


def portfolio_volatility(weights_vector, var_covar_matrix):   #sigma_p = sqrt(w' Sigma w)
    weights_transposed = weights_vector.T
    return np.sqrt(weights_transposed @ var_covar_matrix @ weights_vector)


def risk_decomposition(weights_vector, var_covar_matrix):   #how much of total risk each basket is responsible for
    total_risk = portfolio_volatility(weights_vector, var_covar_matrix)

    marginal_contribution_risk = (var_covar_matrix @ weights_vector) / total_risk   # change in sigma_p per unit of extra weight
    component_risk = weights_vector * marginal_contribution_risk                   # each basket's slice of sigma_p
    risk_share = component_risk / total_risk                                       # sums to 1

    assert np.isclose(component_risk.sum(), total_risk)   # Euler check

    return pd.DataFrame({"weight": weights_vector,
                         "marginal_contribution_risk": marginal_contribution_risk,
                         "component_risk": component_risk,
                         "risk_share": risk_share,
                         "risk_share_to_weight": risk_share / weights_vector})


def tail_coefficients(confidence=cfg.CONFIDENCE):   #the two scaling factors that turn volatility into VaR and ES
    alpha = 1 - confidence
    z_alpha = norm.ppf(alpha)             #left-tail quantile, about -2.326 at 99%
    phi_z = norm.pdf(z_alpha)             #normal density at that quantile
    es_coefficient = phi_z / alpha        #about 2.665 at 99%
    return z_alpha, es_coefficient


def var_es_sensitivities(weights_vector, var_covar_matrix, expected_returns=None,
                         confidence=cfg.CONFIDENCE, horizon_days=None):   #dVaR/dw and dES/dw for every basket
    z_alpha, es_coefficient = tail_coefficients(confidence)
    marginal_contribution_risk = (var_covar_matrix @ weights_vector) / portfolio_volatility(weights_vector, var_covar_matrix)
    if expected_returns is None:
        expected_returns = pd.Series(0.0, index=weights_vector.index)   #mu set to 0 in the sensitivity tables

    horizon_scale = 1.0 if horizon_days is None else np.sqrt(horizon_days / cfg.TRADING_DAYS)   #annual vol to a 10-day vol
    return_scale = 1.0 if horizon_days is None else horizon_days / cfg.TRADING_DAYS

    var_sensitivity = expected_returns * return_scale + z_alpha * marginal_contribution_risk * horizon_scale
    es_sensitivity = expected_returns * return_scale - es_coefficient * marginal_contribution_risk * horizon_scale
    return pd.DataFrame({"var_sensitivity": var_sensitivity, "es_sensitivity": es_sensitivity})


def parametric_var_es(total_risk, confidence=cfg.CONFIDENCE, horizon_days=cfg.HORIZON_DAYS):   #index VaR and ES as returns, mu = 0
    z_alpha, es_coefficient = tail_coefficients(confidence)
    horizon_risk = total_risk * np.sqrt(horizon_days / cfg.TRADING_DAYS)
    return {"var": z_alpha * horizon_risk, "es": -es_coefficient * horizon_risk}


def decomposition_by_estimator(daily_returns, weights_vector):   #run the same decomposition under each covariance estimator
    decomposition_tables = {}
    for estimator_name, var_covar_matrix in covariance.all_estimators(daily_returns).items():
        decomposition_tables[estimator_name] = risk_decomposition(weights_vector, var_covar_matrix)
    return decomposition_tables


def index_risk_by_estimator(daily_returns, weights_vector):   #level of index risk, VaR and ES under each estimator
    summary_rows = {}
    for estimator_name, var_covar_matrix in covariance.all_estimators(daily_returns).items():
        total_risk = portfolio_volatility(weights_vector, var_covar_matrix)
        tail_measures = parametric_var_es(total_risk)
        summary_rows[estimator_name] = {"index_volatility": total_risk,
                                        "var_99_10d": tail_measures["var"],
                                        "es_99_10d": tail_measures["es"]}
    return pd.DataFrame.from_dict(summary_rows, orient="index")


def split_comparison(decomposition_table):   #risk share per unit of weight along the AI split and along the momentum split
    groups = {
        "AI high": ["AI_HI_MOM_HI", "AI_HI_MOM_LO"],
        "AI low": ["AI_LO_MOM_HI", "AI_LO_MOM_LO"],
        "Momentum high": ["AI_HI_MOM_HI", "AI_LO_MOM_HI"],
        "Momentum low": ["AI_HI_MOM_LO", "AI_LO_MOM_LO"],
    }
    comparison_rows = {}
    for group_name, group_baskets in groups.items():
        group_weight = decomposition_table.loc[group_baskets, "weight"].sum()
        group_risk_share = decomposition_table.loc[group_baskets, "risk_share"].sum()
        comparison_rows[group_name] = {"weight": group_weight, "risk_share": group_risk_share,
                                       "risk_share_to_weight": group_risk_share / group_weight}
    return pd.DataFrame.from_dict(comparison_rows, orient="index")


def risk_through_time(daily_returns, weights_vector, window=cfg.ESTIMATION_WINDOW):   #today's weights run through every past covariance estimate
    weights_array = weights_vector.reindex(daily_returns.columns).to_numpy()
    ewma_by_date = covariance.ewma_covariance_path(daily_returns)

    history_rows = {}
    for date in ewma_by_date:   #the EWMA path starts after the first year, the same day the trailing window fills
        day = daily_returns.index.get_loc(date)
        trailing_covar = daily_returns.iloc[day - window + 1 : day + 1].cov().to_numpy() * cfg.TRADING_DAYS
        row = {}
        for estimator_name, covar in [("Trailing 252d", trailing_covar), ("EWMA 0.94", ewma_by_date[date])]:
            total_risk = np.sqrt(weights_array @ covar @ weights_array)
            component_risk = weights_array * (covar @ weights_array) / total_risk
            row[f"{estimator_name} volatility"] = total_risk
            row[f"{estimator_name} M7 risk share"] = component_risk[0] / total_risk   #M7 is the first basket
        history_rows[date] = row
    return pd.DataFrame.from_dict(history_rows, orient="index")


def index_daily_returns(members_returns, weights_all):   #the index itself, from member returns and capital weights
    simple_returns = np.expm1(members_returns)
    weighted_return_each_day = (simple_returns.fillna(0) * weights_all).sum(axis=1)
    available_weight_each_day = (members_returns.notna() * weights_all).sum(axis=1)
    return np.log1p(weighted_return_each_day / available_weight_each_day).iloc[1:]


def stock_marginal_contributions(members_returns, index_returns):   #each stock's marginal contribution to index risk over one window
    index_risk = index_returns.std() * np.sqrt(cfg.TRADING_DAYS)
    covariance_with_index = members_returns.apply(lambda stock_returns: stock_returns.cov(index_returns, min_periods=20)) * cfg.TRADING_DAYS   #a stock not yet listed in this window gets no value
    return covariance_with_index / index_risk   #equals (Sigma w)_i / sigma_p without building the full matrix


def sorting_variable_power(loadings_by_qe, members_returns, weights_all):   #does lambda or momentum line up better with next quarter's risk contributions
    index_returns = index_daily_returns(members_returns, weights_all)
    quarter_end_sorted = sorted(loadings_by_qe)
    power_rows = []
    for quarter_end, following_quarter_end in zip(quarter_end_sorted[:-1], quarter_end_sorted[1:]):
        in_following_quarter = (members_returns.index > quarter_end) & (members_returns.index <= following_quarter_end)
        marginal_contribution = stock_marginal_contributions(members_returns.loc[in_following_quarter],
                                                             index_returns.reindex(members_returns.index[in_following_quarter]))
        sorting_table = loadings_by_qe[quarter_end].drop(index=cfg.M7_FULL, errors="ignore")
        sorting_table = sorting_table.join(marginal_contribution.rename("marginal_contribution")).dropna()

        row = {"quarter_end": quarter_end.date(), "names": len(sorting_table)}
        for variable in ["ai_beta", "momentum", "beta_mkt"]:
            row[f"{variable}_rank_corr"] = sorting_table[variable].corr(sorting_table["marginal_contribution"], method="spearman")
            is_high = sorting_table[variable] >= sorting_table[variable].median()
            group_means = sorting_table["marginal_contribution"].groupby(is_high).transform("mean")
            row[f"{variable}_split_r2"] = 1 - ((sorting_table["marginal_contribution"] - group_means) ** 2).sum() / (
                (sorting_table["marginal_contribution"] - sorting_table["marginal_contribution"].mean()) ** 2).sum()   #share of dispersion explained by the high/low split
        power_rows.append(row)
    return pd.DataFrame(power_rows).set_index("quarter_end")
