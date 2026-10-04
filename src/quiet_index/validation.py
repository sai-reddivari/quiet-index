"""Section 1 validation: are the baskets sensible, and do they survive other ways of forming them?"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from . import baskets
from . import config as cfg
from . import factors


def rank_correlations_by_quarter_end(loadings_by_qe):   #Spearman correlation of the three sorting variables at every formation
    correlation_rows = {}
    for quarter_end, loadings in sorted(loadings_by_qe.items()):
        rank_corr = loadings.drop(index=cfg.M7_FULL, errors="ignore")[["beta_mkt", "ai_beta", "momentum"]].corr(method="spearman")
        correlation_rows[quarter_end.date()] = {"beta_vs_ai": rank_corr.loc["beta_mkt", "ai_beta"],
                                                "beta_vs_momentum": rank_corr.loc["beta_mkt", "momentum"],
                                                "ai_vs_momentum": rank_corr.loc["ai_beta", "momentum"]}
    return pd.DataFrame.from_dict(correlation_rows, orient="index").rename_axis("quarter_end")


def mags_comparison(members_returns, benchmark_returns, weights_all):   #synthetic Mag-7 basket against the listed MAGS ETF
    mags_returns = benchmark_returns["MAGS"].dropna()   #MAGS only starts trading in April 2023
    overlap_dates = mags_returns.index.intersection(members_returns.index)

    cap_weights = weights_all[cfg.M7_FULL]                              #eight tickers at index weights, as in the M7 basket
    equal_weights = pd.Series(1.0, index=cfg.MAG7)                      #seven companies at equal weight, the way MAGS is built
    synthetic_returns = pd.DataFrame({
        "Cap-weighted synthetic": baskets.weighted_basket_return(members_returns.loc[overlap_dates, cfg.M7_FULL], cap_weights),
        "Equal-weighted synthetic": baskets.weighted_basket_return(members_returns.loc[overlap_dates, cfg.MAG7], equal_weights),
    })
    mags_overlap = mags_returns.loc[overlap_dates]

    comparison_rows = {}
    for synthetic_name in synthetic_returns.columns:
        return_difference = synthetic_returns[synthetic_name] - mags_overlap
        comparison_rows[synthetic_name] = {
            "correlation": synthetic_returns[synthetic_name].corr(mags_overlap),
            "tracking_error": return_difference.std() * np.sqrt(cfg.TRADING_DAYS),
            "annualized_volatility": synthetic_returns[synthetic_name].std() * np.sqrt(cfg.TRADING_DAYS),
            "cumulative_log_return": synthetic_returns[synthetic_name].sum(),
            "cumulative_gap_vs_mags": mags_overlap.sum() - synthetic_returns[synthetic_name].sum(),   #positive means MAGS is ahead
        }
    comparison_rows["MAGS"] = {"correlation": 1.0, "tracking_error": 0.0,
                               "annualized_volatility": mags_overlap.std() * np.sqrt(cfg.TRADING_DAYS),
                               "cumulative_log_return": mags_overlap.sum(), "cumulative_gap_vs_mags": 0.0}

    comparison_table = pd.DataFrame.from_dict(comparison_rows, orient="index")
    cumulative_returns = synthetic_returns.assign(MAGS=mags_overlap).cumsum()
    return comparison_table, cumulative_returns


def clustering_cross_check(loadings, basket_assignment, members_returns, spy_returns, quarter_end):   #do data-driven groups agree with the rule-based baskets
    sorting_table = loadings.drop(index=cfg.M7_FULL, errors="ignore").dropna(subset=["ai_beta", "momentum"])
    rule_labels = basket_assignment.reindex(sorting_table.index)

    rank_features = sorting_table[["ai_beta", "momentum"]].rank(pct=True).to_numpy()   #the 2x2 sort only uses ranks, so the clusters get ranks too
    kmeans_labels = KMeans(n_clusters=4, n_init=20, random_state=0).fit_predict(rank_features)
    ward_labels = fcluster(linkage(rank_features, method="ward"), t=4, criterion="maxclust")

    pos = members_returns.index.get_loc(quarter_end)
    members_window = members_returns.iloc[pos - cfg.ESTIMATION_WINDOW + 1 : pos + 1][sorting_table.index]
    spy_window = spy_returns.reindex(members_window.index)
    market_residuals = pd.DataFrame({ticker: factors.ols_regression(members_window[ticker], spy_window)["resid"]
                                     for ticker in members_window.columns})   #strip the market so clusters reflect what is left
    correlation_distance = np.sqrt(2 * (1 - market_residuals.corr().clip(-1, 1)))
    return_labels = fcluster(linkage(squareform(correlation_distance.to_numpy(), checks=False), method="ward"),
                             t=4, criterion="maxclust")

    return {"kmeans_on_sorting_ranks": adjusted_rand_score(rule_labels, kmeans_labels),
            "ward_on_sorting_ranks": adjusted_rand_score(rule_labels, ward_labels),
            "hierarchical_on_residual_correlations": adjusted_rand_score(rule_labels, return_labels)}


def clustering_by_quarter_end(loadings_by_qe, basket_assignment_by_quarter_end, members_returns, spy_returns):   #adjusted Rand index at every formation
    rand_rows = {}
    for quarter_end in sorted(loadings_by_qe):
        rand_rows[quarter_end.date()] = clustering_cross_check(loadings_by_qe[quarter_end], basket_assignment_by_quarter_end[quarter_end],
                                                               members_returns, spy_returns, quarter_end)
    return pd.DataFrame.from_dict(rand_rows, orient="index").rename_axis("quarter_end")


def proxy_robustness(loadings_by_qe, members_returns, spy_returns, tickers_data):   #does the AI sort survive swapping SMH for NVDA as the proxy
    nvda_loadings_by_qe = factors.loadings_at_every_quarter_end(
        members_returns, spy_returns, members_returns[cfg.AI_PROXY_ROBUSTNESS], tickers_data)

    robustness_rows = {}
    for quarter_end in sorted(loadings_by_qe):
        smh_table = loadings_by_qe[quarter_end].drop(index=cfg.M7_FULL, errors="ignore")
        nvda_table = nvda_loadings_by_qe[quarter_end].drop(index=cfg.M7_FULL, errors="ignore")
        names_in_both = smh_table.index.intersection(nvda_table.index)
        smh_high = smh_table.loc[names_in_both, "ai_beta"] >= smh_table.loc[names_in_both, "ai_beta"].median()
        nvda_high = nvda_table.loc[names_in_both, "ai_beta"] >= nvda_table.loc[names_in_both, "ai_beta"].median()
        robustness_rows[quarter_end.date()] = {
            "rank_correlation": smh_table.loc[names_in_both, "ai_beta"].corr(nvda_table.loc[names_in_both, "ai_beta"], method="spearman"),
            "same_side_of_median": (smh_high == nvda_high).mean(),
        }
    return pd.DataFrame.from_dict(robustness_rows, orient="index").rename_axis("quarter_end")
