"""The four sections run in order. Scripts and notebooks both call these functions,
so a number shown in a notebook is the same number written to results/.
"""

import numpy as np
import pandas as pd

from . import backtest, baskets, covariance, data, factors, frontier, risk, validation
from . import config as cfg


def run_section_1(study_data):   #universe, factors, baskets
    members_returns, spy_returns = study_data["members_returns"], study_data["spy_returns"]
    benchmark_returns, tickers_data, weights_all = study_data["benchmark_returns"], study_data["tickers_data"], study_data["weights_all"]

    loadings_by_qe = factors.loadings_at_every_quarter_end(members_returns, spy_returns, benchmark_returns[cfg.AI_PROXY], tickers_data)
    basket_assignment_by_quarter_end = baskets.basket_assignment_at_every_quarter_end(loadings_by_qe)
    basket_daily_returns = baskets.basket_daily_returns(members_returns, weights_all, basket_assignment_by_quarter_end)

    latest_quarter_end = max(loadings_by_qe)
    loadings_latest = loadings_by_qe[latest_quarter_end].join(basket_assignment_by_quarter_end[latest_quarter_end]).join(weights_all)   #latest formation's table, for the cross-section analysis

    basket_summary = pd.DataFrame({
        "annualized_mean_return": basket_daily_returns.mean() * cfg.TRADING_DAYS,
        "annualized_volatility": basket_daily_returns.std() * np.sqrt(cfg.TRADING_DAYS),   #annualized vol of each basket across whole sample
    })
    mags_table, mags_cumulative = validation.mags_comparison(members_returns, benchmark_returns, weights_all)

    return {
        "loadings_by_qe": loadings_by_qe,
        "basket_assignment_by_quarter_end": basket_assignment_by_quarter_end,
        "basket_daily_returns": basket_daily_returns,
        "latest_quarter_end": latest_quarter_end,
        "mags_cumulative": mags_cumulative,
        "tables": {
            "s1_data_audit": data.audit_prices(tickers_data),
            "s1_loadings_latest": loadings_latest.rename_axis("ticker").sort_values("ai_beta", ascending=False),
            "s1_rank_correlations": validation.rank_correlations_by_quarter_end(loadings_by_qe),
            "s1_membership_summary": baskets.membership_summary(basket_assignment_by_quarter_end, weights_all).set_index(["quarter_end", "basket"]),
            "s1_turnover": baskets.turnover_between_formations(basket_assignment_by_quarter_end, weights_all),
            "s1_basket_summary": basket_summary.rename_axis("basket"),
            "s1_basket_correlations": basket_daily_returns.corr().rename_axis("basket"),
            "s1_mags_comparison": mags_table.rename_axis("series"),
            "s1_clustering_rand_index": validation.clustering_by_quarter_end(loadings_by_qe, basket_assignment_by_quarter_end, members_returns, spy_returns),
            "s1_proxy_robustness": validation.proxy_robustness(loadings_by_qe, members_returns, spy_returns, tickers_data),
        },
    }


def run_section_2(study_data, section_1):   #risk decomposition under three covariance estimators
    basket_daily_returns = section_1["basket_daily_returns"]
    latest_baskets = section_1["basket_assignment_by_quarter_end"][section_1["latest_quarter_end"]]
    index_weights_vector = baskets.basket_weights(latest_baskets, study_data["weights_all"])   #each basket's share of current index weight

    decomposition_tables = risk.decomposition_by_estimator(basket_daily_returns, index_weights_vector)
    sensitivity_tables = {estimator_name: risk.var_es_sensitivities(index_weights_vector, var_covar_matrix, horizon_days=cfg.HORIZON_DAYS)
                          for estimator_name, var_covar_matrix in covariance.all_estimators(basket_daily_returns).items()}
    split_tables = {estimator_name: risk.split_comparison(decomposition_table)
                    for estimator_name, decomposition_table in decomposition_tables.items()}

    return {
        "index_weights_vector": index_weights_vector,
        "tables": {
            "s2_risk_decomposition": pd.concat(decomposition_tables, names=["estimator", "basket"]),
            "s2_index_risk": risk.index_risk_by_estimator(basket_daily_returns, index_weights_vector).rename_axis("estimator"),
            "s2_var_es_sensitivities": pd.concat(sensitivity_tables, names=["estimator", "basket"]),
            "s2_split_comparison": pd.concat(split_tables, names=["estimator", "group"]),
            "s2_sorting_variable_power": risk.sorting_variable_power(section_1["loadings_by_qe"], study_data["members_returns"], study_data["weights_all"]),
            "s2_risk_through_time": risk.risk_through_time(basket_daily_returns, index_weights_vector).rename_axis("date"),
        },
    }


def run_section_3(study_data, section_1):   #mean-variance context over the baskets plus three diversifiers
    sample_means, var_covar_matrix = frontier.frontier_inputs(section_1["basket_daily_returns"], study_data["benchmark_returns"])
    risk_free_rate = float(study_data["tbill_yield"].iloc[-1])   #3-month T-bill yield on the last day of the study

    weights_table, statistics_table = frontier.scenario_portfolios(sample_means, var_covar_matrix, risk_free_rate)
    asset_risk = pd.Series(np.sqrt(np.diag(var_covar_matrix)), index=sample_means.index)
    scenario_means = pd.DataFrame(frontier.expected_return_scenarios(sample_means))
    target_returns = [0.05, 0.10, 0.15, 0.20, 0.30]

    return {
        "sample_means": sample_means,
        "var_covar_matrix": var_covar_matrix,
        "risk_free_rate": risk_free_rate,
        "tables": {
            "s3_inputs": scenario_means.assign(annualized_volatility=asset_risk).rename_axis("asset"),
            "s3_correlations": var_covar_matrix.div(asset_risk, axis=0).div(asset_risk, axis=1).rename_axis("asset"),
            "s3_closed_form_check": frontier.closed_form_versus_optimizer(sample_means.to_numpy(), var_covar_matrix.to_numpy(), target_returns),
            "s3_portfolio_weights": weights_table.T.rename_axis(["scenario", "portfolio"]),
            "s3_portfolio_statistics": statistics_table.T.rename_axis(["scenario", "portfolio"]),
        },
    }


def run_section_4(study_data):   #VaR backtest on SPY
    spy_returns, spy_data = study_data["spy_returns"], study_data["spy_data"]

    loss_curve = backtest.lambda_loss_curve(spy_returns)
    calibrated_lambda = backtest.calibrate_lambda(loss_curve)   #lowest one-step variance forecast loss before 2020
    garch_forecasts, garch_parameters = backtest.garch_variance_forecasts(spy_returns)

    var_by_model = backtest.var_models(spy_returns, calibrated_lambda, garch_forecasts)
    ret_10d_forward = backtest.forward_returns(spy_data)
    backtest_data, breach_data = backtest.backtest_table(var_by_model, ret_10d_forward)

    return {
        "calibrated_lambda": calibrated_lambda,
        "backtest_data": backtest_data,
        "breach_data": breach_data,
        "tables": {
            "s4_lambda_loss_curve": loss_curve,
            "s4_garch_parameters": garch_parameters,
            "s4_backtest_grades": backtest.grade_all_windows(backtest_data, breach_data),
            "s4_non_overlapping": backtest.non_overlapping_check(backtest_data, breach_data),
            "s4_current_var_by_half_life": backtest.current_var_by_half_life(spy_returns, calibrated_lambda, garch_forecasts, garch_parameters),
        },
    }


def run_all():   #the whole study from the cached data
    study_data = data.load_study_data()
    section_1 = run_section_1(study_data)
    return {
        "study_data": study_data,
        "section_1": section_1,
        "section_2": run_section_2(study_data, section_1),
        "section_3": run_section_3(study_data, section_1),
        "section_4": run_section_4(study_data),
    }


def save_tables(results):   #write every table to results/ as a csv
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    for section_name in ["section_1", "section_2", "section_3", "section_4"]:
        for table_name, table in results[section_name]["tables"].items():
            table.to_csv(cfg.RESULTS / f"{table_name}.csv", float_format="%.6g")
