"""Shared dashboard/report calculations and reproducible, offline input preparation."""

import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import config as cfg
from . import covariance, data, forecast, pipeline, risk

DEMO_VERSION = "risk-demo-v1"
BUNDLE_PATH = cfg.RESULTS / "demo_inputs.json"
DEFAULT_SETTINGS = {
    "estimator": "EWMA 0.94", "portfolio_value": 1_000_000,
    "concentration_limit": 0.55, "scenario_loss_limit": 0.10,
    "stale_calendar_days": 7, "scenario": "hypothetical",
    "shocks": [-0.20, -0.15, -0.10, -0.05, -0.05],
}
ASSUMPTIONS = [
    "Demonstration thresholds are illustrative controls, not regulatory limits or institutional risk appetite.",
    "Current S&P 100 membership and 1 October 2026 OEF weights are projected backward. Survivorship bias and historical-weight bias remain.",
    "Covariance uses the original quarterly characteristic-basket log returns. Those changing groups are a proxy for current basket risk, not a fixed-holdings return history.",
    "Volatility and 10-day 99% VaR/ES reuse the study's zero-mean normal, square-root-time approximation on weighted basket log returns. Positive loss magnitudes shown here are linear approximations, not exact compounded simple-return tail losses.",
    "Hypothetical shocks are end-to-end simple returns: portfolio P&L / NAV = sum(weight × shock). They are scenarios, not forecasts; no time horizon or probability is assigned.",
    "Historical replay holds today's constituents and initial weights fixed, applies adjusted-price changes between exact endpoints, then sums P&L by today's baskets. It is a hindsight exposure replay, not actual historical fund performance.",
    "No derivatives repricing, liquidity liquidation model, transaction costs, funding constraints or counterparty model is implemented. Equity price stress is not a liquidity-risk measurement.",
    "SPY forecast evaluation is separate from basket risk measurement. AI loading is semiconductor-proxy exposure, not an ML model.",
]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_weights(weights, names):
    weights = pd.Series(weights, dtype=float)
    if weights.index.has_duplicates or set(weights.index) != set(names):
        raise ValueError("Weights must contain each basket exactly once")
    weights = weights.reindex(names)
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1, atol=1e-8, rtol=0):
        raise ValueError("Long-only weights must be finite, nonnegative, and sum to 100%; no silent normalization")
    return weights


def scenario_result(weights, shocks, portfolio_value):
    if not shocks.index.equals(weights.index) or not np.isfinite(shocks).all() or (shocks < -1).any():
        raise ValueError("Each basket needs one finite simple-return shock of at least −100%")
    contribution = weights * shocks
    return {"portfolio_return": float(contribution.sum()), "loss_fraction": float(-contribution.sum()),
            "loss_amount": float(-contribution.sum() * portfolio_value),
            "contributions": [{"basket": name, "weight": float(weights[name]), "shock": float(shocks[name]),
                               "pnl_fraction": float(contribution[name]), "pnl_amount": float(contribution[name] * portfolio_value)} for name in weights.index]}


def historical_replay(prices, stock_weights, assignment, start, end):
    if pd.Timestamp(start) not in prices.index or pd.Timestamp(end) not in prices.index or start >= end:
        return {"available": False, "reason": "Exact, ordered scenario endpoints are not available"}
    if assignment.index.has_duplicates or set(stock_weights.index) != set(assignment.index):
        raise ValueError("Historical holdings and assignments do not align")
    endpoints = prices.reindex(columns=stock_weights.index).loc[[start, end]]
    missing = endpoints.columns[(endpoints.isna() | ~np.isfinite(endpoints) | (endpoints <= 0)).any()].tolist()
    if missing:
        return {"available": False, "reason": "Missing valid endpoint prices; replay blocked without dropping or reweighting holdings", "missing_tickers": missing}
    stock_shocks = endpoints.iloc[1] / endpoints.iloc[0] - 1
    basket_weights = stock_weights.groupby(assignment).sum().reindex(cfg.BASKETS)
    basket_pnl = (stock_weights * stock_shocks).groupby(assignment).sum().reindex(cfg.BASKETS)
    return {"available": True, "shocks": (basket_pnl / basket_weights).tolist(), "start": start, "end": end,
            "description": "Present-day holdings; buy-and-hold adjusted-price replay. Basket allocations can be changed while within-basket weights stay fixed."}


def reproduce_tables(study):
    comparisons = []
    for section in ["section_1", "section_2", "section_3", "section_4"]:
        for name, table in study[section]["tables"].items():
            saved = cfg.RESULTS / f"{name}.csv"
            actual = pd.read_csv(io.StringIO(table.to_csv(float_format="%.6g")))
            expected = pd.read_csv(saved)
            try:
                pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=2e-5, atol=1e-8)
                comparisons.append({"table": name, "status": "reproduced"})
            except AssertionError as error:
                comparisons.append({"table": name, "status": "difference", "detail": str(error)[:1200]})
    return comparisons


def build_bundle():
    raw_weights = pd.read_csv(cfg.WEIGHTS_FILE)
    if raw_weights.ticker.duplicated().any() or not np.isfinite(raw_weights.weight_pct).all() or (raw_weights.weight_pct < 0).any() or raw_weights.weight_pct.sum() <= 0:
        raise ValueError("Holdings snapshot contains duplicate tickers or missing/invalid capital weights")
    spy_prices = data.load_prices("spy")[cfg.MARKET]
    if spy_prices.index.has_duplicates or not spy_prices.index.is_monotonic_increasing or not np.isfinite(spy_prices).all() or (spy_prices <= 0).any():
        raise ValueError("SPY prices must be complete, finite, positive and uniquely ordered; no silent gap removal")
    print("Reproducing the four existing research sections from cached prices…", flush=True)
    study = pipeline.run_all()
    comparisons = reproduce_tables(study)
    section = study["section_1"]
    source_data = study["study_data"]
    prices = source_data["tickers_data"]
    stock_weights = source_data["weights_all"]
    assignment = section["basket_assignment_by_quarter_end"][section["latest_quarter_end"]]
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing or prices.columns.has_duplicates:
        raise ValueError("Price dates and tickers must be sorted/unique")
    if stock_weights.index.has_duplicates or not np.isfinite(stock_weights).all() or (stock_weights < 0).any():
        raise ValueError("Invalid constituent weights")
    if set(stock_weights.index) != set(assignment.index):
        raise ValueError("Every weighted constituent needs a current basket; no silent exclusion")
    if (prices.dropna(how="all") <= 0).any().any():
        raise ValueError("Nonpositive adjusted prices")
    if prices.iloc[-2:].isna().any().any() or not np.isfinite(prices.iloc[-2:]).all().all():
        raise ValueError("Missing/nonfinite current constituent prices; rebuild blocked")
    weights = study["section_2"]["index_weights_vector"]
    validate_weights(weights, cfg.BASKETS)
    historical = {}
    for name, start, end in [("COVID selloff · 2020", "2020-02-19", "2020-03-23"),
                             ("2022 equity drawdown", "2022-01-03", "2022-10-12"),
                             ("2025 spring selloff", "2025-02-19", "2025-04-08")]:
        historical[name] = historical_replay(prices, stock_weights, assignment, start, end)
    print("Evaluating the chronological SPY variance experiment…", flush=True)
    experiment = forecast.run_experiment(source_data["spy_returns"])
    source_paths = list(data.PRICE_FILES.values()) + [cfg.WEIGHTS_FILE, cfg.MEMBERS_FILE, data.MANIFEST_FILE]
    code_paths = (sorted((cfg.ROOT / "src" / "quiet_index").glob("*.py"))
                  + sorted((cfg.ROOT / "src" / "quiet_index" / "demo_assets").glob("*"))
                  + [cfg.ROOT / name for name in ["scripts/build_demo.py", "scripts/run_demo.py", "pyproject.toml", "requirements.txt"]])
    source_hashes = {str(path.relative_to(cfg.ROOT)): fingerprint(path) for path in source_paths}
    code_hashes = {str(path.relative_to(cfg.ROOT)): fingerprint(path) for path in code_paths}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cfg.ROOT, capture_output=True, text=True, check=False).stdout.strip()
    loadings = section["tables"]["s1_loadings_latest"].reset_index()
    holdings = pd.read_csv(cfg.WEIGHTS_FILE)[["ticker", "name", "sector"]]
    loadings = loadings.merge(holdings, on="ticker", how="left")
    audit = section["tables"]["s1_data_audit"]
    missing_inside = []
    for name in prices:
        valid = prices[name].dropna()
        if len(valid):
            count = int(prices.loc[valid.index[0]:valid.index[-1], name].isna().sum())
            if count:
                missing_inside.append({"ticker": name, "gaps": count})
    flat_tickers = prices.columns[(prices.tail(6).nunique() <= 1)].tolist()
    bundle = {
        "version": DEMO_VERSION, "generated_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "study_end": cfg.STUDY_END, "weights_as_of": str(pd.read_csv(cfg.WEIGHTS_FILE).as_of_date.iloc[0]),
        "formation_date": str(section["latest_quarter_end"].date()),
        "risk_sample": [str(section["basket_daily_returns"].index[0].date()), str(section["basket_daily_returns"].index[-1].date()), len(section["basket_daily_returns"])],
        "basket_names": cfg.BASKETS, "weights": weights.tolist(),
        "covariances": {name: matrix.to_numpy().tolist() for name, matrix in covariance.all_estimators(section["basket_daily_returns"]).items()},
        "holdings": json.loads(loadings.to_json(orient="records")), "historical": historical,
        "ml": experiment, "assumptions": ASSUMPTIONS, "reproduction": comparisons,
        "data_checks": {"interior_missing_prices": missing_inside, "flat_last_six_prices": flat_tickers,
                        "late_listing_tickers": audit.index[~audit.full_history].tolist(),
                        "latest_price_date": str(prices.index[-1].date()), "constituents": len(stock_weights)},
        "provenance": {"manifest": json.loads(data.MANIFEST_FILE.read_text()), "git_base_commit": commit,
                       "source_identity": "Generated from working-tree files. git_base_commit records HEAD at generation; it does not assert that this commit contains the exact report source. code_sha256 identifies the source, assets, build scripts and dependency declarations used, including then-uncommitted files.",
                       "source_sha256": source_hashes, "code_sha256": code_hashes,
                       "python": platform.python_version(),
                       "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "arch"]}},
    }
    BUNDLE_PATH.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
    print(f"Saved {BUNDLE_PATH}", flush=True)
    return bundle


def analyze(bundle, supplied=None, report_date=None):
    settings = {**DEFAULT_SETTINGS, **(supplied or {})}
    names = bundle["basket_names"]
    settings["weights"] = settings.get("weights", bundle["weights"])
    if len(settings["weights"]) != len(names):
        raise ValueError("Exactly five basket weights are required")
    weights = validate_weights(pd.Series(settings["weights"], index=names), names)
    for key in ["portfolio_value", "concentration_limit", "scenario_loss_limit", "stale_calendar_days"]:
        value = float(settings[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a positive finite number")
        settings[key] = value
    if settings["concentration_limit"] > 1 or settings["scenario_loss_limit"] > 1:
        raise ValueError("Fractional limits cannot exceed 100%")
    if settings["estimator"] not in bundle["covariances"]:
        raise ValueError("Unknown covariance estimator")
    metrics = []
    decompositions = {}
    for estimator, values in bundle["covariances"].items():
        matrix = pd.DataFrame(values, index=names, columns=names)
        if not np.isfinite(matrix).all().all() or not np.allclose(matrix, matrix.T) or np.linalg.eigvalsh(matrix).min() < -1e-10:
            raise ValueError("Covariance must be finite, symmetric and positive semidefinite")
        volatility = float(risk.portfolio_volatility(weights, matrix))
        if volatility <= 0:
            raise ValueError("Portfolio has zero model volatility")
        tail = risk.parametric_var_es(volatility)
        metrics.append({"estimator": estimator, "volatility": volatility, "var_loss": -float(tail["var"]), "es_loss": -float(tail["es"])})
        decomposition = risk.risk_decomposition(weights, matrix)
        # A zero capital allocation has zero component risk; its ratio is undefined.
        decomposition["risk_share_to_weight"] = decomposition.risk_share.div(weights.replace(0, np.nan))
        decompositions[estimator] = json.loads(decomposition.rename_axis("basket").reset_index().to_json(orient="records"))
    if settings["scenario"] == "hypothetical":
        if len(settings["shocks"]) != len(names):
            raise ValueError("Exactly five shocks are required")
        shocks = pd.Series(settings["shocks"], index=names, dtype=float)
        scenario_description = ("Uniform basket repricing tests broad market exposure. " if np.allclose(shocks, shocks.iloc[0]) else "Differentiated basket repricing tests whether concentrated exposures dominate losses when baskets move by different amounts. ") + "End-to-end simple returns; no assigned probability or horizon."
        scenario_type = "Hypothetical scenario"
    elif settings["scenario"] in bundle["historical"]:
        historical = bundle["historical"][settings["scenario"]]
        if not historical["available"]:
            raise ValueError(historical["reason"])
        shocks = pd.Series(historical["shocks"], index=names)
        scenario_description = f"{historical['start']} to {historical['end']}. {historical['description']}"
        scenario_type = "Historical exposure replay"
    else:
        raise ValueError("Unknown scenario")
    scenario = {**scenario_result(weights, shocks, settings["portfolio_value"]), "type": scenario_type, "description": scenario_description}
    as_of = pd.Timestamp(report_date or datetime.now(ZoneInfo("America/Chicago")).date())
    data_age = (as_of - pd.Timestamp(bundle["study_end"])).days
    if data_age < 0:
        raise ValueError("Report date cannot precede the study cutoff")
    exceptions = []
    selected = decompositions[settings["estimator"]]
    for row in selected:
        if row["risk_share"] > settings["concentration_limit"] + 1e-12:
            exceptions.append({"control": "Component risk concentration", "status": "EXCEPTION", "detail": f"{row['basket']} contributes {row['risk_share']:.1%} of volatility against a {settings['concentration_limit']:.1%} limit.", "action": "Inspect constituent and correlated-basket exposure; compare all covariance estimators before considering a hedge or rebalance."})
    if scenario["loss_fraction"] > settings["scenario_loss_limit"] + 1e-12:
        exceptions.append({"control": "Scenario loss", "status": "EXCEPTION", "detail": f"Loss is {scenario['loss_fraction']:.1%} of NAV against a {settings['scenario_loss_limit']:.1%} limit.", "action": "Investigate the largest basket P&L contributors, hedge basis risk and execution capacity. This demo does not calculate liquidation costs."})
    if data_age > settings["stale_calendar_days"]:
        exceptions.append({"control": "Data freshness", "status": "STALE", "detail": f"Frozen study prices are {data_age} calendar days old; configured threshold is {settings['stale_calendar_days']:g}.", "action": "Use as a dated research demonstration. Refresh and revalidate source data before any live decision."})
    checks = bundle["data_checks"]
    if checks["interior_missing_prices"] or checks["flat_last_six_prices"]:
        exceptions.append({"control": "Price quality", "status": "REVIEW", "detail": f"Interior gaps: {checks['interior_missing_prices']}; unchanged last six prices: {checks['flat_last_six_prices']}.", "action": "Inspect source prices. Research basket code reweights available names; this can change historical exposure."})
    differences = [row["table"] for row in bundle["reproduction"] if row["status"] != "reproduced"]
    if differences:
        exceptions.append({"control": "Research reproduction", "status": "REVIEW", "detail": f"Saved-table differences: {', '.join(differences)}.", "action": "Review recorded numerical differences before revising research conclusions."})
    unavailable_raw_inputs = []
    for relative, expected in {**bundle["provenance"]["source_sha256"], **bundle["provenance"]["code_sha256"]}.items():
        path = cfg.ROOT / relative
        if not path.exists() and relative.startswith("data/raw/"):
            unavailable_raw_inputs.append(relative)
            continue
        if not path.exists() or fingerprint(path) != expected:
            exceptions.append({"control": "Bundle freshness", "status": "REBUILD", "detail": f"Input/code changed since generation: {relative}", "action": "Run scripts/build_demo.py to rebuild the shared dashboard and report inputs."})
    if unavailable_raw_inputs:
        exceptions.append({"control": "Raw-data verification", "status": "NOT VERIFIED LOCALLY",
                           "detail": "Using the published frozen bundle; raw price caches are intentionally excluded from Git: " + ", ".join(unavailable_raw_inputs),
                           "action": "The dashboard remains usable. Reproduction from prices requires the original cache or a fresh download; revised vendor prices may change results. Missing caches are not evidence that code changed."})
    return {"settings": settings, "report_date": str(as_of.date()), "data_age_calendar_days": data_age,
            "metrics": metrics, "decomposition": selected, "scenario": scenario, "exceptions": exceptions,
            "checks_passed": ["Long-only weights sum to 100%", "Covariance matrices are finite, symmetric and positive semidefinite", "Simple-return scenario contributions sum to portfolio P&L", "All weighted names have current basket assignments"],
            "bundle": bundle}
