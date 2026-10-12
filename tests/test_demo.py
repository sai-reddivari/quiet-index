"""Financial identities, rejected inputs, replay completeness and control behavior."""

import copy

import numpy as np
import pandas as pd
import pytest

from quiet_index import config as cfg
from quiet_index.demo import analyze, historical_replay, scenario_result, validate_weights


@pytest.fixture
def bundle():
    return {"basket_names": cfg.BASKETS, "weights": [0.2] * 5,
            "covariances": {"EWMA 0.94": (np.eye(5) * 0.04).tolist()},
            "historical": {}, "study_end": "2026-09-30", "reproduction": [],
            "data_checks": {"interior_missing_prices": [], "flat_last_six_prices": []},
            "provenance": {"source_sha256": {}, "code_sha256": {}}}


def test_scenario_aggregates_simple_returns_and_currency():
    weights = pd.Series([0.7, 0.3], index=["A", "B"])
    shocks = pd.Series([-0.2, 0.1], index=weights.index)
    result = scenario_result(weights, shocks, 1_000_000)
    assert result["portfolio_return"] == pytest.approx(-0.11)
    assert result["loss_amount"] == pytest.approx(110_000)
    assert sum(row["pnl_amount"] for row in result["contributions"]) == pytest.approx(-result["loss_amount"])


def test_uniform_shock_is_independent_of_weights(bundle):
    result = analyze(bundle, {"weights": [0.7, 0.1, 0.1, 0.1, 0], "shocks": [-0.1] * 5}, "2026-10-12")
    assert result["scenario"]["loss_fraction"] == pytest.approx(0.1)
    assert sum(row["risk_share"] for row in result["decomposition"]) == pytest.approx(1)
    assert result["decomposition"][-1]["risk_share_to_weight"] is None


@pytest.mark.parametrize("values", [[0.2] * 4 + [0.19], [-0.1, 0.3, 0.3, 0.3, 0.2], [np.nan] * 5, [np.inf] * 5])
def test_invalid_weights_are_not_normalized(values):
    with pytest.raises(ValueError, match="sum to 100%"):
        validate_weights(pd.Series(values, index=cfg.BASKETS), cfg.BASKETS)


def test_duplicate_or_inconsistent_weight_labels_rejected():
    with pytest.raises(ValueError, match="exactly once"):
        validate_weights(pd.Series([0.5, 0.5], index=["A", "A"]), ["A", "B"])


@pytest.mark.parametrize("shocks", [[-1.01] * 5, [np.nan] * 5, [np.inf] * 5])
def test_invalid_scenario_inputs_rejected(bundle, shocks):
    with pytest.raises(ValueError, match="simple-return shock"):
        analyze(bundle, {"shocks": shocks}, "2026-10-12")


def test_invalid_covariance_and_limits_rejected(bundle):
    broken = copy.deepcopy(bundle)
    broken["covariances"]["EWMA 0.94"][0][0] = -1
    with pytest.raises(ValueError, match="positive semidefinite"):
        analyze(broken, report_date="2026-10-12")
    for invalid in [{"portfolio_value": -1}, {"concentration_limit": 1.5}, {"scenario_loss_limit": np.nan}]:
        with pytest.raises(ValueError):
            analyze(bundle, invalid, "2026-10-12")


def test_controls_react_to_configured_limits_and_report_date(bundle):
    result = analyze(bundle, {"concentration_limit": 0.19, "scenario_loss_limit": 0.01}, "2026-10-12")
    controls = [row["control"] for row in result["exceptions"]]
    assert controls.count("Component risk concentration") == 5
    assert "Scenario loss" in controls
    assert "Data freshness" in controls
    assert result["data_age_calendar_days"] == 12
    relaxed = analyze(bundle, {"concentration_limit": 1, "scenario_loss_limit": 1, "stale_calendar_days": 30}, "2026-10-12")
    assert relaxed["exceptions"] == []


def test_threshold_equality_does_not_breach_due_to_floating_point_noise(bundle):
    settings = {"weights": [0.4663663664, 0.2336336336, 0.0873873874, 0.0744744745, 0.1381381381],
                "shocks": [-0.1] * 5, "scenario_loss_limit": 0.1}
    result = analyze(bundle, settings, "2026-10-12")
    assert not any(row["control"] == "Scenario loss" for row in result["exceptions"])


def test_historical_replay_uses_endpoints_not_summed_daily_returns():
    dates = pd.to_datetime(["2025-02-19", "2025-02-20", "2025-02-21"])
    prices = pd.DataFrame({name: [100, 110, 99] for name in cfg.BASKETS}, index=dates)
    weights = pd.Series(0.2, index=cfg.BASKETS)
    assignment = pd.Series(cfg.BASKETS, index=cfg.BASKETS)
    result = historical_replay(prices, weights, assignment, "2025-02-19", "2025-02-21")
    assert result["available"]
    assert np.allclose(result["shocks"], -0.01)  # +10%, then -10%, compounds to -1%
    prices.loc[dates[0], "M7"] = np.nan
    blocked = historical_replay(prices, weights, assignment, "2025-02-19", "2025-02-21")
    assert not blocked["available"]
    assert blocked["missing_tickers"] == ["M7"]


def test_blocked_replay_is_not_silently_substituted(bundle):
    bundle["historical"] = {"missing": {"available": False, "reason": "Missing prices"}}
    with pytest.raises(ValueError, match="Missing prices"):
        analyze(bundle, {"scenario": "missing"}, "2026-10-12")


def test_changed_input_fingerprint_is_visible(bundle, tmp_path, monkeypatch):
    source = tmp_path / "input.csv"
    source.write_text("changed")
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    bundle["provenance"]["source_sha256"] = {"input.csv": "old-hash"}
    result = analyze(bundle, report_date="2026-10-12")
    assert any(row["status"] == "REBUILD" for row in result["exceptions"])


def test_fresh_clone_distinguishes_excluded_raw_cache_from_changed_code(bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    bundle["provenance"]["source_sha256"] = {"data/raw/prices_spy.parquet": "original-hash"}
    result = analyze(bundle, report_date="2026-10-12")
    assert any(row["status"] == "NOT VERIFIED LOCALLY" for row in result["exceptions"])
    assert not any(row["status"] == "REBUILD" for row in result["exceptions"])
    bundle["provenance"]["code_sha256"] = {"src/quiet_index/demo.py": "original-hash"}
    result = analyze(bundle, report_date="2026-10-12")
    assert any(row["status"] == "REBUILD" for row in result["exceptions"])


def test_present_but_changed_raw_cache_still_requires_rebuild(bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "prices_spy.parquet").write_bytes(b"changed")
    bundle["provenance"]["source_sha256"] = {"data/raw/prices_spy.parquet": "original-hash"}
    result = analyze(bundle, report_date="2026-10-12")
    assert any(row["status"] == "REBUILD" for row in result["exceptions"])
    assert not any(row["status"] == "NOT VERIFIED LOCALLY" for row in result["exceptions"])
