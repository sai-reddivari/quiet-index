"""Leakage prevention, target alignment, baseline causality and common-date scoring."""

import numpy as np
import pandas as pd
import pytest

from quiet_index.forecast import FEATURES, chronological_splits, forecast_frame, garch_one_day, metrics, qlike


def simulated_returns(days=3600):
    return pd.Series(np.random.default_rng(19).normal(0, 0.01, days), index=pd.bdate_range("2013-01-01", periods=days))


def test_features_and_targets_use_explicit_information_dates():
    returns = simulated_returns()
    frame = forecast_frame(returns)
    date = frame.index[100]
    position = returns.index.get_loc(date)
    assert frame.loc[date, "target"] == pytest.approx(returns.iloc[position + 1] ** 2)
    assert frame.loc[date, "label_end"] == returns.index[position + 1]
    assert frame.loc[date, "variance_5d"] == pytest.approx(returns.iloc[position - 4:position + 1].pow(2).mean())
    assert frame.index[0] == returns.index[252]  # no use of a future-seeded EWMA
    assert frame.iloc[-1].target is np.nan or pd.isna(frame.iloc[-1].target)


def test_future_returns_cannot_change_available_features():
    returns = simulated_returns()
    before = forecast_frame(returns)
    changed = returns.copy()
    changed.iloc[1001:] *= 5
    after = forecast_frame(changed)
    pd.testing.assert_frame_equal(before.loc[:returns.index[1000], FEATURES], after.loc[:returns.index[1000], FEATURES])


def test_split_boundary_labels_are_observed_and_do_not_overlap():
    frame = forecast_frame(simulated_returns())
    train, validation, test = chronological_splits(frame)
    assert train.label_end.max() <= pd.Timestamp("2018-12-31")
    assert validation.label_end.max() <= pd.Timestamp("2020-12-31")
    assert train.label_end.max() < validation.index.min()
    assert validation.label_end.max() < test.index.min()
    assert test.label_end.is_unique
    assert (test.label_end > test.index).all()
    # Returns changed after test begins cannot alter training or validation data.
    changed = simulated_returns()
    changed.loc["2021-01-01":] *= 10
    new_train, new_validation, _ = chronological_splits(forecast_frame(changed))
    pd.testing.assert_frame_equal(train, new_train)
    pd.testing.assert_frame_equal(validation, new_validation)


def test_garch_forecast_is_unchanged_by_future_observations():
    returns = simulated_returns(650)
    full, failures = garch_one_day(returns, returns.index[300])
    truncated, _ = garch_one_day(returns.iloc[:480], returns.index[300])
    assert not [row for row in failures if row["kind"] == "failed block"]
    np.testing.assert_allclose(full.iloc[300:480], truncated.iloc[300:], rtol=1e-12)
    assert (full.dropna() > 0).all()


def test_qlike_units_and_positive_forecasts():
    target = np.array([0.0001, 0.0004])
    prediction = np.array([0.0002, 0.0003])
    expected = np.log(prediction) + target / prediction
    np.testing.assert_allclose(qlike(target, prediction), expected)
    with pytest.raises(ValueError, match="positive"):
        qlike(target, [0, 0.0001])


def test_metrics_compare_identical_units_on_identical_dates():
    target = pd.Series([0.0001, 0.0004, 0.0002])
    predictions = pd.DataFrame({"A": [0.0002] * 3, "B": [0.0002] * 3})
    rows = metrics(target, predictions)
    assert rows[0]["observations"] == rows[1]["observations"] == 3
    assert rows[0]["qlike"] == rows[1]["qlike"]
    assert rows[0]["variance_rmse"] == pytest.approx(np.sqrt(((target - 0.0002) ** 2).mean()))
    with pytest.raises(ValueError, match="same nonempty target dates"):
        metrics(target, predictions.set_axis([1, 2, 3]))
