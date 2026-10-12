"""A frozen, interpretable next-day variance experiment, separate from AI exposure."""

import warnings

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

from . import backtest

FEATURES = ["variance_5d", "variance_21d", "variance_63d", "downside_21d", "squared_return", "ewma_094"]
MODEL_VERSION = "spy-variance-tree-v1"
TRAIN_END = "2018-12-31"
VALIDATION_END = "2020-12-31"
TEST_START = "2021-01-01"


def forecast_frame(returns):
    """At close t, X includes r_t; y is r_(t+1)^2 in decimal-return squared units."""
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError("Returns must have sorted, unique dates")
    if len(returns) < 253 or not np.isfinite(returns).all():
        raise ValueError("At least 253 finite returns are required")
    squared_returns = returns.pow(2)
    frame = pd.DataFrame(index=returns.index)
    for window in [5, 21, 63]:
        frame[f"variance_{window}d"] = squared_returns.rolling(window).mean()
    frame["downside_21d"] = returns.clip(upper=0).pow(2).rolling(21).mean()
    frame["squared_return"] = squared_returns
    frame["ewma_094"] = backtest.ewma_variance(returns, 0.94)
    # The existing EWMA seed uses 252 observations; discard all dates before it is observable.
    frame.loc[frame.index[:252], "ewma_094"] = np.nan
    frame["target"] = squared_returns.shift(-1)
    frame["label_end"] = pd.Series(returns.index, index=returns.index).shift(-1)
    return frame.dropna(subset=FEATURES)


def chronological_splits(frame):
    """Purge boundary labels that would not yet have been observed at the cutoff."""
    train = frame.loc[(frame.index <= TRAIN_END) & (frame.label_end <= TRAIN_END)]
    validation = frame.loc[(frame.index > TRAIN_END) & (frame.index <= VALIDATION_END)
                           & (frame.label_end <= VALIDATION_END)]
    test = frame.loc[(frame.index >= TEST_START) & frame.target.notna()]
    if min(len(train), len(validation), len(test)) < 50:
        raise ValueError("Insufficient history for the fixed chronological experiment")
    return train, validation, test


def qlike(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("QLIKE requires positive finite variance forecasts")
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("QLIKE requires nonnegative finite realized variance proxies")
    return np.log(prediction) + target / prediction


def garch_one_day(returns, first_date):
    """Existing zero-mean normal GARCH(1,1) specification, with explicit fit checks."""
    from arch import arch_model

    values = returns.to_numpy() * 100
    predictions = pd.Series(np.nan, index=returns.index, dtype=float)
    failures = []
    first = int(returns.index.searchsorted(first_date))
    for start in range(first, len(values), 21):
        stop = min(start + 21, len(values))
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                fit = arch_model(values[:start + 1], mean="Zero", vol="GARCH", p=1, q=1).fit(
                    disp="off", show_warning=True)
            if fit.convergence_flag != 0:
                raise ValueError(f"optimizer status {fit.convergence_flag}")
            omega, alpha, beta = (float(fit.params[key]) for key in ["omega", "alpha[1]", "beta[1]"])
            if min(omega, alpha, beta) < 0 or not np.isfinite([omega, alpha, beta]).all():
                raise ValueError("invalid GARCH parameters")
            conditional_variance = float(fit.conditional_volatility[-1] ** 2)
            for day in range(start, stop):
                next_variance = omega + alpha * values[day] ** 2 + beta * conditional_variance
                if not np.isfinite(next_variance) or next_variance <= 0:
                    raise ValueError("invalid GARCH forecast")
                predictions.iloc[day] = next_variance / 10000
                conditional_variance = next_variance
            for warning in captured:
                failures.append({"date": str(returns.index[start].date()), "kind": "warning", "message": str(warning.message)})
        except (ValueError, RuntimeError, FloatingPointError) as error:
            predictions.iloc[start:stop] = np.nan
            failures.append({"date": str(returns.index[start].date()), "kind": "failed block", "message": str(error)})
    return predictions, failures


def metrics(target, predictions):
    if not target.index.equals(predictions.index) or len(target) == 0:
        raise ValueError("Every model must be scored on the same nonempty target dates")
    rows = []
    for name in predictions:
        predicted = predictions[name]
        rows.append({"model": name, "observations": len(target),
                     "qlike": float(qlike(target, predicted).mean()),
                     "variance_rmse": float(np.sqrt(np.mean((target - predicted) ** 2))),
                     "volatility_mae_pp": float(np.mean(np.abs(np.sqrt(target * 252) - np.sqrt(predicted * 252))) * 100)})
    return rows


def block_interval(loss_difference, seed=17, block=20, repetitions=1000):
    """Paired moving-block bootstrap retains local serial dependence; descriptive only."""
    values = np.asarray(loss_difference)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(values) - block + 1, int(np.ceil(len(values) / block)))
        draw = np.concatenate([values[start:start + block] for start in starts])[:len(values)]
        means.append(draw.mean())
    return np.quantile(means, [0.025, 0.975]).tolist()


def run_experiment(returns):
    frame = forecast_frame(returns)
    train, validation, test = chronological_splits(frame)
    candidates = []
    fitted_candidates = []
    for depth in [2, 3, 4]:
        for minimum_leaf in [40, 100]:
            tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=minimum_leaf, random_state=17)
            tree.fit(train[FEATURES], train.target * 10000)
            prediction = np.maximum(tree.predict(validation[FEATURES]) / 10000, 1e-12)
            score = float(qlike(validation.target, prediction).mean())
            candidates.append({"max_depth": depth, "min_samples_leaf": minimum_leaf, "validation_qlike": score})
            fitted_candidates.append(tree)
    selected = int(np.argmin([row["validation_qlike"] for row in candidates]))
    validation_tree = fitted_candidates[selected]
    selected_settings = candidates[selected]
    # Importance is measured only on validation, before any test score is inspected.
    rng = np.random.default_rng(17)
    importance = []
    original_score = selected_settings["validation_qlike"]
    for feature in FEATURES:
        differences = []
        for _ in range(20):
            shuffled = validation[FEATURES].copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            prediction = np.maximum(validation_tree.predict(shuffled) / 10000, 1e-12)
            differences.append(float(qlike(validation.target, prediction).mean()) - original_score)
        importance.append({"feature": feature, "qlike_increase": float(np.mean(differences)),
                           "shuffle_std": float(np.std(differences))})
    # Freeze the selected model after refitting on labels observed by 2020-12-31.
    fit_data = frame.loc[frame.label_end <= VALIDATION_END]
    tree = DecisionTreeRegressor(max_depth=selected_settings["max_depth"],
                                 min_samples_leaf=selected_settings["min_samples_leaf"], random_state=17)
    tree.fit(fit_data[FEATURES], fit_data.target * 10000)
    garch, failures = garch_one_day(returns, validation.index[0])
    predictions = pd.DataFrame(index=frame.index)
    predictions["Rolling 21d"] = frame.variance_21d
    predictions["EWMA 0.94"] = frame.ewma_094
    predictions["EWMA 0.72"] = backtest.ewma_variance(returns, 0.72).reindex(frame.index)
    predictions["GARCH(1,1)"] = garch.reindex(frame.index)
    predictions["Decision tree"] = np.maximum(tree.predict(frame[FEATURES]) / 10000, 1e-12)
    baseline_names = [name for name in predictions if name != "Decision tree"]
    common_validation = validation.index.intersection(predictions[baseline_names].dropna().index)
    if len(common_validation) < 50:
        raise ValueError("Insufficient common validation dates after model failures")
    validation_predictions = predictions.loc[common_validation].copy()
    validation_predictions["Decision tree"] = np.maximum(validation_tree.predict(frame.loc[common_validation, FEATURES]) / 10000, 1e-12)
    validation_metrics = metrics(frame.loc[common_validation, "target"], validation_predictions)
    baseline = min([row for row in validation_metrics if row["model"] != "Decision tree"], key=lambda row: row["qlike"])["model"]
    common_test = test.index.intersection(predictions.dropna().index)
    if len(common_test) < 50:
        raise ValueError("Insufficient common test dates after model failures")
    test_predictions = predictions.loc[common_test]
    target = frame.loc[common_test, "target"]
    test_metrics = metrics(target, test_predictions)
    difference = qlike(target, test_predictions["Decision tree"]) - qlike(target, test_predictions[baseline])
    threshold = float(train.variance_21d.quantile(0.75))
    regimes = []
    for name, mask in [("High trailing volatility", frame.loc[common_test, "variance_21d"] > threshold),
                       ("Ordinary trailing volatility", frame.loc[common_test, "variance_21d"] <= threshold)]:
        for row in metrics(target.loc[mask], test_predictions.loc[mask]):
            regimes.append({"regime": name, **row})
    series = test_predictions.copy()
    series.insert(0, "realized_variance", target)
    series["label_end"] = frame.loc[common_test, "label_end"].dt.strftime("%Y-%m-%d")
    series.index = series.index.strftime("%Y-%m-%d")
    delta = float(np.mean(difference))
    return {
        "version": MODEL_VERSION, "target": "Next trading day's squared SPY log return; noisy zero-mean daily variance proxy",
        "horizon_days": 1, "units": "decimal-return squared; annualized volatility = sqrt(252 × variance)",
        "features": FEATURES, "train": [str(train.index[0].date()), str(train.index[-1].date()), len(train)],
        "validation": [str(validation.index[0].date()), str(validation.index[-1].date()), len(validation)],
        "test": [str(common_test[0].date()), str(common_test[-1].date()), len(common_test)],
        "last_test_label": str(frame.loc[common_test[-1], "label_end"].date()),
        "final_training_label_end": str(fit_data.label_end.max().date()),
        "selected_settings": selected_settings, "candidates": candidates,
        "validation_metrics": validation_metrics, "test_metrics": test_metrics,
        "validation_selected_baseline": baseline, "tree_minus_baseline_qlike": delta,
        "paired_block_95_interval": block_interval(difference), "bootstrap_block_days": 20,
        "conclusion": f"The tree {'improved' if delta < 0 else 'did not improve'} test QLIKE versus {baseline}, selected on validation. Lower QLIKE is better.",
        "regime_threshold_daily_variance": threshold, "regime_metrics": regimes,
        "importance": importance, "tree_rules": export_text(tree, feature_names=FEATURES, decimals=6),
        "failures": failures, "excluded_test_dates": len(test) - len(common_test),
        "excluded_validation_dates": len(validation) - len(common_validation),
        "series": series.rename_axis("date").reset_index().to_dict(orient="records"),
        "latest_forecast_date": str(frame.index[-1].date()),
        "latest_forecasts": {name: float(value) if pd.notna(value) else None for name, value in predictions.iloc[-1].items()},
        "limitations": [
            "No overlapping labels: each forecast targets exactly one subsequent trading day. Volatility clustering still induces dependence; a paired 20-day block bootstrap is descriptive, not proof of significance.",
            "The tree is selected on 2019–2020 validation QLIKE, refit through observed 2020 labels and frozen. GARCH parameters refit every 21 trading days; rolling and EWMA states update daily. No test-based tuning.",
            "Features contain information through the forecast close only. Target scaling by 10,000 is numerical; no preprocessing is fit to test data.",
            "Permutation importance uses validation data; correlated volatility features share information, shuffling disrupts temporal structure, and importance is neither causal nor an independent test.",
            "A single squared return is a noisy variance proxy. Volatility MAE compares annualized forecast volatility to annualized absolute daily return, not to observable true volatility.",
            "This SPY experiment does not validate forecasts or VaR for the five-basket portfolio. The tree cannot extrapolate beyond its training leaf averages.",
            "AI loading is SMH proxy exposure, not machine learning. This decision tree is the separate ML experiment.",
        ],
    }
