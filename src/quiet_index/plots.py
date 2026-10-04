"""Figures. Each function takes tables produced by the pipeline and returns a matplotlib figure."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from . import config as cfg
from . import frontier

SURFACE = "#fcfcfb"          #chart background
INK = "#0b0b0b"              #titles
INK_SECONDARY = "#52514e"    #axis labels, legends, value labels
MUTED = "#898781"            #tick labels
GRID = "#e1e0d9"             #gridlines
AXIS = "#c3c2b7"             #baseline and axis lines
REFERENCE = "#898781"        #the series everything else is compared against
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   #fixed order: blue, orange, aqua, yellow, magenta
BREACH = "#d03b3b"           #reserved for VaR breaches

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK_SECONDARY,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 9,
})

BASKET_LABELS = {"M7": "Mag-7", "AI_HI_MOM_HI": "AI high\nmomentum high", "AI_HI_MOM_LO": "AI high\nmomentum low",
                 "AI_LO_MOM_HI": "AI low\nmomentum high", "AI_LO_MOM_LO": "AI low\nmomentum low"}
ASSET_LABELS = {"M7": "Mag-7", "AI_HI_MOM_HI": "AI hi / mom hi", "AI_HI_MOM_LO": "AI hi / mom lo",
                "AI_LO_MOM_HI": "AI lo / mom hi", "AI_LO_MOM_LO": "AI lo / mom lo"}


def save(fig, name):   #write one figure into figures/
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIGURES / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_weight_vs_risk_share(risk_decomposition):   #capital weight next to risk share under each estimator
    estimators = risk_decomposition.index.get_level_values("estimator").unique()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    positions = np.arange(len(cfg.BASKETS))
    bar_width = 0.17

    capital_weight = risk_decomposition.loc[estimators[0]].loc[cfg.BASKETS, "weight"]
    ax.bar(positions - 1.5 * (bar_width + 0.02), capital_weight, bar_width, color=REFERENCE, label="Capital weight")
    ax.text(positions[0] - 1.5 * (bar_width + 0.02), capital_weight["M7"] + 0.012, f"{capital_weight['M7']:.0%}", ha="center", fontsize=9)
    for number, estimator_name in enumerate(estimators):
        risk_share = risk_decomposition.loc[estimator_name].loc[cfg.BASKETS, "risk_share"]
        bar_position = positions + (number - 0.5) * (bar_width + 0.02)
        ax.bar(bar_position, risk_share, bar_width, color=SERIES[number], label=f"Risk share, {estimator_name}")
        ax.text(bar_position[0], risk_share["M7"] + 0.012, f"{risk_share['M7']:.0%}", ha="center", fontsize=8.5)   #label the Mag-7 bars only

    ax.set_xticks(positions, [BASKET_LABELS[basket] for basket in cfg.BASKETS])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylim(0, 0.70)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Share of index")
    ax.set_title("Capital weight versus share of index volatility, by basket")
    ax.legend(loc="upper right")
    return fig


def plot_ai_loadings(loadings_latest, names_shown=15):   #highest AI loadings outside the Mag-7, with the Mag-7 for comparison
    outside_mag7 = loadings_latest[loadings_latest["basket"] != "M7"].nlargest(names_shown, "ai_beta")
    mag7 = loadings_latest[loadings_latest["basket"] == "M7"]
    shown = pd.concat([outside_mag7["ai_beta"].to_frame().assign(group="Outside the Mag-7"),
                       mag7["ai_beta"].to_frame().assign(group="Mag-7")]).sort_values("ai_beta")

    fig, ax = plt.subplots(figsize=(8, 6.4))
    colors = shown["group"].map({"Outside the Mag-7": SERIES[0], "Mag-7": SERIES[1]})
    ax.barh(shown.index, shown["ai_beta"], height=0.62, color=colors)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AI loading (λ): response to SMH after removing the market")
    ax.set_title(f"AI loading at the latest formation: top {names_shown} names outside the Mag-7, and the Mag-7 itself")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[0]), plt.Rectangle((0, 0), 1, 1, color=SERIES[1])]
    ax.legend(handles, ["Outside the Mag-7", "Mag-7"], loc="lower right")
    return fig


def plot_risk_through_time(risk_through_time):   #today's weights run through each day's covariance estimate
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True)
    panels = [("volatility", "Index volatility at current weights (annualized)"),
              ("M7 risk share", "Mag-7 share of index volatility")]
    for ax, (measure, title) in zip(axes, panels):
        for number, estimator_name in enumerate(["Trailing 252d", "EWMA 0.94"]):
            series = risk_through_time[f"{estimator_name} {measure}"]
            ax.plot(series.index, series, color=SERIES[number], linewidth=1.5, label=estimator_name)
            ax.scatter(series.index[-1], series.iloc[-1], s=36, color=SERIES[number], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.annotate(f"{series.iloc[-1]:.0%}", (series.index[-1], series.iloc[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_title(title)
    axes[0].legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncols=2)
    if "m7_weight" in risk_through_time.attrs:   #the level the risk share would sit at if risk were spread like capital
        axes[1].axhline(risk_through_time.attrs["m7_weight"], color=REFERENCE, linewidth=1)
        axes[1].annotate("Mag-7 capital weight", (risk_through_time.index[len(risk_through_time) // 2], risk_through_time.attrs["m7_weight"]),
                         xytext=(0, -12), textcoords="offset points", ha="center", fontsize=9)
    return fig


def plot_frontier(sample_means, var_covar_matrix, risk_free_rate):   #frontier with and without short positions, under two return scenarios
    Sigma = var_covar_matrix.to_numpy()
    asset_risk = np.sqrt(np.diag(Sigma))
    scenarios = frontier.expected_return_scenarios(sample_means)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, scenario_name in zip(axes, ["Sample means", "50% shrinkage"]):
        mu = scenarios[scenario_name].to_numpy()
        target_returns = np.linspace(mu.min() - 0.02, mu.max() + 0.06, 200)
        ax.plot(frontier.frontier_volatility(mu, Sigma, target_returns), target_returns, color=SERIES[0], label="Frontier, shorts allowed (closed form)")
        long_only = frontier.long_only_frontier(mu, Sigma)
        ax.plot(long_only["volatility"], long_only["expected_return"], color=SERIES[1], label="Frontier, long-only (numerical)")

        ax.scatter(asset_risk, mu, s=36, color=REFERENCE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="Single assets")
        for asset, x, y in zip(sample_means.index, asset_risk, mu):
            label_on_left = asset in ["GLD", "TLT"]   #these two sit beside other points, so their labels go on the other side
            ax.annotate(ASSET_LABELS.get(asset, asset), (x, y), xytext=(-6, -3) if label_on_left else (6, -3),
                        ha="right" if label_on_left else "left", textcoords="offset points", fontsize=8)

        tangency = frontier.long_only_tangency(mu, Sigma, risk_free_rate)
        tangency_risk, tangency_return = np.sqrt(tangency @ Sigma @ tangency), tangency @ mu
        ax.plot([0, tangency_risk], [risk_free_rate, tangency_return], color=SERIES[2], linewidth=1.5, label="Line from the T-bill rate to the long-only tangency")
        ax.scatter(tangency_risk, tangency_return, s=56, color=SERIES[2], edgecolor=SURFACE, linewidth=1.5, zorder=4)

        ax.set_title(scenario_name)
        ax.set_xlabel("Annualized volatility")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlim(0, 0.36)
        ax.set_ylim(-0.06, 0.45)
    axes[0].set_ylabel("Expected return (scenario)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4, bbox_to_anchor=(0.5, -0.06), fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_lambda_loss_curve(loss_curve, calibrated_lambda):   #one-step variance forecast loss across the lambda grid
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(loss_curve.index, loss_curve["qlike"], color=SERIES[0])
    ax.scatter(calibrated_lambda, loss_curve.loc[calibrated_lambda, "qlike"], s=56, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.annotate(f"minimum at λ = {calibrated_lambda}", (calibrated_lambda, loss_curve.loc[calibrated_lambda, "qlike"]),
                xytext=(0, 70), textcoords="offset points", ha="center", fontsize=9,
                arrowprops={"arrowstyle": "-", "color": AXIS})
    ax.set_xlabel("EWMA decay factor λ")
    ax.set_ylabel("QLIKE loss (lower is better)")
    ax.set_title(f"One-step variance forecast loss on SPY, 2014 to {cfg.CALIBRATION_END[:4]}")
    return fig


def plot_backtest_windows(backtest_data, breach_data, calibrated_lambda):   #forward returns against VaR in the four regime windows
    fast_model = f"EWMA {cfg.EWMA_LAMBDA_FAST}"
    calibrated_model = f"EWMA {calibrated_lambda} (calibrated)"
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), sharey=True)
    for ax, (window_name, (window_start, window_end)) in zip(axes.ravel(), cfg.REGIME_WINDOWS.items()):
        window = backtest_data.loc[window_start:window_end]
        breaches = breach_data.loc[window_start:window_end, calibrated_model] == 1
        ax.plot(window.index, window["Forward 10D Return"], color=REFERENCE, linewidth=1.5, label="Forward 10-day return")
        ax.plot(window.index, window[calibrated_model], color=SERIES[0], label=f"VaR, {calibrated_model}")
        ax.plot(window.index, window[fast_model], color=SERIES[1], label=f"VaR, {fast_model}")
        ax.scatter(window.index[breaches], window.loc[breaches, "Forward 10D Return"], s=30, color=BREACH,
                   edgecolor=SURFACE, linewidth=1, zorder=3, label="Breach of the calibrated EWMA VaR")
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_title(f"{window_name}: {int(breaches.sum())} breaches in {len(window)} days (calibrated EWMA)")
        ax.tick_params(axis="x", labelrotation=30)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    return fig


def plot_var_by_half_life(current_var):   #today's VaR from each estimator, ordered from shortest memory to longest
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = [f"{name}  (half-life {half_life:,.0f} d)" for name, half_life in current_var["half_life_days"].items()]
    var_size = -current_var["var_99_10d"]
    ax.barh(labels, var_size, height=0.55, color=SERIES[0])
    for label, value in zip(labels, var_size):
        ax.text(value + 0.001, label, f"{value:.1%}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlim(0, var_size.max() * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("10-day 99% VaR on SPY (size of loss)")
    ax.set_title(f"Current VaR by estimator memory, as of {cfg.STUDY_END}")
    return fig


def plot_mags(mags_cumulative):   #synthetic Mag-7 baskets against the listed ETF
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for number, series_name in enumerate(["Cap-weighted synthetic", "Equal-weighted synthetic", "MAGS"]):
        series = mags_cumulative[series_name]
        ax.plot(series.index, series, color=SERIES[number], label=series_name)
        ax.annotate(f"{series.iloc[-1]:.2f}", (series.index[-1], series.iloc[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_ylabel("Cumulative log return")
    ax.set_title("Synthetic Mag-7 baskets against the MAGS ETF since its listing")
    ax.legend(loc="upper left")
    return fig


def save_all(results):   #every figure in the study
    section_1, section_2, section_3, section_4 = (results[name] for name in ["section_1", "section_2", "section_3", "section_4"])
    risk_through_time = section_2["tables"]["s2_risk_through_time"]
    risk_through_time.attrs["m7_weight"] = section_2["index_weights_vector"]["M7"]

    save(plot_ai_loadings(section_1["tables"]["s1_loadings_latest"]), "s1_ai_loadings")
    save(plot_mags(section_1["mags_cumulative"]), "s1_mags_comparison")
    save(plot_weight_vs_risk_share(section_2["tables"]["s2_risk_decomposition"]), "s2_weight_vs_risk_share")
    save(plot_risk_through_time(risk_through_time), "s2_risk_through_time")
    save(plot_frontier(section_3["sample_means"], section_3["var_covar_matrix"], section_3["risk_free_rate"]), "s3_frontier")
    save(plot_lambda_loss_curve(section_4["tables"]["s4_lambda_loss_curve"], section_4["calibrated_lambda"]), "s4_lambda_loss_curve")
    save(plot_backtest_windows(section_4["backtest_data"], section_4["breach_data"], section_4["calibrated_lambda"]), "s4_backtest_windows")
    save(plot_var_by_half_life(section_4["tables"]["s4_current_var_by_half_life"]), "s4_var_by_half_life")
