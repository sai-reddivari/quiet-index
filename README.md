# quiet-index

Risk decomposition and estimator-sensitivity analysis of the US large-cap
index (S&P 100), in Python. Self-directed research project alongside the CQF.
All analysis lives in [`notebooks/quiet_index.ipynb`](notebooks/quiet_index.ipynb).

## Research questions

Using the S&P 100 as a tractable proxy for the US large-cap index:

**Q1 — Concentration.** How is index variance distributed across holdings
relative to capital weights? Specifically: (i) the risk contribution of the
largest technology holdings (the Mag-7 group, both Alphabet classes) versus
their combined weight; (ii) whether AI-related return covariation, measured as
a factor loading, extends materially beyond that group; (iii) which of two
sorting variables — AI factor loading or 12-1 price momentum — explains more
of the cross-sectional dispersion in risk contributions.

**Q2 — Estimator dependence.** How sensitive are standard risk measures
(portfolio volatility, parametric VaR/ES) to the effective lookback of the
covariance estimator, and would those estimators have passed standard
backtests (breach counts, Kupiec, Christoffersen) through recent stress
periods? Of interest because current point-in-time risk estimates are
conditioned on an unusually calm trailing window.

## Data

- Universe: current S&P 100 membership from the iShares OEF holdings file
  (snapshot date and source recorded in `data/reference/sp100_members.csv`).
  Membership is held fixed over the study period; this introduces survivorship
  bias, which is acceptable for risk decomposition, and no return or
  performance claims are made.
- Daily adjusted closes via yfinance: members, SMH, VXUS, TLT, GLD, MAGS from
  2018-01-01; SPY and ^IRX from 2013-01-01 (the longer SPY history is required
  for the backtesting section). Log returns throughout. Raw downloads are not
  committed; the notebook rebuilds them.
- Portfolio weights: the "Weight (%)" column of the same holdings file.

## Definitions and conventions

- OLS slope b = Cov(y,x)/Var(x); intercept a = mean(y) − b·mean(x); residuals
  e = y − a − bx. Minimum 200 joint observations per fit.
- Market beta: slope of stock daily returns on SPY daily returns over a
  trailing 252-trading-day window.
- AI loading (λ): two-stage estimate. Regress SMH on SPY and keep the
  residual; regress each stock's market residual on the SMH residual; λ is the
  second-stage slope. Equivalent (Frisch–Waugh–Lovell) to the SMH coefficient
  in a joint two-factor regression on SPY and SMH. Robustness variant: NVDA as
  the proxy.
- Momentum: trailing 252-day return excluding the most recent 21 days. Used as
  a sorting characteristic, not a regression factor.
- Formation dates: last trading day of each calendar quarter. All estimates at
  a formation date use only data available up to that date; assignments apply
  to the following quarter (walk-forward, no lookahead). The first usable
  formation falls one estimation window after the data start.
- Baskets: Mag-7 plus GOOG carved out as one group; the remaining names are
  double-sorted 2×2 on λ and momentum at the within-group medians
  (labels AI_HI/LO × MOM_HI/LO), cap-weighted within basket.
- Covariance estimators: full-sample; trailing 252 days; EWMA with λ = 0.94
  (RiskMetrics recursion, seeded with the trailing-1y covariance).
- Risk decomposition: portfolio vol σp = √(wᵀΣw); marginal contribution
  MCRᵢ = (Σw)ᵢ/σp; component risk wᵢ·MCRᵢ (Euler: components sum to σp);
  risk share = component/σp. Sensitivities at 99%: ∂VaR/∂wᵢ = μᵢ + z·MCRᵢ with
  z = Φ⁻¹(0.01) ≈ −2.326; ∂ES/∂wᵢ = μᵢ − [φ(z)/0.01]·MCRᵢ (coefficient
  ≈ 2.665). Ten-day scaling by √10; μ set to 0 in sensitivity tables.
- Backtest conventions: 99% confidence, 10-day horizon; VaR_t = z·σ_t·√10;
  forward return ln(S_{t+10}/S_t); breach when the forward return is below
  VaR_t. Zone thresholds from Binomial(T, 0.01) percentiles (0.95, 0.9999) — a
  stylized version of the Basel traffic light, which is defined on 250 one-day
  comparisons. Formal tests: Kupiec proportion-of-failures and Christoffersen
  independence.

## Work plan

**1. Universe, factors, baskets.** Aligned daily return series with a
data-quality audit; per-formation market beta, λ, and momentum; cross-sectional
distributions and the Spearman rank-correlation structure of the three sorting
variables; basket formation with membership counts, turnover between reforms,
and capital shares; validation — pairwise basket correlations, the synthetic
Mag-7 basket against MAGS on their overlap (with attribution of any shortfall
to the weighting-scheme difference), and a clustering cross-check
(hierarchical and k-means against the rule-based labels, adjusted Rand index).

**2. Risk decomposition.** Σ across the five baskets under all three
estimators; marginal and component contributions at current capital weights
with the Euler identity verified; VaR/ES sensitivities; the weight-versus-
risk-share comparison, with attention to which conclusions are stable across
estimator lookbacks.

**3. Mean-variance context.** Closed-form frontier over the baskets plus
VXUS/TLT/GLD, verified against a numerical optimizer, under three
expected-return scenarios (sample means, 50% shrinkage, none); long-only
portfolios at two volatility targets; tangency portfolio at the current
3-month T-bill rate.

**4. VaR backtesting.** SPY, 2013→present, across four regime windows
(2020-02→2020-06, calendar 2022, 2025-02→2025-06, 2026 year-to-date);
rolling-21d and EWMA λ = 0.72 baselines, a calibrated-λ EWMA (one-step
variance-forecast loss, with the loss curve shown) and GARCH(1,1); zone,
Kupiec and Christoffersen grades per model × window; mean reserved VaR against
breach count as the capital-cost tradeoff; ranking of current VaR estimates by
estimator half-life. Robustness: non-overlapping 10-day windows.

## Numerical checks

Implementations are spot-checked against closed-form test cases before use:
a four-asset minimum-variance problem (μ = (2, 7, 15, 20)%, σ = (5, 12, 17,
25)%, ρ = 0.3 versus the first asset and 0.6 elsewhere, target return 4.5%)
with known solution w = (78.51, 5.39, 13.36, 2.75)% and σp = 5.84%; a
three-asset VaR/ES sensitivity case (σ = (30, 20, 15)%, w = (50, 20, 30)%,
ρ = (0.8, 0.5, 0.3), 99%) with ∂VaR = (−0.684, −0.387, −0.221) and |∂ES| =
(0.783, 0.443, 0.253); Binomial(260, 0.01) zone cutoffs of 5 and 10; and
regression sanity checks (a known high-beta name lands in its expected range;
residuals are uncorrelated with the regressor to machine precision).

## Status

In progress. Section 1 (factor estimation and basket construction) is
implemented — 31 quarterly formations, 2019 to present. Section 2 results are
being finalized; Sections 3–4 are next. Remaining work is expected to land
over the coming week.

## Reproduce

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/quiet_index.ipynb
```

## Limitations

Parametric-normal VaR on overlapping 10-day returns; fixed current membership
projected historically (survivorship); current portfolio weights applied
across the study period; sorting medians estimated within a ~90-name
cross-section; expected returns treated as scenarios, not forecasts; no
transaction costs. This is a risk-analytics study, not investment advice.

---

Analysis code and results are my own; some project scaffolding was
AI-assisted. *Sai Reddivari · MIT License*
