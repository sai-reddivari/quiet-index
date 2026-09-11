# quiet-index — Task Paper

Working document. Sai Reddivari. Last revised 2026-09-14.
Self-directed research project; methods draw on CQF Module 2 material (portfolio
theory, factor models, VaR/ES, volatility estimation). This file is the single
reference for scope, definitions, and task list. Results are recorded in the
notebook and summarized in README.md once reproduced and reviewed.

## 1. Research questions

Using the S&P 100 as a tractable proxy for the US large-cap index:

**Q1 (concentration).** How is index variance distributed across holdings relative
to capital weights? Specifically: (i) the risk contribution of the largest
technology holdings (Mag-7 group, both Alphabet classes) versus their combined
weight; (ii) whether AI-related return covariation, measured as a factor loading,
extends materially beyond that group; (iii) which of two sorting variables —
AI factor loading or 12-1 price momentum — explains more of the cross-sectional
dispersion in risk contributions.

**Q2 (estimator dependence).** How sensitive are standard risk measures
(portfolio volatility, parametric VaR/ES) to the effective lookback of the
covariance/volatility estimator, and would those estimators have passed standard
backtests (breach counts, Kupiec, Christoffersen) through recent stress periods?
Of interest because current point-in-time risk estimates are conditioned on an
unusually calm trailing window.

## 2. Data

- Universe: current S&P 100 membership from the iShares OEF holdings file
  (snapshot 2026-08-31; URL and date recorded in `data/reference/sp100_members.csv`).
  Membership held fixed over the study period. This introduces survivorship bias;
  acceptable for risk decomposition, and no return/performance claims are made.
- Daily adjusted closes via yfinance: members, SMH, VXUS, TLT, GLD, MAGS from
  2018-01-01; SPY and ^IRX from 2013-01-01 (the longer SPY history is required
  for Task 4). Log returns throughout. Raw downloads are not committed.
- Portfolio weights: iShares "Weight (%)" column from the same holdings file.

## 3. Definitions and conventions

- OLS slope b = Cov(y,x)/Var(x); intercept a = mean(y) − b·mean(x); residuals
  e = y − a − bx. Minimum 200 joint observations per fit.
- Market beta: slope of stock daily returns on SPY daily returns, trailing 252
  trading days.
- AI loading (λ): two-stage estimate. Regress SMH on SPY and keep the residual;
  regress each stock's market residual on the SMH residual; λ is the second-stage
  slope. Equivalent (Frisch–Waugh–Lovell) to the SMH coefficient in a joint
  two-factor regression on SPY and SMH. Robustness variant: NVDA as proxy.
- Momentum: trailing 252-day return excluding the most recent 21 days.
- Formation dates: last trading day of each calendar quarter. All estimates at a
  formation date use only data up to that date; assignments apply to the
  following quarter (walk-forward; no lookahead). First usable formation is one
  estimation window after data start.
- Baskets: Mag-7 plus GOOG carved out as one group; remaining names double-sorted
  2×2 on λ and momentum at the within-group medians, labels
  AI_HI/LO × MOM_HI/LO. Cap-weighted within basket using the iShares weights.
- Covariance estimators: full-sample; trailing 252 days; EWMA with λ=0.94
  (RiskMetrics recursion, seeded with the trailing-1y covariance).
- Risk decomposition: portfolio vol σp = √(wᵀΣw); marginal contribution
  MCRᵢ = (Σw)ᵢ/σp; component risk wᵢ·MCRᵢ (sums to σp — verify); risk share =
  component / σp. VaR/ES sensitivities at 99%: ∂VaR/∂wᵢ = μᵢ + z·MCRᵢ with
  z = Φ⁻¹(0.01) ≈ −2.326; ∂ES/∂wᵢ = μᵢ − [φ(z)/0.01]·MCRᵢ, coefficient ≈ 2.665.
  10-day scaling by √10. μ set to 0 for sensitivity tables.
- Backtest conventions: 99% confidence, 10-day horizon; VaR_t = z·σ_t·√10;
  forward return ln(S_{t+10}/S_t); breach when the forward return is below VaR_t.
  Zone thresholds from Binomial(T, 0.01) percentiles (0.95, 0.9999) — a stylized
  version of the Basel traffic light, which is defined on 250 one-day comparisons.
  Formal tests: Kupiec proportion-of-failures; Christoffersen independence.

## 4. Tasks

### Task 1 — Universe, factors, baskets
(a) Build aligned daily log-return series for the universe and benchmarks.
    Report date range, observation counts, exclusions (min-obs rule) and when
    short-history names enter.
(b) At each formation date, estimate market beta and λ per name; compute
    momentum. At the latest formation, plot both cross-sectional distributions
    with medians marked, and report the Spearman rank-correlation matrix of
    {beta, λ, momentum}.
(c) Form the five baskets at each formation date. Report membership counts,
    quarter-to-quarter turnover, current top-5 names per basket, and each
    basket's share of slice capital. Build basket daily return series
    (buy-and-hold within quarter is the target convention; if weights are
    re-fixed daily, say so).
(d) Validation: (i) pairwise basket return correlations, flagging any pair
    above 0.95; (ii) synthetic Mag-7 basket vs MAGS on the overlap — report the
    correlation and attribute any shortfall (MAGS is approximately equal-weighted;
    compare an equal-weight variant); (iii) cluster the names (hierarchical on
    correlation distance; k-means, k=5) and compare with the rule-based labels
    (confusion matrix, adjusted Rand index).
(e) Discussion: how collinear are the two sorting variables, what populates the
    off-diagonal cells, and does the data's own cluster structure resemble the
    rule-based grouping? Report the result either way.

### Task 2 — Risk decomposition
(a) Estimate Σ across the five baskets under the three estimators. Report
    annualized vols and correlation matrices.
(b) At current capital weights: σp, MCR, component risk (verify the Euler sum),
    risk shares. Produce the weight-vs-risk-share bar chart.
(c) ∂VaR/∂w and ∂ES/∂w per basket at 99%/10d.
(d) Discussion: the carve-out group's risk share relative to its weight (report
    the ratio under each estimator); dispersion of risk shares across the λ split
    versus the momentum split; and how all of the above changes with estimator
    lookback. Note explicitly what is and is not stable across estimators.

### Task 3 — Mean-variance context
(a) Closed-form frontier over baskets plus VXUS/TLT/GLD (A, B, C, D scalars);
    verify against a numerical optimizer. Three expected-return scenarios:
    sample means, 50% shrinkage to the grand mean, and none (min-variance only).
(b) Long-only maximum-return portfolios at two vol targets: SPY's current EWMA
    vol and SPY's full-sample vol. Report the carve-out allocation vs its
    cap weight under each μ scenario.
(c) Tangency portfolio using the current 3-month T-bill rate; plot named
    portfolios (cap-weight mix, 60/40, equal-weight, min-var, tangency) in
    (σ, μ) space.
(d) Discussion: which conclusions survive all three μ scenarios; relation to
    the estimation-error critique of mean-variance optimization.

### Task 4 — VaR backtesting
Regime windows: 2020-02-01→2020-06-30; calendar 2022; 2025-02-10→2025-06-30;
2026-01-01→present. SPY, full 2013→present history.
(a) 99%/10d VaR under rolling-21d and EWMA λ=0.72; breach counts/percentages per
    window; plots marking breaches.
(b) Calibrate the EWMA λ by minimizing a one-step variance-forecast loss (QLIKE
    or Gaussian NLL); show the loss curve. Add GARCH(1,1). Re-run (a).
(c) Grade each model × window: zone, Kupiec, Christoffersen. Identify cases
    where the count-based and independence-based tests disagree.
(d) For each model × window, report mean |VaR| (reserved-capital proxy) against
    breach count.
(e) Discussion: rank current VaR estimates across models; relate the ordering to
    estimator half-life; state what this implies about the reliability of
    current risk readings. Robustness: non-overlapping 10-day windows.

Each task ends with a review of outputs before results are recorded in README.md.

## 5. Verification anchors

Known-good values for spot-checking implementations (from CQF Exam 1 work, 93.9%):
- 4-asset MVO universe (μ=(2,7,15,20)%, σ=(5,12,17,25)%, ρ: 0.3 vs A, 0.6 among
  B/C/D; m=4.5%): A=423.61, B=6.807, C=0.9065, D=337.68;
  w=(78.51, 5.39, 13.36, 2.75)%; σp=5.84%.
- 3-asset sensitivity universe (σ=(30,20,15)%, w=(50,20,30)%, ρ=(.8,.5,.3), 99%):
  σp=20.87%; ∂VaR=(−0.684, −0.387, −0.221); |∂ES|=(0.783, 0.443, 0.253).
- Binomial(260, 0.01) zone cutoffs: 5 (p=0.95) and 10 (p=0.9999).
- Current-window sanity: NVDA market beta ≈ 1.9; residuals uncorrelated with the
  regressor to machine precision.

## 6. Status

- Task 1: (a)–(c) implemented in `notebooks/01_data_and_sleeves.ipynb`
  (31 formations, 2019-03-29 → present). (d) validation and turnover reporting
  outstanding; M7-vs-MAGS correlation 0.956 pending attribution.
- Task 2: preliminary full run completed as a scratch pass; results are not
  recorded until reproduced independently in the notebook.
- Tasks 3–4: not started.

## 7. Limitations

Parametric-normal VaR on overlapping 10-day returns; fixed current membership
projected historically (survivorship); current portfolio weights used across the
study period; sorting medians estimated within a ~93-name cross-section;
expected returns treated as scenarios, not forecasts; no transaction costs.
This is a risk-analytics study, not investment advice.
