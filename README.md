# The Quiet Index

*Is a calm, concentrated S&P actually low-risk — or are the estimators asleep?*

> **Status:** 🚧 in progress. Spec is final ([PROJECT_SPEC.md](PROJECT_SPEC.md)); results land as the notebooks run.

## Why this exists

August 2026: the VIX at its low for the year, the S&P at record highs on AI leadership, index concentration at historic levels — and every standard risk dashboard reading "calm." This project asks whether that calm is real, in three steps:

1. **Decomposition** — *How much of the index's risk is one bet?* The US slice is cut into **five bespoke, non-overlapping baskets** — the Mag-7 anchor, then the ex-M7 S&P 100 double-sorted on two price-only factors, **AI-beta × 12-1 momentum** (the way a bank desk builds custom baskets for institutional clients). Marginal and component VaR/ES per basket: capital weights vs risk contributions, side by side. The rule-based slicing is then cross-checked against the data's own structure via hierarchical/k-means clustering — if the clusters don't recover the baskets, that's reported too.
2. **Allocation** — *What does the optimizer actually want?* Closed-form and long-only mean-variance frontiers under today's covariance structure (and three different expected-return assumptions, because μ estimates are the weak joint of MVO), with SPY, cap-weight, 60/40 and the tangency portfolio located on the map.
3. **Vigilance** — *Would the standard models have caught the last shock — and how confident should we be now?* 99%/10-day VaR backtests with rolling, EWMA (fixed and calibrated λ) and GARCH(1,1) volatility, graded across four regimes (COVID 2020, the 2022 hiking cycle, Liberation Day 2025, the 2026 calm) using the Basel-style traffic light plus Kupiec and Christoffersen tests — and a capital-cost scoreboard, because a model that never breaches by permanently over-reserving isn't free either.

The risk machinery here (constrained MVO via Lagrangian closed forms, marginal VaR/ES decomposition, EWMA recursion, binomial backtest grading) is the toolkit from Level 1 of the CQF, which I completed with a 93.9% on the risk-and-return exam — applied to a question the exam didn't ask.

## Headline results

*(placeholders — filled from `notebooks/05_findings.ipynb`)*

| Question | Finding |
|---|---|
| Mag-7 share of the US slice: capital weight vs risk contribution | ⏳ `X% of weight, Y% of variance (Σ: EWMA λ=0.94)` |
| Which factor axis carries the ex-M7 risk: AI-beta or momentum? | ⏳ |
| Do data-driven clusters recover the rule-based baskets? | ⏳ `adjusted Rand = …` |
| Frontier allocation to Mag-7 at SPY's current risk budget | ⏳ |
| Today's 99%/10D VaR, by estimator (memory-ordered?) | ⏳ |
| Backtest verdicts by regime (traffic light / Kupiec / Christoffersen) | ⏳ |
| Capital cost: breaches vs mean reserved VaR, by model | ⏳ |

<p align="center"><em>figures/ — weight-vs-risk-contribution bars · frontier map · breach charts · capital-cost scatter</em></p>

## Reproduce

```bash
pip install -r requirements.txt
python -m quiet_index.data     # rebuilds data/raw/ from public sources (nothing vendor-owned is committed)
pytest                          # sanity checks: Euler decomposition, closed-form vs numerical frontier, breach counts
# then run notebooks/01 → 05 in order
```

## Honest limitations

Parametric-normal VaR on overlapping 10-day returns; ETF/synthetic-basket sleeve proxies; baskets are formed from **current** S&P 100 membership, so historical basket series carry survivorship bias (fine for risk decomposition, fatal for alpha claims — none are made); expected returns are scenario assumptions, not forecasts; no transaction costs or taxes. This is a risk-analytics study, not investment advice.

## Provenance

Repo structure and function contracts were scaffolded with AI assistance; all analysis code, results, and findings are written and independently verified by me — the same verify-everything discipline the CQF exam format demands.

*Sai Reddivari · [linkedin.com/in/sai-reddivari-42750srs](https://linkedin.com/in/sai-reddivari-42750srs/) · MIT License*
