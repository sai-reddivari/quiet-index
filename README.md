# quiet-index

Risk decomposition and estimator-sensitivity analysis of the US large-cap index
(S&P 100), in Python. Self-directed project alongside the CQF.

## Questions

1. **Concentration.** How is index variance distributed across holdings relative
   to capital weights — in particular, the risk contribution of the largest
   technology names versus their weight, and how far AI-related return
   covariation extends beyond them. Measured by splitting the index into five
   non-overlapping groups (the Mag-7 block, then the remainder double-sorted on
   an AI factor loading and 12-1 momentum) and computing marginal and component
   risk contributions under several covariance estimators.
2. **Estimator dependence.** How sensitive portfolio volatility and parametric
   VaR/ES are to the estimator's effective lookback (full-sample vs trailing
   window vs EWMA), and how the same estimators grade under standard VaR
   backtests (breach counts, Kupiec, Christoffersen) across 2020, 2022, 2025,
   and the current low-volatility period.

Methods: OLS factor loadings with market orthogonalization (two-stage,
equivalent to a joint two-factor regression), quarterly walk-forward formation
with a trailing 252-day estimation window, Euler risk decomposition, parametric
99%/10-day VaR and ES, EWMA/GARCH volatility. Full definitions and task list in
[TASK_PAPER.md](TASK_PAPER.md).

## Data

Current S&P 100 membership and weights from the iShares OEF holdings file
(snapshot date and source recorded in `data/reference/sp100_members.csv`);
daily adjusted closes via yfinance (members and benchmark ETFs from 2018,
SPY from 2013). Membership is held fixed over the study period — survivorship
bias is disclosed and no performance claims are made. Raw price data is not
committed; the notebook rebuilds it.

## Status

In progress. Factor estimation and basket construction are implemented in
`notebooks/01_data_and_sleeves.ipynb` (31 quarterly formations, 2019–present).
Risk-decomposition results will be recorded here once finalized; backtesting
is next.

## Reproduce

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/01_data_and_sleeves.ipynb
```

Analysis code and results are my own; some project scaffolding was AI-assisted.
Not investment advice.

*Sai Reddivari · MIT License*
