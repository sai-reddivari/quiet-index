# quiet-index

A reproducible study of concentration, factor exposure, covariance-estimator sensitivity and tail-risk measurement in the S&P 100.

The project asks two related questions:

1. Where does the index’s risk actually come from, relative to where its capital is invested?
2. How much does the reported level of risk depend on how quickly the estimator forgets the past?

The analysis is implemented as an installable Python package, organized into four research notebooks and backed by committed result tables, figures and estimator tests. All reported results are through **30 September 2026**.

Self-directed research project by Sai Reddivari alongside the CQF.

## Headline findings

- The Mag-7 basket—seven companies represented by eight tickers because both Alphabet share classes are included—holds **46.6% of current index capital** but contributes **55.6%–60.0% of index volatility**, depending on the covariance estimator.
- Combining the Mag-7 with the high-AI-loading, high-momentum basket produces **70.0% of current capital** and **81.6%–89.1% of index risk**.
- AI-related co-movement extends beyond the Mag-7, but narrowly: **13 companies outside the Mag-7 have an AI loading above 0.5**, representing **14.4% of current index weight** and concentrated largely in semiconductors, memory, networking and power equipment.
- AI loading is substantially more informative than momentum about next-quarter marginal risk contribution: its average rank correlation is **0.69**, versus **0.17** for momentum, and it wins in **28 of 30 quarters**. Ordinary market beta remains stronger at **0.77**.
- The same current portfolio is measured at **24.1% annualized volatility** using the full sample, **15.6%** using the trailing year and **11.6%** using EWMA with λ = 0.94.
- All four parametric-normal VaR models materially under-cover the left tail. From 2014 through September 2026, they record **75–111 breaches**, versus approximately **32 expected** at 99% confidence.
- Current 10-day 99% SPY VaR ranges from **3.7% to 7.8%** solely because estimator memory ranges from a two-day half-life to the full sample since 2013.
- Mean-variance allocations are much more sensitive to expected-return assumptions than to the covariance mechanics. In the long-only portfolios, the Mag-7 allocation moves from **34% to 12% to 0%** as the return assumption moves from sample means to shrunk means to no views.

![Capital weight versus risk share](figures/s2_weight_vs_risk_share.png)

## Research design

### 1. Universe, factors and baskets

The study fixes the current S&P 100 membership over history and divides it into five non-overlapping baskets:

| Basket | Construction |
|---|---|
| `M7` | AAPL, MSFT, NVDA, AMZN, GOOGL, GOOG, META and TSLA |
| `AI_HI_MOM_HI` | Above-median AI loading and above-median momentum |
| `AI_HI_MOM_LO` | Above-median AI loading and below-median momentum |
| `AI_LO_MOM_HI` | Below-median AI loading and above-median momentum |
| `AI_LO_MOM_LO` | Below-median AI loading and below-median momentum |

The two sorting variables are estimated at each completed calendar quarter-end:

- **Market beta:** OLS slope of the stock’s daily return on SPY over the trailing 252 trading days.
- **AI loading (λ):** the stock’s exposure to SMH after the market component has been removed from both series. By Frisch–Waugh–Lovell, this equals the SMH coefficient in a joint regression on SPY and SMH.
- **Momentum:** the trailing 252-day price return excluding the most recent 21 trading days.

Each formation uses only information available on that date and governs the following quarter. Basket membership is therefore reconstructed quarterly without look-ahead.

The capital weights are handled separately. Constituent weights come from the **1 October 2026 OEF snapshot** and are reused throughout the historical basket-return series. At each quarter-end, the newly formed baskets inherit those fixed constituent weights; within each basket, the weights are renormalized across the members with an available return on that day.

The resulting basket-return history runs from **1 April 2019 through 30 September 2026**.

“AI high” means above the cross-sectional median, not necessarily positive AI exposure. At the latest formation, the median AI loading outside the Mag-7 is negative.

![Latest AI loadings](figures/s1_ai_loadings.png)

### 2. Covariance estimation and risk decomposition

The latest basket assignment and current capital-weight vector are evaluated under three annualized covariance estimators:

- Full sample
- Trailing 252 trading days
- EWMA with λ = 0.94, approximately an 11-day half-life

For weights \(w\) and covariance matrix \(\Sigma\):

\[
\sigma_p=\sqrt{w^\top\Sigma w}
\]

\[
\mathrm{MCR}_i=\frac{(\Sigma w)_i}{\sigma_p}
\]

\[
\mathrm{ComponentRisk}_i=w_i\mathrm{MCR}_i
\]

Component risks satisfy the Euler identity and sum to total portfolio volatility. Risk share is component risk divided by portfolio volatility.

| Estimator | Index volatility | 10-day 99% VaR | Mag-7 risk share |
|---|---:|---:|---:|
| Full sample | 24.1% | −11.2% | 56.2% |
| Trailing 252 days | 15.6% | −7.2% | 55.6% |
| EWMA 0.94 | 11.6% | −5.4% | 60.0% |

The level of measured risk is highly estimator-dependent, while the broad shape of the decomposition is comparatively stable.

The historical risk chart holds today’s basket weights fixed and runs them through each date’s covariance estimate. It is therefore a comparison of estimator behavior through time, not a reconstruction of the index’s historical composition.

![Risk through time](figures/s2_risk_through_time.png)

### 3. Mean-variance context

The five basket return series are combined with three potential diversifiers:

- `VXUS`: non-US equity
- `TLT`: long-duration US Treasuries
- `GLD`: gold

The project computes:

- the unconstrained efficient frontier in closed form;
- an independent SLSQP optimizer check;
- unconstrained and long-only tangency portfolios;
- long-only portfolios at 12% and 18% volatility targets;
- minimum-variance portfolios under a no-views scenario.

Three expected-return scenarios are used:

1. Historical sample means
2. 50% shrinkage toward the cross-sectional grand mean
3. No relative return views

Expected returns are scenarios, not forecasts. The extreme unconstrained allocations produced from sample means are treated as evidence of estimation error rather than implementable portfolios.

![Mean-variance frontiers](figures/s3_frontier.png)

### 4. VaR backtesting

Four volatility models are compared on SPY:

- Rolling 21-day volatility
- Fast EWMA with λ = 0.72
- EWMA calibrated on pre-2020 data
- Expanding-window GARCH(1,1), re-estimated every 21 trading days

The calibrated EWMA uses λ = **0.93**, selected by one-step QLIKE variance-forecast loss over 2014–2019.

Conventions:

- 99% confidence
- 10-trading-day horizon
- VaR on day \(t\) uses returns available through day \(t\)
- Realized return is \(\ln(S_{t+10}/S_t)\)
- A breach occurs when the forward return falls below VaR
- Zones use Binomial(\(T,0.01\)) percentiles
- Formal checks use Kupiec coverage and Christoffersen independence tests

| Window | Rolling 21d | EWMA 0.72 | EWMA 0.93 | GARCH(1,1) |
|---|---:|---:|---:|---:|
| Feb–Jun 2020 | 17 | 13 | 17 | 17 |
| 2022 | 6 | 12 | 5 | 13 |
| Feb–Jun 2025 | 10 | 10 | 8 | 8 |
| 2026 YTD | 0 | 0 | 0 | 0 |
| Full period | 96 | 111 | 81 | 75 |

The full-period target is approximately 32 breaches. All four models finish in the red zone. Repeating the test on ten sets of non-overlapping 10-day observations reduces mechanical breach clustering but does not remove the under-coverage.

![VaR backtest windows](figures/s4_backtest_windows.png)

## Repository guide

| Path | Purpose |
|---|---|
| [`notebooks/01_universe_factors_baskets.ipynb`](notebooks/01_universe_factors_baskets.ipynb) | Universe audit, factor estimation, basket construction and validation |
| [`notebooks/02_risk_decomposition.ipynb`](notebooks/02_risk_decomposition.ipynb) | Covariance sensitivity, Euler decomposition and VaR/ES sensitivities |
| [`notebooks/03_mean_variance.ipynb`](notebooks/03_mean_variance.ipynb) | Closed-form frontier, optimizer checks and scenario portfolios |
| [`notebooks/04_var_backtest.ipynb`](notebooks/04_var_backtest.ipynb) | EWMA calibration, GARCH forecasts and VaR backtesting |
| [`src/quiet_index/`](src/quiet_index/) | Reusable analysis package |
| [`scripts/download_data.py`](scripts/download_data.py) | Downloads and caches the raw market data |
| [`scripts/run_analysis.py`](scripts/run_analysis.py) | Runs all four sections and writes every table and figure |
| [`results/`](results/) | Committed CSV outputs behind the reported numbers |
| [`figures/`](figures/) | Committed publication-ready figures |
| [`tests/`](tests/) | Closed-form and simulated-data estimator checks |
| [`data/reference/`](data/reference/) | Fixed membership and dated holdings-weight snapshots |
| [`results/data_manifest.json`](results/data_manifest.json) | Data sources, pull timestamp and effective study date |

The notebook in [`notebooks/archive/`](notebooks/archive/) is the original exploratory implementation. It is retained for provenance but is not maintained and should not be used to reproduce current results.

## Data

- **Membership:** 101 tickers representing 100 S&P 100 companies, snapshotted on 31 August 2026 from the [iShares OEF holdings file](https://www.ishares.com/us/products/239723/ishares-s-p-100-etf/latest-holdings.csv).
- **Capital weights:** a separate OEF snapshot dated 1 October 2026, normalized to sum to one and applied throughout the historical basket-return construction.
- **Member and benchmark prices:** Yahoo Finance adjusted closes from 2018 onward.
- **SPY and 13-week T-bill history:** Yahoo Finance adjusted closes from 2013 onward.
- **Study cutoff:** 30 September 2026, regardless of later dates present in the downloaded cache.
- **Benchmarks:** SMH, NVDA, MAGS, VXUS, TLT, GLD, SPY and `^IRX`.

Raw price data are cached locally as Parquet files under `data/raw/` and are intentionally excluded from version control. The committed manifest records exactly what was downloaded.

## Reproduce the analysis

Python 3.11 or later is required.

```bash
git clone https://github.com/sai-reddivari/quiet-index.git
cd quiet-index

python -m venv .venv
source .venv/bin/activate

python -m pip install -e ".[dev]"
python scripts/download_data.py
python scripts/run_analysis.py
```

This writes all tables to `results/` and all figures to `figures/`.

To regenerate tables without rendering figures:

```bash
python scripts/run_analysis.py --no-figures
```

To inspect the analysis interactively:

```bash
jupyter lab
```

Then open the numbered notebooks in order.

To create a new dated holdings snapshot:

```bash
python scripts/download_data.py --refresh-weights
```

After refreshing, update `WEIGHTS_FILE` in `src/quiet_index/config.py` if the new snapshot should become the active input.

## Tests

The repository contains 18 tests covering:

- OLS recovery and residual orthogonality
- equivalence of the two-stage and joint AI-loading estimates
- minimum-observation and formation-date rules
- basket assignment and missing-member reweighting
- EWMA recursion and half-life calculations
- positive-definite covariance output
- closed-form versus numerical frontier solutions
- Euler risk decomposition
- analytical VaR and ES sensitivities
- Kupiec and Christoffersen behavior
- prevention of look-ahead in volatility estimates

Run them with:

```bash
pytest -q
```

## Interpretation and limitations

- Current membership is projected backward, creating survivorship bias. Historical returns are descriptive and are not performance claims.
- Basket membership is reconstructed at every quarter-end using contemporaneous AI loading and momentum, but constituent capital weights are not reconstructed historically. The 1 October 2026 OEF weights are reused throughout the basket-return history.
- Because fixed snapshot weights are used, the historical basket series should be interpreted as returns to changing characteristic groups under a constant constituent-weight scheme—not as exact reconstructions of historical OEF or S&P 100 returns.
- The baskets are quarterly characteristic portfolios, not permanent company groups. Roughly one-third of names change basket at a typical formation.
- AI loading measures residual co-movement with SMH after removing SPY. It is not a fundamental measure of a company’s AI revenue or strategy.
- Clustering on residual stock returns does not reproduce the baskets. They should be interpreted as factor sorts, not naturally occurring return clusters.
- The two off-diagonal baskets can contain relatively few companies, making their estimates noisier.
- Expected returns in the optimization section are deliberately presented as scenarios.
- VaR and ES are parametric-normal. The observed return distribution is negatively skewed and heavy-tailed.
- Most backtest observations use overlapping 10-day returns; a separate non-overlapping analysis is included.
- The analysis excludes transaction costs, taxes, liquidity constraints and turnover penalties.

This is a risk-analytics study, not investment advice.

## License

MIT License. See [`LICENSE`](LICENSE).

Analysis code and results are my own; some project scaffolding was AI-assisted.

*Sai Reddivari · 2026*
