# PROJECT SPEC — The Quiet Index

**Working title:** `quiet-index`
**Tagline:** *Is a calm, concentrated S&P actually low-risk — or are the estimators asleep?*
**Author:** Sai Reddivari · **Spec version:** 1.1 (2026-08-28)
**Changelog:** v1.1 — Q1 redesigned around five bespoke rule-based baskets (M7 + 2×2 AI-beta × momentum double-sort), with a clustering cross-check and H5 registered. v1.0 — initial spec (ETF sleeves).
**Provenance note:** this spec and the repo scaffold (structure, function contracts, notebook outlines) were drafted with AI assistance; all analysis code, numbers, and written findings are produced and verified by the author. That division of labor is deliberate and should stay true — it mirrors the CQF Exam 1 format (verify AI output independently) and it's the honest answer if anyone asks how the repo was built.

---

## 1. The question

August 2026: the VIX sits at its 2026 low, the S&P has run to record highs on AI leadership, index concentration is at/near record levels, and credible voices are split between melt-up targets and bubble warnings. Every standard risk number an allocator looks at says "calm."

**Central question:** is SPY's measured risk genuinely low, or is it an artifact of (a) a risk budget quietly concentrated in one correlated bet, and (b) volatility estimators whose memory only contains a one-way tape?

Three sub-questions, one notebook arc each:

- **Q1 — Decomposition:** How much of the index's risk is one bet? (weights vs risk contributions)
- **Q2 — Allocation:** Given today's covariance structure, what does mean-variance actually want at a given risk appetite, and how far is cap-weight from it?
- **Q3 — Vigilance:** Would the standard VaR estimators have caught the last shock, and how confident should we be in the current calm? (backtests across regimes, formal tests, capital cost)

## 2. Registered hypotheses (state before running; report even if falsified)

| # | Hypothesis | Falsification criterion |
|---|---|---|
| H1 | The Mag-7 sleeve's **component risk share** of SPY exceeds its cap weight by ≥ 1.3× (e.g. ~35% weight → ≥ 45% of variance). | Component share / weight < 1.3 under both full-sample and EWMA Σ. |
| H2 | At SPY's **current (EWMA) measured σ**, the long-only frontier portfolio allocates materially less to the Mag-7 sleeve than cap-weight does (< half the cap weight). | Frontier weight ≥ half of cap weight. |
| H3 | Current 99%/10D VaR estimates are **ordered by estimator memory** — shortest memory (low-λ EWMA) reports the lowest risk — i.e. the "calm" is partly an estimator artifact. | Ordering does not hold / spread between estimators is economically trivial. |
| H4 | Breach **clustering** (not count) is the dominant failure mode: Christoffersen independence rejects for the rolling-window model in the 2025 window even where Kupiec POF passes. | Independence not rejected. |
| H5 | Within the ex-M7 baskets, the **AI-beta axis explains more cross-sectional risk than the momentum axis**: component-risk shares differ more across the AI-beta split than across the momentum split. | Momentum-split dispersion ≥ AI-beta-split dispersion under both full-sample and EWMA Σ. |

H-numbers are talking points, not thesis claims. If one falsifies, the finding *is* the content — write it up.

## 3. Data plan

**A. The US equity slice — five bespoke baskets (the bank-desk analog).**

The US index is decomposed into five rule-based, non-overlapping baskets formed from the **current S&P 100 membership** (price-only factors — no fundamentals needed):

| Basket | Rule | Working name |
|---|---|---|
| `M7` | AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA — carved out first (preserves H1 and E1 continuity) | The Anchor |
| `AI_HI_MOM_HI` | ex-M7, AI-beta ≥ median **and** 12-1 momentum ≥ median | The Rush |
| `AI_HI_MOM_LO` | ex-M7, AI-beta ≥ median, momentum < median | Left Behind by the Theme |
| `AI_LO_MOM_HI` | ex-M7, AI-beta < median, momentum ≥ median | The Quiet Winners |
| `AI_LO_MOM_LO` | ex-M7, AI-beta < median, momentum < median | The Residual Economy |

**Factor definitions (both computable from daily closes already being pulled):**
- **AI-beta:** two-stage regression per stock — strip the market first (regress rᵢ on r_SPY), then regress the residual on the AI proxy's market-residual (proxy: SMH; NVDA as robustness). The loading is the AI-beta. Trailing 252d window.
- **Momentum:** classic 12-1 (trailing 252d return excluding the most recent 21d).

Median splits within the ex-M7 universe; **cap-weight within basket; reform quarterly** (factor values as of each quarter-end, applied forward — no lookahead). Union of the five baskets ≈ the S&P 100 ≈ ~70% of SPY by weight; the study decomposes that slice and says so plainly.

**Known bias, disclosed not hidden:** current membership introduces survivorship bias historically. Acceptable for a risk-decomposition study (we are not claiming alpha); goes in Limitations. Point-in-time membership is a stretch-tier upgrade.

**Distinctness check (the part that makes this rigorous, not decorative):** after forming baskets, (i) pairwise basket return correlations — if any pair > ~0.95 the slicing failed and that is reported as a finding; (ii) **clustering cross-check (Scheme C as validator):** hierarchical clustering (correlation distance) and k-means (k=5) on the 100 names — do the data's own clusters roughly recover the rule-based baskets? Agreement measured with a confusion matrix / adjusted Rand index. Recovery or non-recovery both go in the write-up. (Also foreshadows CQF Module 5 unsupervised learning.)

**Time-box:** if basket construction exceeds ~3 hours of work, fall back to v1.0 ETF sleeves (MAGS/RSP proxying the split) and ship — the slicing is a supporting act, not the headline.

**B. The multi-asset sleeves (daily adjusted closes):**

| Sleeve | Primary construction | Notes |
|---|---|---|
| US equity | SPY (subject) + the five baskets above as its decomposition | Legacy `XM7` residual `(r_spy − w_t·r_m7)/(1 − w_t)` kept as a robustness check on the basket approach. |
| Ex-US equity | VXUS | EFA if longer history needed |
| Duration | TLT | IEF as milder alternative |
| Gold | GLD | |
| Cash / RFA | BIL (investable) + 3-month T-bill yield for the rate | Rate: FRED `DGS3MO` (or `^IRX`). |

**Mag-7 index weight `w_t`:** hand-collect quarterly snapshots (index factsheets / SlickCharts / press archives) into `data/reference/mag7_weights.csv` with a `source` column per row; forward-fill or linearly interpolate between snapshots. Alternative (stretch): compute from shares outstanding × price. Document whichever is used. Do NOT silently hardcode "35%".

**Sources & licensing:** prices via `yfinance` (or Stooq fallback). Raw downloads land in `data/raw/` which is **gitignored** — the repo ships the *code* to rebuild the dataset plus the small hand-collected reference CSV, not redistributed vendor data. Pin the download date in `data/reference/data_manifest.json`.

**Date range:** 2013-01-01 → present (constrained by youngest primary series). Log returns throughout (consistent with E1).

**Regime windows (used in Q3 and for sub-sample Σ in Q1/Q2):**

| Regime | Window | Why |
|---|---|---|
| COVID crash | 2020-02-01 → 2020-06-30 | fastest vol regime shift on record |
| 2022 hiking cycle | 2022-01-01 → 2022-12-31 | slow grind, positive stock-bond correlation |
| Liberation Day | 2025-02-10 → 2025-06-30 | the shock already analyzed in CQF Exam 1 — continuity |
| The Calm | 2026-01-01 → present | the regime under interrogation |

## 4. Methods (mapped to what's already proven in CQF Exam 1)

### Notebook 01 — Data, factors & basket construction
Download S&P 100 constituents + sleeve ETFs; audit; compute AI-beta and 12-1 momentum per name; form the five baskets (quarterly reform, cap-weight within); build basket return series; validate M7 vs MAGS (corr, tracking diff on overlap); persist tidy returns parquet. **Gates:** stale/missing-data audit passes; baskets partition the ex-M7 universe exactly (no overlap, no orphans — unit-tested); M7-vs-MAGS daily corr ≥ ~0.99 on overlap.

### Notebook 02 — Risk decomposition (Q1) — *E1 Task 2 machinery, new question*
- Σ from (a) full sample, (b) trailing 1y, (c) EWMA (RiskMetrics λ=0.94 daily; report sensitivity to λ).
- **Five-basket view of the US slice** (M7 + the four ex-M7 factor baskets at their cap weights): marginal contribution `MCR_i = (Σw)_i / √(wᵀΣw)`, component risk `w_i·MCR_i` (Euler: components sum to σ_p — unit-tested), component **share** vs **weight** → H1 verdict (M7) and **H5 verdict** (dispersion of component shares across the AI-beta split vs the momentum split).
- Robustness: two-sleeve view via the legacy `XM7` residual — does the M7 risk share match the basket construction?
- **Distinctness & clustering cross-check:** pairwise basket correlations; hierarchical/k-means clusters vs rule-based baskets (confusion matrix, adjusted Rand). Reported either way.
- Multi-asset "allocator view" (US slice, VXUS, TLT, GLD at illustrative documented weights) — same decomposition.
- ∂VaR/∂w_i and ∂ES/∂w_i at 99% (E1 formulas verbatim, cite the exam).
- **Headline figure:** paired bars — capital weight vs risk contribution across the five baskets ("the risk budget doesn't look like the account statement").

### Notebook 03 — Frontier vs cap-weight (Q2) — *E1 Task 1 machinery, multi-asset, honest μ*
- μ is the weak joint of MVO (Module 2 lecture: estimation error, static). Run three μ regimes and show all three rather than pretending one is true: (i) sample means (naive), (ii) shrunk toward grand mean (James-Stein-style or 50% shrink — keep simple, document), (iii) μ-agnostic (min-variance / risk-only).
- Closed-form frontier via A, B, C, D (E1 Task 1); long-only via `scipy.optimize` (E1 Task 2 extension); tangency portfolio with the *actual* current T-bill rate.
- Place on the frontier plot: SPY, current cap-weight 5-sleeve mix, 60/40, equal-weight, min-var, tangency.
- Target-risk portfolios at two appetites: σ = SPY's current EWMA vol ("the calm budget") and σ = SPY's full-sample vol ("the through-cycle budget"); report the M7 allocation each wants → tests H2.
- **Headline figures:** frontier with named portfolios; allocation transition map (weights vs target σ).

### Notebook 04 — VaR backtests across regimes (Q3) — *E1 Tasks 3–4, upgraded*
- 99% / 10-day VaR on SPY and on the M7 sleeve; forward 10D log return; breach if `r_10D < VaR_t` (E1 convention).
- Estimators: rolling-21d SD; EWMA λ=0.72 (E1 continuity); EWMA λ **calibrated** (minimize QLIKE or Gaussian NLL on 1-step variance forecasts — this answers the "I'd tune λ" self-critique from E1); GARCH(1,1) via `arch` (stretch: Student-t innovations).
- Per regime window: breach count/%, binomial traffic light (E1 method, T = window comparisons; note the real Basel light is 250×1-day — stylized version, say so), **Kupiec POF**, **Christoffersen independence & conditional coverage**.
- Overlapping 10D returns inflate clustering — acknowledge; report a non-overlapping variant as robustness.
- **Capital-cost scoreboard:** per model × regime, breaches vs mean |VaR| (a model that never breaches by over-reserving isn't free). 2-axis scatter.
- **The punchline table:** *today's* VaR by estimator, ranked → tests H3. "The model with the shortest memory is the most confident right now."

### Notebook 05 — Findings
Assemble the story: 5–8 exported figures + a findings narrative; write the README results section from here. Include a **Limitations** section (parametric-normal VaR, overlapping windows, proxy sleeves, μ estimation, no transaction costs, not investment advice).

## 5. Engineering quality bar (this is half the signal)

- Notebooks stay thin; logic lives in `src/quiet_index/` with typed signatures and docstrings.
- `tests/` carries the E1 "verify independently" ethos: Euler decomposition sums to σ_p; closed-form frontier σ(m) matches the numerical optimizer; vectorized breach counts match a naive loop; EWMA recursion matches a hand-rolled loop.
- `requirements.txt` pinned; one-command rebuild (`make data && make test` or a `run_all.py`).
- Every figure self-contained (title, units, source, date stamp).
- No committed vendor price data; no secrets; MIT license; "not investment advice" disclaimer.

## 6. Scope tiers & calendar (around CQF exams)

| Tier | Contents | Target |
|---|---|---|
| **MVP** | Notebooks 01 + 02 + 04 with rolling & EWMA-0.72 only; README with 3 figures. Baskets included **only if** construction stays inside the ~3h time-box; otherwise ETF-sleeve fallback for MVP and baskets move to Full. | before Exam 2 crunch (~Sep 7) |
| **Full** | + calibrated λ, Kupiec/Christoffersen, capital-cost scoreboard; Notebook 03 | after Exam 2 (Sep 24) → ~Oct 5, before Exam 3 |
| **Stretch** | GARCH(1,1)-t, non-overlapping robustness, MAGS validation appendix, weights-from-shares-outstanding, point-in-time S&P 100 membership, frontier run over the five baskets as assets | after Exam 3 (Oct 22+) |

MVP alone is already interview-usable: decomposition + backtest is the risk-team double.

## 7. Interview one-liners (what each artifact buys)

- **Repo as a whole:** "I asked whether the calmest tape since the AI run began is actually low-risk, or whether the estimators are asleep. Here's the decomposition, the frontier, and the backtests."
- **NB02:** "Mag-7 is X% of the account statement and Y% of the risk budget — component ES, same math I used on the CQF exam, pointed at a live question."
- **Baskets:** "I sliced the ex-Mag-7 index into four bespoke baskets on two price-only factors — AI-beta and momentum, the way a bank desk builds custom baskets — then let clustering check whether my rules matched the data's own structure." *(APT/multi-factor is Module 2 material; clustering foreshadows Module 5.)*
- **NB03:** "I don't trust point estimates of μ, so I show the frontier under three μ assumptions — the concentration conclusion survives all three." *(verify before claiming)*
- **NB04:** "Breach counts flattered the rolling model; the Christoffersen test didn't — its breaches cluster exactly when you can least afford them."
- **Capital cost:** "Model selection is a trade between breaches and reserved capital — I priced both sides."

## 8. Repo layout

```
quiet-index/
├── README.md                  # public narrative + results (placeholders until run)
├── PROJECT_SPEC.md            # this file
├── LICENSE                    # MIT
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                   # gitignored — rebuilt by src/quiet_index/data.py
│   └── reference/
│       ├── mag7_weights.csv   # hand-collected, sourced per row
│       ├── sp100_members.csv  # current S&P 100 tickers, snapshot date + source
│       └── data_manifest.json # tickers, ranges, download date
├── src/quiet_index/
│   ├── __init__.py
│   ├── data.py                # download, returns, sleeve construction
│   ├── baskets.py             # factor computation, 2x2 double-sort, distinctness/cluster checks
│   ├── risk.py                # Σ builders, VaR/ES, marginal/component
│   ├── optimize.py            # ABCD closed-form, long-only, tangency
│   ├── backtest.py            # 10D pipeline, traffic light, Kupiec, Christoffersen
│   └── plots.py               # shared figure style
├── notebooks/
│   ├── 01_data_and_sleeves.ipynb
│   ├── 02_risk_decomposition.ipynb
│   ├── 03_frontier_vs_capweight.ipynb
│   ├── 04_var_backtests.ipynb
│   └── 05_findings.ipynb
├── figures/                   # exported PNGs referenced by README
└── tests/
    └── test_sanity.py
```
