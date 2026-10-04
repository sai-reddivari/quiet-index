"""Study settings. Every number that could be changed lives here, defined once."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      #repo root, two levels above this file
DATA_RAW = ROOT / "data" / "raw"                #downloaded prices, never committed
DATA_REFERENCE = ROOT / "data" / "reference"    #small dated snapshots, committed
RESULTS = ROOT / "results"                      #the tables behind every reported number
FIGURES = ROOT / "figures"

MEMBERS_FILE = DATA_REFERENCE / "sp100_members.csv"           #fixed index membership
WEIGHTS_FILE = DATA_REFERENCE / "oef_weights_2026-10-01.csv"  #capital weights, dated snapshot
HOLDINGS_URL = "https://www.ishares.com/us/products/239723/ishares-s-p-100-etf/latest-holdings.csv"

MEMBERS_START = "2018-01-01"   #members and benchmark ETFs
SPY_START = "2013-01-01"       #SPY needs the longer history for the VaR backtest
STUDY_END = "2026-09-30"       #last completed quarter, every result is as of this date

MARKET = "SPY"                 #market factor
AI_PROXY = "SMH"               #AI factor proxy, semiconductor ETF
AI_PROXY_ROBUSTNESS = "NVDA"   #alternative AI proxy for the robustness check
TBILL = "^IRX"                 #13-week T-bill yield, quoted in percent
DIVERSIFIERS = ["VXUS", "TLT", "GLD"]                  #non-US equity, long treasuries, gold
BENCHMARKS = [AI_PROXY] + DIVERSIFIERS + ["MAGS"]

MAG7 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
M7_FULL = MAG7 + ["GOOG"]      #carve-out = Mag-7 plus both Alphabet share classes
BASKETS = ["M7", "AI_HI_MOM_HI", "AI_HI_MOM_LO", "AI_LO_MOM_HI", "AI_LO_MOM_LO"]

TRADING_DAYS = 252             #annualization factor
ESTIMATION_WINDOW = 252        #trailing days used for market beta and the AI loading
MIN_OBS = 200                  #minimum joint observations before a regression is trusted
MOMENTUM_SKIP = 21             #the most recent month is left out of 12-1 momentum
EWMA_LAMBDA = 0.94             #RiskMetrics daily decay for the Section 2 covariance

CONFIDENCE = 0.99              #VaR and ES confidence level
HORIZON_DAYS = 10              #VaR and ES horizon in trading days

SHRINKAGE = 0.5                #weight placed on the grand mean in the shrunk return scenario
VOL_TARGETS = [0.12, 0.18]     #annualized volatility targets for the long-only portfolios

ROLLING_WINDOW = 21            #rolling standard deviation baseline
EWMA_LAMBDA_FAST = 0.72        #fast-decay EWMA baseline
LAMBDA_GRID = [round(0.70 + 0.01 * step, 2) for step in range(30)]   #0.70 to 0.99
CALIBRATION_END = "2019-12-31" #lambda is calibrated on data before the first regime window
BACKTEST_START = "2014-01-02"  #leaves 2013 as burn-in for every volatility model
GARCH_REFIT_EVERY = 21         #trading days between GARCH re-estimations
REGIME_WINDOWS = {
    "2020 Feb-Jun": ("2020-02-01", "2020-06-30"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2025 Feb-Jun": ("2025-02-01", "2025-06-30"),
    "2026 YTD": ("2026-01-01", STUDY_END),
}
ZONE_PERCENTILES = (0.95, 0.9999)   #binomial percentiles that separate green, yellow and red
