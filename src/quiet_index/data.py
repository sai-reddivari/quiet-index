"""Data layer: downloads, return construction, sleeve building.

All prices are daily adjusted closes; all returns are LOG returns
(consistent with CQF E1). Raw downloads land in data/raw/ (gitignored);
this module is the only thing that touches the network.
"""
from __future__ import annotations

import pandas as pd

MAG7 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
SLEEVE_TICKERS = {"SPY": "SPY", "EXUS": "VXUS", "DUR": "TLT", "GOLD": "GLD", "CASH": "BIL"}
VALIDATION_TICKERS = ["MAGS", "RSP"]
START = "2013-01-01"


def download_prices(tickers: list[str], start: str = START, end: str | None = None) -> pd.DataFrame:
    """Fetch daily adjusted closes (yfinance primary, Stooq fallback).

    Returns a Date-indexed DataFrame, one column per ticker. Persists to
    data/raw/<ticker>.csv and records the pull in data/reference/data_manifest.json.
    TODO(sai): implement; fail loudly on missing/short series.
    """
    raise NotImplementedError


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """ln(P_t / P_{t-1}), aligned on the intersection of trading calendars."""
    raise NotImplementedError


def load_mag7_weights(path: str = "data/reference/mag7_weights.csv") -> pd.Series:
    """Daily Mag-7 combined S&P weight w_t, interpolated from quarterly
    hand-collected snapshots. Every snapshot row in the CSV carries a `source`.
    """
    raise NotImplementedError


def build_m7_basket(prices: pd.DataFrame) -> pd.Series:
    """Cap-weighted Mag-7 sleeve return series.

    MVP: weight by market cap snapshots (same cadence as mag7_weights.csv,
    renormalized within the 7). Stretch: daily shares-outstanding x price.
    Validate: daily corr vs MAGS >= ~0.99 on the post-2023-04 overlap.
    """
    raise NotImplementedError


def build_xm7_residual(r_spy: pd.Series, r_m7: pd.Series, w_t: pd.Series) -> pd.Series:
    """Ex-Mag-7 US sleeve: r_xm7 = (r_spy - w_t * r_m7) / (1 - w_t).

    Note: exact for simple returns; applied to log returns it is a
    first-order approximation — document the choice (or convert to simple
    returns for this step and back).
    """
    raise NotImplementedError


def build_sleeves() -> pd.DataFrame:
    """End-to-end: returns DataFrame with columns
    [M7, XM7, SPY, EXUS, DUR, GOLD, CASH] ready for notebooks 02-04.
    Persist to data/raw/sleeves.parquet.
    """
    raise NotImplementedError


if __name__ == "__main__":
    build_sleeves()
