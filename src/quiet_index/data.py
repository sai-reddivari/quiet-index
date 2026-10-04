"""Data layer: holdings snapshot, price downloads, local cache, daily log returns.

The analysis never calls yfinance directly. `scripts/download_data.py` pulls
everything once into data/raw, and the functions below read from that cache.
"""

import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from . import config as cfg

PRICE_FILES = {
    "members": cfg.DATA_RAW / "prices_members.parquet",
    "benchmarks": cfg.DATA_RAW / "prices_benchmarks.parquet",
    "spy": cfg.DATA_RAW / "prices_spy.parquet",
    "tbill": cfg.DATA_RAW / "tbill_yield.parquet",
}
MANIFEST_FILE = cfg.RESULTS / "data_manifest.json"   #records source and pull date of every series


def fetch_holdings(url=cfg.HOLDINGS_URL):   #download the iShares OEF holdings file as it stands today
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    response.raise_for_status()
    holdings_text = response.text.lstrip("\ufeff")   #the file starts with a byte-order mark

    file_header = pd.read_csv(io.StringIO(holdings_text), nrows=1, header=None, skiprows=1)   #second line holds "Fund Holdings as of"
    as_of_date = pd.to_datetime(file_header.iloc[0, 1]).strftime("%Y-%m-%d")

    holdings = pd.read_csv(io.StringIO(holdings_text), skiprows=9)
    equity_holdings = holdings[holdings["Asset Class"] == "Equity"].copy()   #drop cash and futures lines
    equity_holdings["Ticker"] = equity_holdings["Ticker"].replace({"BRK B": "BRK-B"})   #Yahoo Finance ticker
    equity_holdings["Weight (%)"] = pd.to_numeric(equity_holdings["Weight (%)"], errors="coerce")
    return equity_holdings, as_of_date


def save_weights_snapshot(equity_holdings, as_of_date):   #write the dated weights file the analysis reads
    weights_snapshot = pd.DataFrame({
        "ticker": equity_holdings["Ticker"].values,
        "name": equity_holdings["Name"].values,
        "sector": equity_holdings["Sector"].values,
        "weight_pct": equity_holdings["Weight (%)"].values,
        "as_of_date": as_of_date,
        "source": cfg.HOLDINGS_URL,
    })
    weights_snapshot.to_csv(cfg.DATA_REFERENCE / f"oef_weights_{as_of_date}.csv", index=False)
    return weights_snapshot


def load_tickers():   #fixed index membership used across the whole study
    return pd.read_csv(cfg.MEMBERS_FILE)["ticker"].tolist()


def load_weights():   #capital weight per ticker from the dated snapshot, rescaled to sum to 1
    weights_snapshot = pd.read_csv(cfg.WEIGHTS_FILE).set_index("ticker")
    weights_all = weights_snapshot["weight_pct"].dropna()
    return (weights_all / weights_all.sum()).rename("weight")


def load_sectors():   #GICS sector per ticker, used only to describe the baskets
    return pd.read_csv(cfg.WEIGHTS_FILE).set_index("ticker")["sector"]


def download_close(tickers, start):   #adjusted closes for a list of tickers
    import yfinance as yfinance   #imported here so the analysis runs without a network connection

    close = yfinance.download(tickers, start=start, auto_adjust=True, group_by="column",
                              threads=True, progress=False)["Close"]
    if isinstance(close, pd.Series):   #a single ticker can come back as a series
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index()


def download_all():   #pull every series the study needs into data/raw and record what was pulled
    cfg.DATA_RAW.mkdir(parents=True, exist_ok=True)
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)

    downloads = {
        "members": download_close(load_tickers(), cfg.MEMBERS_START),
        "benchmarks": download_close(cfg.BENCHMARKS, cfg.MEMBERS_START),
        "spy": download_close([cfg.MARKET], cfg.SPY_START),
        "tbill": download_close([cfg.TBILL], cfg.SPY_START),
    }
    for name, prices in downloads.items():
        prices.to_parquet(PRICE_FILES[name])

    manifest = {
        "pulled_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "price_source": "Yahoo Finance via yfinance, adjusted close (auto_adjust=True)",
        "weights_source": cfg.HOLDINGS_URL,
        "weights_file": cfg.WEIGHTS_FILE.name,
        "members_file": cfg.MEMBERS_FILE.name,
        "study_end": cfg.STUDY_END,
        "series": {name: {"tickers": int(prices.shape[1]),
                          "first_date": prices.index.min().strftime("%Y-%m-%d"),
                          "last_date_pulled": prices.index.max().strftime("%Y-%m-%d")}
                   for name, prices in downloads.items()},
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_prices(name):   #read one cached price table, cut off at the study end date
    if not PRICE_FILES[name].exists():
        raise FileNotFoundError(f"{PRICE_FILES[name]} not found. Run `python scripts/download_data.py` first.")
    return pd.read_parquet(PRICE_FILES[name]).loc[:cfg.STUDY_END]


def log_returns(prices):   #daily log returns, r_t = ln(P_t / P_{t-1})
    return np.log(prices / prices.shift(1))


def load_study_data():   #everything Sections 1 to 4 need, in one dictionary
    tickers_data = load_prices("members")        #member adjusted closes from 2018
    benchmark_data = load_prices("benchmarks")   #SMH, VXUS, TLT, GLD, MAGS from 2018
    spy_data = load_prices("spy")[cfg.MARKET]    #SPY from 2013
    tbill_yield = load_prices("tbill")[cfg.TBILL].dropna() / 100   #quoted in percent, stored as a fraction

    return {
        "tickers_data": tickers_data,
        "benchmark_data": benchmark_data,
        "spy_data": spy_data,
        "members_returns": log_returns(tickers_data),       #individual daily log returns of member stocks
        "benchmark_returns": log_returns(benchmark_data),   #daily log returns of the AI proxy and diversifiers
        "spy_returns": log_returns(spy_data).dropna(),      #market returns, used for beta and the backtest
        "tbill_yield": tbill_yield,
        "weights_all": load_weights(),
    }


def audit_prices(tickers_data):   #data-quality table: coverage, gaps, extremes and stale prices per ticker
    members_returns = log_returns(tickers_data)
    audit_rows = {}
    for ticker in tickers_data.columns:
        first_date = tickers_data[ticker].first_valid_index()   #listing date, or the study start
        prices_since_listing = tickers_data[ticker].loc[first_date:]
        returns_since_listing = members_returns[ticker].loc[first_date:].dropna()
        audit_rows[ticker] = {
            "first_date": first_date.date(),
            "last_date": prices_since_listing.last_valid_index().date(),
            "n_prices": int(prices_since_listing.notna().sum()),
            "gaps_after_start": int(prices_since_listing.isna().sum()),   #missing prices once the stock is trading
            "full_history": bool(first_date == tickers_data.index[0]),
            "zero_return_days": int((returns_since_listing == 0).sum()),  #a long run of zeros would mean a stale price
            "min_daily_return": float(returns_since_listing.min()),
            "max_daily_return": float(returns_since_listing.max()),
        }
    return pd.DataFrame.from_dict(audit_rows, orient="index").rename_axis("ticker")
