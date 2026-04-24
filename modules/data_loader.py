"""
modules/data_loader.py
─────────────────────
Handles all data retrieval from Yahoo Finance.
Provides clean, validated OHLCV data for MASI20 tickers.

Author  : [Your Name]
Project : Early Warning System — CDG
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_START = "2019-01-01"
DEFAULT_END   = datetime.today().strftime("%Y-%m-%d")
MIN_DATA_THRESHOLD = 0.70   # Drop ticker if more than 30% of data is missing


# ─── Core Loader ──────────────────────────────────────────────────────────────

def fetch_prices(
    tickers: list[str],
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    progress: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Download adjusted closing prices for a list of tickers from Yahoo Finance.

    Parameters
    ----------
    tickers  : list of Yahoo Finance ticker strings (e.g. ['ATW.CS', 'IAM.CS'])
    start    : start date string 'YYYY-MM-DD'
    end      : end date string  'YYYY-MM-DD'
    progress : show yfinance download progress bar

    Returns
    -------
    prices   : DataFrame (index=Date, columns=tickers) — clean adj. close prices
    valid    : list of tickers that passed data quality checks
    """
    if not tickers:
        raise ValueError("Ticker list cannot be empty.")

    # Download raw data
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=progress,
        threads=True,
    )

    # Extract closing prices
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Ensure columns match requested tickers
    if len(tickers) == 1:
        prices.columns = tickers

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"

    # ── Quality filter ─────────────────────────────────────────────────────
    total_days = len(prices)
    valid_tickers = []
    dropped = []

    for ticker in prices.columns:
        missing_pct = prices[ticker].isna().mean()
        if missing_pct <= (1 - MIN_DATA_THRESHOLD):
            valid_tickers.append(ticker)
        else:
            dropped.append(ticker)

    if dropped:
        print(f"[DataLoader] Dropped tickers (insufficient data): {dropped}")

    prices = prices[valid_tickers]

    # ── Clean missing values ───────────────────────────────────────────────
    # Forward fill first (carry last known price), then backfill for leading NaN
    prices = prices.ffill().bfill()

    # Drop weekends / non-trading days (rows where all tickers are NaN)
    prices = prices.dropna(how="all")

    return prices, valid_tickers


def fetch_index(
    ticker: str = "^MASI",
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.Series:
    """
    Download the MASI index closing prices.

    Returns
    -------
    pd.Series with Date index and index close values.
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    series = raw["Close"].squeeze()
    series.index = pd.to_datetime(series.index)
    series.name = "MASI"
    return series.ffill().bfill().dropna()


# ─── Returns Calculator ───────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from price DataFrame.

    Log returns are preferred over simple returns for:
    - Time-additivity (compounding consistency)
    - Better statistical properties (closer to normal distribution)
    - Coherence with VaR / risk models

    Returns
    -------
    DataFrame of daily log returns (same shape as prices, first row NaN dropped)
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily simple returns from price DataFrame.
    Used for portfolio value reconstruction.
    """
    return prices.pct_change().dropna()


# ─── Data Validation ──────────────────────────────────────────────────────────

def validate_date_range(start: str, end: str) -> None:
    """Validate that start < end and start >= 2010-01-01."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    if start_dt >= end_dt:
        raise ValueError(f"Start date ({start}) must be before end date ({end}).")
    if start_dt < datetime(2010, 1, 1):
        raise ValueError("Start date cannot be earlier than 2010-01-01.")
    if end_dt > datetime.today():
        raise ValueError("End date cannot be in the future.")


def get_data_summary(prices: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """
    Returns a summary dictionary of the loaded dataset.
    Used for display in Streamlit.
    """
    return {
        "n_tickers":    len(prices.columns),
        "n_days":       len(prices),
        "start_date":   prices.index[0].strftime("%d/%m/%Y"),
        "end_date":     prices.index[-1].strftime("%d/%m/%Y"),
        "missing_pct":  (prices.isna().sum().sum() / prices.size * 100).round(2),
        "tickers":      list(prices.columns),
    }
