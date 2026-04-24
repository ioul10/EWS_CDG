"""
modules/data_loader.py
──────────────────────
Data retrieval for MASI20 portfolio construction.

Priority order:
  1. Local CSV files (from Investing.com or BVC) — recommended
  2. Yahoo Finance API                            — fallback if available

CSV files must be placed in:  data/prices/<TICKER>.csv
Expected format (Investing.com export):
  Date,Price,Open,High,Low,Vol.,Change %
  "Jan 02, 2019","420.50","418.00","422.00","417.50","125K","0.48%"

Author  : [Your Name]
Project : Early Warning System — CDG
"""

import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_START  = "2019-01-01"
DEFAULT_END    = datetime.today().strftime("%Y-%m-%d")
MIN_DATA_RATIO = 0.70
PRICES_DIR     = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "prices")


# ─── CSV Loader (Investing.com format) ────────────────────────────────────────

def _parse_investing_csv(filepath: str) -> pd.Series:
    """
    Parse a CSV exported from Investing.com.
    Handles date formats like 'Jan 02, 2019' and price strings like '4,200.50'.
    """
    df = pd.read_csv(filepath, encoding="utf-8-sig", thousands=",")
    df.columns = [c.strip().strip('"') for c in df.columns]

    date_col  = next((c for c in df.columns if "date" in c.lower()), None)
    price_col = next((c for c in df.columns if c.lower() in ["price", "close"]), None)

    if not date_col or not price_col:
        raise ValueError(
            f"Colonnes Date/Price introuvables dans {filepath}.\n"
            f"Colonnes trouvées : {list(df.columns)}"
        )

    df[date_col] = df[date_col].astype(str).str.strip().str.strip('"')
    try:
        dates = pd.to_datetime(df[date_col], format="%b %d, %Y")
    except Exception:
        dates = pd.to_datetime(df[date_col], infer_datetime_format=True)

    prices = (
        df[price_col].astype(str)
        .str.strip().str.strip('"')
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    series = pd.Series(prices.values, index=dates, name="Close")
    return series.sort_index()


def _load_from_csv(ticker: str, start: str, end: str) -> Optional[pd.Series]:
    """Try loading from local CSV. Returns None if not found."""
    candidates = [
        os.path.join(PRICES_DIR, f"{ticker}.csv"),
        os.path.join(PRICES_DIR, f"{ticker.replace('.CS','')}.csv"),
        os.path.join(PRICES_DIR, f"{ticker.upper()}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                series = _parse_investing_csv(path)
                series = series.loc[start:end]
                if len(series) > 10:
                    return series
            except Exception as e:
                print(f"[DataLoader] Warning: {path}: {e}")
    return None


def _load_from_yahoo(ticker: str, start: str, end: str) -> Optional[pd.Series]:
    """Try loading from Yahoo Finance. Returns None if unavailable."""
    try:
        import yfinance as yf
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or len(raw) == 0:
            return None
        series = raw["Close"].squeeze()
        series.index = pd.to_datetime(series.index)
        series.name = "Close"
        return series.sort_index()
    except Exception:
        return None


# ─── Main Fetch ───────────────────────────────────────────────────────────────

def fetch_prices(
    tickers: list,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    progress: bool = False,
) -> tuple:
    """
    Load close prices for a list of tickers.
    Tries local CSV first, then Yahoo Finance.

    Returns
    -------
    prices        : DataFrame (Date x Tickers)
    valid_tickers : list of successfully loaded tickers
    sources       : dict {ticker: 'csv' | 'yahoo' | 'failed'}
    """
    if not tickers:
        raise ValueError("La liste de tickers est vide.")

    os.makedirs(PRICES_DIR, exist_ok=True)

    all_series, sources = {}, {}

    for ticker in tickers:
        # 1. CSV
        series = _load_from_csv(ticker, start, end)
        if series is not None:
            sources[ticker] = "csv"
            all_series[ticker] = series
            if progress:
                print(f"✓ {ticker} — CSV ({len(series)} jours)")
            continue

        # 2. Yahoo Finance
        series = _load_from_yahoo(ticker, start, end)
        if series is not None:
            sources[ticker] = "yahoo"
            all_series[ticker] = series
            if progress:
                print(f"✓ {ticker} — Yahoo Finance ({len(series)} jours)")
            continue

        # 3. Failed
        sources[ticker] = "failed"
        if progress:
            print(f"✗ {ticker} — aucune donnée")

    if not all_series:
        raise ValueError(
            "Aucune donnée chargée.\n\n"
            "📁 Placez vos fichiers CSV dans : data/prices/\n"
            "📄 Exemple : data/prices/ATW.csv\n"
            "🌐 Source : https://fr.investing.com → Historique → Télécharger"
        )

    prices = pd.DataFrame(all_series)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    prices = prices.sort_index()

    # Quality filter
    valid_tickers, dropped = [], []
    for ticker in prices.columns:
        if prices[ticker].isna().mean() <= (1 - MIN_DATA_RATIO):
            valid_tickers.append(ticker)
        else:
            dropped.append(ticker)
            sources[ticker] = "failed"

    if dropped:
        print(f"[DataLoader] Ignorés (données insuffisantes): {dropped}")

    prices = prices[valid_tickers].ffill().bfill().dropna(how="all")
    return prices, valid_tickers, sources


def fetch_index(
    ticker: str = "MASI",
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.Series:
    """Load MASI index. Tries CSV first, then Yahoo Finance."""
    series = _load_from_csv(ticker, start, end)
    if series is not None:
        series.name = "MASI"
        return series.ffill().bfill()

    series = _load_from_yahoo("^MASI", start, end)
    if series is not None:
        series.name = "MASI"
        return series.ffill().bfill()

    print("[DataLoader] MASI index non disponible.")
    return pd.Series(dtype=float, name="MASI")


# ─── Returns ──────────────────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns."""
    return prices.pct_change().dropna()


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_date_range(start: str, end: str) -> None:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    if s >= e:
        raise ValueError(f"Date début ({start}) doit être avant date fin ({end}).")
    if s < datetime(2010, 1, 1):
        raise ValueError("Date début ne peut pas être avant 2010.")


def get_data_summary(prices: pd.DataFrame, sources: dict) -> dict:
    return {
        "n_tickers":  len(prices.columns),
        "n_days":     len(prices),
        "start_date": prices.index[0].strftime("%d/%m/%Y"),
        "end_date":   prices.index[-1].strftime("%d/%m/%Y"),
        "sources":    sources,
        "tickers":    list(prices.columns),
    }
