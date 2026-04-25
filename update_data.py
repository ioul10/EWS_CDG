"""
update_data.py
──────────────
Script de mise à jour automatique des données de marché.

Stratégie :
  1. Lit les CSV existants dans data/prices/
  2. Identifie la dernière date disponible par ticker
  3. Tente de compléter via Yahoo Finance (nouveaux jours uniquement)
  4. Sauvegarde les CSV mis à jour
  5. Génère un rapport de mise à jour

Usage :
  python update_data.py              # Met à jour tous les tickers
  python update_data.py --ticker ATW # Met à jour un seul ticker
  python update_data.py --report     # Affiche l'état des données sans mettre à jour

Planification automatique (crontab) :
  # Mise à jour tous les jours ouvrés à 18h30 (après clôture BVC)
  30 18 * * 1-5 cd /path/to/EWS_CDG && python update_data.py >> logs/update.log 2>&1

Author  : [Your Name]
Project : Early Warning System — CDG
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PRICES_DIR = os.path.join(BASE_DIR, "data", "prices")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")

os.makedirs(PRICES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ─── Tickers ──────────────────────────────────────────────────────────────────

sys.path.insert(0, BASE_DIR)
from data.masi20_tickers import MASI20_UNIVERSE, get_name

TICKERS     = list(MASI20_UNIVERSE.keys())
MASI_TICKER = "MASI"

# Yahoo Finance ticker mapping
YAHOO_MAP = {
    "ATW":   "ATW.CS",
    "BCP":   "BCP.CS",
    "CIH":   "CIH.CS",
    "IAM":   "IAM.CS",
    "LHM":   "LHM.CS",
    "CSR":   "CSR.CS",
    "HPS":   "HPS.CS",
    "MARSA": "MARSA.CS",
    "MNG":   "MNG.CS",
    "SID":   "SID.CS",
    "ADH":   "ADH.CS",
    "ADI":   "ADI.CS",
    "AUTO":  "AUTO.CS",
    "LBV":   "LBV.CS",
    "MUT":   "MUT.CS",
    "MASI":  "^MASI",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    print(line)
    log_path = os.path.join(LOGS_DIR, f"update_{datetime.now().strftime('%Y-%m')}.log")
    with open(log_path, "a") as f:
        f.write(line + "\n")


def get_last_date(ticker: str) -> Optional[datetime]:
    """Return last available date in the CSV for a ticker."""
    path = os.path.join(PRICES_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col is None:
            return None
        dates = pd.to_datetime(df[date_col], infer_datetime_format=True, errors="coerce", format="mixed")
        return dates.dropna().max().to_pydatetime()
    except Exception:
        return None


def is_business_day(d: datetime) -> bool:
    return d.weekday() < 5  # Monday=0 ... Friday=4


def next_business_day(d: datetime) -> datetime:
    d += timedelta(days=1)
    while not is_business_day(d):
        d += timedelta(days=1)
    return d


# ─── Yahoo Finance Fetcher ────────────────────────────────────────────────────

def fetch_new_data(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Fetch new price data from Yahoo Finance for a given date range.
    Returns DataFrame with columns [Date, Price] or None if failed.
    """
    yahoo_ticker = YAHOO_MAP.get(ticker)
    if not yahoo_ticker:
        return None

    try:
        import yfinance as yf
        raw = yf.download(
            yahoo_ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
        if raw is None or len(raw) == 0:
            return None

        df = raw[["Close"]].reset_index()
        df.columns = ["Date", "Price"]
        df["Date"]  = df["Date"].dt.strftime("%b %d, %Y")
        df["Price"] = df["Price"].round(2)
        return df

    except Exception as e:
        log(f"{ticker}: Yahoo Finance error — {e}", "WARNING")
        return None


# ─── CSV Updater ──────────────────────────────────────────────────────────────

def update_ticker(ticker: str, force: bool = False) -> dict:
    """
    Update a single ticker's CSV with the latest available data.

    Returns a status dict.
    """
    path      = os.path.join(PRICES_DIR, f"{ticker}.csv")
    today     = datetime.today().date()
    result    = {"ticker": ticker, "status": None, "rows_added": 0, "last_date": None}

    # Get last available date
    last_date = get_last_date(ticker)

    if last_date:
        result["last_date"] = last_date.date()

        # Already up to date?
        if last_date.date() >= today - timedelta(days=1) and not force:
            result["status"] = "up_to_date"
            log(f"{ticker}: already up to date ({last_date.date()})")
            return result

        fetch_start = next_business_day(last_date).strftime("%Y-%m-%d")
    else:
        # No existing data — full download from 2019
        fetch_start = "2019-01-01"
        log(f"{ticker}: no existing data, full download from {fetch_start}")

    fetch_end = today.strftime("%Y-%m-%d")

    # Fetch new data
    new_df = fetch_new_data(ticker, start=fetch_start, end=fetch_end)

    if new_df is None or len(new_df) == 0:
        result["status"] = "no_new_data"
        log(f"{ticker}: no new data available from Yahoo Finance", "WARNING")
        return result

    # Append to existing CSV
    if os.path.exists(path) and not force:
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    # Deduplicate by date
    date_col = next((c for c in combined.columns if "date" in c.lower()), "Date")
    combined = combined.drop_duplicates(subset=[date_col], keep="last")

    combined.to_csv(path, index=False)
    result["status"]    = "updated"
    result["rows_added"] = len(new_df)
    result["last_date"] = today

    log(f"{ticker}: updated — {len(new_df)} new rows added")
    return result


# ─── Report Generator ─────────────────────────────────────────────────────────

def generate_report(tickers: list) -> None:
    """Print a data status report for all tickers."""
    today = datetime.today().date()
    print("\n" + "═" * 60)
    print("  EWS-CDG — Rapport d'état des données")
    print(f"  Généré le {today.strftime('%d/%m/%Y')}")
    print("═" * 60)

    rows = []
    for ticker in tickers:
        path      = os.path.join(PRICES_DIR, f"{ticker}.csv")
        exists    = os.path.exists(path)
        last_date = get_last_date(ticker) if exists else None
        n_rows    = len(pd.read_csv(path)) if exists else 0

        if not exists:
            status = "❌ Manquant"
        elif last_date and last_date.date() >= today - timedelta(days=3):
            status = "✅ À jour"
        else:
            days_late = (today - last_date.date()).days if last_date else "?"
            status = f"⚠️  Retard ({days_late}j)"

        rows.append({
            "Ticker":       ticker,
            "Société":      get_name(ticker),
            "Dernière date": last_date.strftime("%d/%m/%Y") if last_date else "—",
            "Jours":        n_rows,
            "Statut":       status,
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("═" * 60 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EWS-CDG — Mise à jour des données de marché")
    parser.add_argument("--ticker", type=str, help="Mettre à jour un seul ticker")
    parser.add_argument("--report", action="store_true", help="Afficher le rapport d'état uniquement")
    parser.add_argument("--force",  action="store_true", help="Forcer le re-téléchargement complet")
    args = parser.parse_args()

    all_tickers = TICKERS + [MASI_TICKER]

    # Report only
    if args.report:
        generate_report(all_tickers)
        return

    # Single ticker
    if args.ticker:
        if args.ticker not in all_tickers:
            print(f"❌ Ticker inconnu : {args.ticker}")
            print(f"   Tickers disponibles : {all_tickers}")
            return
        update_ticker(args.ticker, force=args.force)
        generate_report([args.ticker])
        return

    # All tickers
    log("═" * 50)
    log("Début de la mise à jour des données MASI20")
    log("═" * 50)

    results = []
    for ticker in all_tickers:
        r = update_ticker(ticker, force=args.force)
        results.append(r)

    # Summary
    updated    = [r for r in results if r["status"] == "updated"]
    up_to_date = [r for r in results if r["status"] == "up_to_date"]
    failed     = [r for r in results if r["status"] in ("no_new_data", None)]

    log("═" * 50)
    log(f"Mise à jour terminée :")
    log(f"  ✅ Mis à jour  : {len(updated)} ticker(s)")
    log(f"  ⏭️  À jour      : {len(up_to_date)} ticker(s)")
    log(f"  ⚠️  Échec       : {len(failed)} ticker(s)")
    log("═" * 50)

    generate_report(all_tickers)


if __name__ == "__main__":
    main()
