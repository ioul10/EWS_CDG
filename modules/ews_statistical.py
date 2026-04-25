"""
modules/ews_statistical.py
──────────────────────────
EWS — Couche Statistique
Calcule les 4 indicateurs de risque rolling et génère les scores d'alerte.

Indicateurs :
  1. Volatilité rolling (fenêtre 30j)
  2. Z-score du rendement journalier
  3. Drawdown depuis le dernier pic
  4. VaR historique rolling (99%, fenêtre 252j)

Scoring :
  0 = Normal    (vert)
  1 = Vigilance (orange)
  2 = Alerte    (rouge)

Score total = somme des 4 scores → sur 8
  0-2  : 🟢 Normal
  3-5  : 🟡 Vigilance
  6-8  : 🔴 Critique

Author  : [Your Name]
Project : Early Warning System — CDG
"""

import numpy as np
import pandas as pd
from typing import Optional


# ─── Config ───────────────────────────────────────────────────────────────────

TRADING_DAYS = 252

# Default windows
VOL_WINDOW  = 30    # Rolling volatility window (days)
VAR_WINDOW  = 252   # Rolling VaR window (days)
ZSCORE_WINDOW = 60  # Rolling mean/std for Z-score

# Default alert thresholds
VOL_THRESHOLDS    = (1.5, 2.0)   # (vigilance, alerte) × historical mean
ZSCORE_THRESHOLDS = (2.0, 3.0)   # |Z| thresholds
DRAWDOWN_THRESHOLDS = (-0.05, -0.10)  # -5% vigilance, -10% alerte
VAR_THRESHOLDS    = (1.5, 2.0)   # × historical VaR mean

# Alert levels
LEVEL_NORMAL    = 0
LEVEL_VIGILANCE = 1
LEVEL_ALERT     = 2

SCORE_LABELS = {
    (0, 2):  ("🟢 Normal",    "normal",    "#2e7d32"),
    (3, 5):  ("🟡 Vigilance", "vigilance", "#f57f17"),
    (6, 8):  ("🔴 Critique",  "critique",  "#c62828"),
}


# ─── Individual Indicators ────────────────────────────────────────────────────

def compute_volatility(
    returns: pd.Series,
    window: int = VOL_WINDOW,
) -> pd.DataFrame:
    """
    Rolling annualized volatility.

    Vol(t) = std(returns[t-window:t]) × √252
    Score  = 0 if Vol < 1.5× mean | 1 if < 2× mean | 2 otherwise
    """
    vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    vol_mean = vol.expanding(min_periods=window).mean()

    ratio = vol / vol_mean

    score = pd.Series(LEVEL_NORMAL, index=returns.index, name="Score_Vol")
    score[ratio >= VOL_THRESHOLDS[0]] = LEVEL_VIGILANCE
    score[ratio >= VOL_THRESHOLDS[1]] = LEVEL_ALERT

    return pd.DataFrame({
        "Vol_30j":    vol.round(6),
        "Vol_Mean":   vol_mean.round(6),
        "Vol_Ratio":  ratio.round(4),
        "Score_Vol":  score,
    })


def compute_zscore(
    returns: pd.Series,
    window: int = ZSCORE_WINDOW,
) -> pd.DataFrame:
    """
    Rolling Z-score of daily returns.

    Z(t) = (r(t) - μ_rolling) / σ_rolling
    Score = 0 if |Z| < 2 | 1 if < 3 | 2 if ≥ 3
    """
    mu    = returns.rolling(window).mean()
    sigma = returns.rolling(window).std()

    zscore = (returns - mu) / sigma
    abs_z  = zscore.abs()

    score = pd.Series(LEVEL_NORMAL, index=returns.index, name="Score_Z")
    score[abs_z >= ZSCORE_THRESHOLDS[0]] = LEVEL_VIGILANCE
    score[abs_z >= ZSCORE_THRESHOLDS[1]] = LEVEL_ALERT

    return pd.DataFrame({
        "Z_Score":   zscore.round(4),
        "Z_Abs":     abs_z.round(4),
        "Score_Z":   score,
    })


def compute_drawdown(total_value: pd.Series) -> pd.DataFrame:
    """
    Drawdown from rolling peak portfolio value.

    DD(t) = (V(t) - max(V[0:t])) / max(V[0:t])
    Score  = 0 if DD > -5% | 1 if > -10% | 2 if ≤ -10%
    """
    rolling_peak = total_value.cummax()
    drawdown     = (total_value - rolling_peak) / rolling_peak

    score = pd.Series(LEVEL_NORMAL, index=total_value.index, name="Score_DD")
    score[drawdown <= DRAWDOWN_THRESHOLDS[0]] = LEVEL_VIGILANCE
    score[drawdown <= DRAWDOWN_THRESHOLDS[1]] = LEVEL_ALERT

    return pd.DataFrame({
        "Drawdown":    drawdown.round(6),
        "Peak_Value":  rolling_peak.round(2),
        "Score_DD":    score,
    })


def compute_var(
    returns: pd.Series,
    window: int = VAR_WINDOW,
    confidence: float = 0.99,
) -> pd.DataFrame:
    """
    Rolling historical VaR at given confidence level.

    VaR(t) = percentile(returns[t-window:t], 1-confidence)
    Score   = 0 if |VaR| < 1.5× mean | 1 if < 2× mean | 2 otherwise

    Note: VaR is expressed as a negative number (loss).
    """
    var = returns.rolling(window).quantile(1 - confidence)
    var_mean = var.expanding(min_periods=window).mean()

    # Compare absolute values (both negative)
    ratio = var.abs() / var_mean.abs()
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    score = pd.Series(LEVEL_NORMAL, index=returns.index, name="Score_VaR")
    score[ratio >= VAR_THRESHOLDS[0]] = LEVEL_VIGILANCE
    score[ratio >= VAR_THRESHOLDS[1]] = LEVEL_ALERT

    return pd.DataFrame({
        "VaR_99":      var.round(6),
        "VaR_Mean":    var_mean.round(6),
        "VaR_Ratio":   ratio.round(4),
        "Score_VaR":   score,
    })


# ─── EWS Statistical Engine ───────────────────────────────────────────────────

class EWSStatistical:
    """
    Early Warning System — Statistical Layer.

    Takes portfolio returns and total value series.
    Computes all 4 indicators and generates a composite alert score.

    Parameters
    ----------
    returns      : pd.Series — daily portfolio returns
    total_value  : pd.Series — daily portfolio total value (MAD)
    vol_window   : rolling window for volatility (default 30)
    var_window   : rolling window for VaR (default 252)
    zscore_window: rolling window for Z-score mean/std (default 60)
    """

    def __init__(
        self,
        returns: pd.Series,
        total_value: pd.Series,
        vol_window: int = VOL_WINDOW,
        var_window: int = VAR_WINDOW,
        zscore_window: int = ZSCORE_WINDOW,
    ):
        self.returns       = returns
        self.total_value   = total_value
        self.vol_window    = vol_window
        self.var_window    = var_window
        self.zscore_window = zscore_window

        # Compute all indicators
        self._vol_df  = compute_volatility(returns, vol_window)
        self._z_df    = compute_zscore(returns, zscore_window)
        self._dd_df   = compute_drawdown(total_value)
        self._var_df  = compute_var(returns, var_window)

        # Build full results table
        self.results = self._build_results()

    def _build_results(self) -> pd.DataFrame:
        """Combine all indicators into a single results DataFrame."""
        df = pd.concat([
            self._vol_df,
            self._z_df,
            self._dd_df,
            self._var_df,
            self.returns.rename("Rendement"),
            self.total_value.rename("Valeur_Totale"),
        ], axis=1)

        # Composite score
        score_cols = ["Score_Vol", "Score_Z", "Score_DD", "Score_VaR"]
        df["Score_Total"] = df[score_cols].sum(axis=1)

        # Alert level
        df["Niveau_Alerte"] = df["Score_Total"].apply(self._get_level_label)
        df["Couleur_Alerte"] = df["Score_Total"].apply(self._get_level_color)

        return df

    @staticmethod
    def _get_level_label(score: float) -> str:
        if score <= 2:  return "🟢 Normal"
        if score <= 5:  return "🟡 Vigilance"
        return "🔴 Critique"

    @staticmethod
    def _get_level_color(score: float) -> str:
        if score <= 2:  return "#2e7d32"
        if score <= 5:  return "#f57f17"
        return "#c62828"

    # ── Public Methods ────────────────────────────────────────────────────────

    def get_alerts(self, min_score: int = 3) -> pd.DataFrame:
        """Return only rows where Score_Total >= min_score."""
        return self.results[self.results["Score_Total"] >= min_score].copy()

    def get_current_status(self) -> dict:
        """Return the latest day's full status."""
        last = self.results.dropna(subset=["Score_Total"]).iloc[-1]
        return {
            "date":           last.name.strftime("%d/%m/%Y"),
            "rendement":      f"{last['Rendement']*100:.2f}%",
            "volatilite":     f"{last['Vol_30j']*100:.2f}%",
            "z_score":        f"{last['Z_Score']:.2f}",
            "drawdown":       f"{last['Drawdown']*100:.2f}%",
            "var_99":         f"{last['VaR_99']*100:.2f}%",
            "score_total":    int(last["Score_Total"]),
            "niveau_alerte":  last["Niveau_Alerte"],
            "score_vol":      int(last["Score_Vol"]),
            "score_z":        int(last["Score_Z"]),
            "score_dd":       int(last["Score_DD"]),
            "score_var":      int(last["Score_VaR"]),
        }

    def get_stress_periods(self, min_score: int = 6) -> pd.DataFrame:
        """
        Identify contiguous stress periods (Score_Total >= min_score).
        Returns DataFrame with start, end, duration, max score.
        """
        alerts = self.results["Score_Total"] >= min_score
        periods = []
        in_period = False
        start = None

        for date, is_alert in alerts.items():
            if is_alert and not in_period:
                in_period = True
                start = date
            elif not is_alert and in_period:
                in_period = False
                period_data = self.results.loc[start:date]
                periods.append({
                    "Début":        start.strftime("%d/%m/%Y"),
                    "Fin":          date.strftime("%d/%m/%Y"),
                    "Durée (jours)": len(period_data),
                    "Score max":    int(period_data["Score_Total"].max()),
                    "DD max":       f"{period_data['Drawdown'].min()*100:.2f}%",
                    "Vol max":      f"{period_data['Vol_30j'].max()*100:.2f}%",
                })

        if in_period:
            period_data = self.results.loc[start:]
            periods.append({
                "Début":        start.strftime("%d/%m/%Y"),
                "Fin":          "En cours",
                "Durée (jours)": len(period_data),
                "Score max":    int(period_data["Score_Total"].max()),
                "DD max":       f"{period_data['Drawdown'].min()*100:.2f}%",
                "Vol max":      f"{period_data['Vol_30j'].max()*100:.2f}%",
            })

        return pd.DataFrame(periods) if periods else pd.DataFrame()

    def summary_stats(self) -> dict:
        """Return summary statistics over the full period."""
        r = self.results.dropna(subset=["Score_Total"])
        total = len(r)
        return {
            "Jours analysés":      total,
            "Jours normaux":       int((r["Score_Total"] <= 2).sum()),
            "Jours vigilance":     int(((r["Score_Total"] >= 3) & (r["Score_Total"] <= 5)).sum()),
            "Jours critiques":     int((r["Score_Total"] >= 6).sum()),
            "% jours critiques":   f"{(r['Score_Total'] >= 6).mean()*100:.1f}%",
            "Drawdown max":        f"{r['Drawdown'].min()*100:.2f}%",
            "Vol max":             f"{r['Vol_30j'].max()*100:.2f}%",
            "Z-score min":         f"{r['Z_Score'].min():.2f}",
            "VaR 99% max":         f"{r['VaR_99'].min()*100:.2f}%",
        }
