"""
modules/ews_engine.py
─────────────────────
EWS — Moteur Central
Combine la couche statistique et la couche ML en un score global unique.

Architecture du score :
  Couche Stat  : Vol + Z-score + Drawdown + VaR  → sur 8
  Couche ML    : Isolation Forest + PCA + Régimes → sur 6
  ─────────────────────────────────────────────────────
  Score Global : Stat (pondéré 60%) + ML (40%)   → sur 10

Niveaux d'alerte globaux :
  0 – 3  : 🟢 Normal    — surveillance standard
  4 – 6  : 🟡 Vigilance — surveillance renforcée
  7 – 10 : 🔴 Critique  — déclenchement hedging

Author  : [Your Name]
Project : Early Warning System — CDG
"""

import numpy as np
import pandas as pd
from typing import Optional

from modules.ews_statistical import EWSStatistical
from modules.ews_ml          import EWSML


# ─── Config ───────────────────────────────────────────────────────────────────

WEIGHT_STAT = 0.60      # Weight of statistical layer in global score
WEIGHT_ML   = 0.40      # Weight of ML layer in global score

SCALE_STAT  = 8         # Max score of statistical layer
SCALE_ML    = 6         # Max score of ML layer
SCALE_GLOBAL = 10       # Global score scale

# Alert thresholds (on global score /10)
THRESHOLD_VIGILANCE = 4
THRESHOLD_CRITIQUE  = 7

ALERT_CONFIG = {
    "normal":    {"label": "🟢 Normal",    "color": "#2e7d32", "action": "Surveillance standard"},
    "vigilance": {"label": "🟡 Vigilance", "color": "#f57f17", "action": "Surveillance renforcée — préparer le hedge"},
    "critique":  {"label": "🔴 Critique",  "color": "#c62828", "action": "Déclencher la couverture via Futures MASI20"},
}


# ─── EWS Engine ───────────────────────────────────────────────────────────────

class EWSEngine:
    """
    Moteur central de l'Early Warning System.

    Orchestre la couche statistique et la couche ML,
    les combine en un score global, et produit les alertes finales.

    Parameters
    ----------
    portfolio       : Portfolio object (from modules.portfolio)
    vol_window      : rolling window for volatility (default 30)
    var_window      : rolling window for VaR (default 252)
    zscore_window   : rolling window for Z-score (default 60)
    feature_window  : rolling window for ML features (default 20)
    pca_window      : rolling window for PCA (default 60)
    weight_stat     : weight of statistical layer (default 0.60)
    weight_ml       : weight of ML layer (default 0.40)
    """

    def __init__(
        self,
        portfolio,
        vol_window: int = 30,
        var_window: int = 252,
        zscore_window: int = 60,
        feature_window: int = 20,
        pca_window: int = 60,
        weight_stat: float = WEIGHT_STAT,
        weight_ml: float = WEIGHT_ML,
    ):
        self.portfolio    = portfolio
        self.weight_stat  = weight_stat
        self.weight_ml    = weight_ml

        # ── Layer 1: Statistical ──────────────────────────────────────────────
        self.stat_layer = EWSStatistical(
            returns=portfolio.returns,
            total_value=portfolio.total_value,
            vol_window=vol_window,
            var_window=var_window,
            zscore_window=zscore_window,
        )

        # ── Layer 2: ML ───────────────────────────────────────────────────────
        self.ml_layer = EWSML(
            returns=portfolio.returns,
            ticker_returns=portfolio.ticker_returns,
            total_value=portfolio.total_value,
            feature_window=feature_window,
            pca_window=pca_window,
        )

        # ── Combine ───────────────────────────────────────────────────────────
        self.results = self._combine()

    # ── Score Combination ─────────────────────────────────────────────────────

    def _combine(self) -> pd.DataFrame:
        """
        Merge statistical and ML results into a unified DataFrame.

        Global score formula:
          score_norm_stat = Score_Total_Stat / 8   ∈ [0,1]
          score_norm_ml   = Score_ML / 6           ∈ [0,1]
          score_global    = (w_stat × norm_stat + w_ml × norm_ml) × 10
        """
        stat = self.stat_layer.results[[
            "Rendement", "Valeur_Totale",
            "Vol_30j", "Z_Score", "Drawdown", "VaR_99",
            "Score_Vol", "Score_Z", "Score_DD", "Score_VaR",
            "Score_Total", "Niveau_Alerte",
        ]].copy()
        stat.columns = [
            "Rendement", "Valeur_Totale",
            "Vol_30j", "Z_Score", "Drawdown", "VaR_99",
            "Score_Vol", "Score_Z", "Score_DD", "Score_VaR",
            "Score_Stat", "Niveau_Stat",
        ]

        ml = self.ml_layer.results[[
            "IF_Score", "IF_Anomaly",
            "PC1_Variance", "PCA_Stress",
            "Regime", "Regime_Color",
            "Score_IF", "Score_PCA", "Score_Regime",
            "Score_ML", "Niveau_ML",
        ]].copy()

        # Align on common index
        df = stat.join(ml, how="left")

        # Normalize both scores to [0,1]
        df["Norm_Stat"] = df["Score_Stat"].fillna(0) / SCALE_STAT
        df["Norm_ML"]   = df["Score_ML"].fillna(0)   / SCALE_ML

        # Weighted global score on [0,10]
        df["Score_Global"] = (
            self.weight_stat * df["Norm_Stat"] +
            self.weight_ml   * df["Norm_ML"]
        ) * SCALE_GLOBAL

        df["Score_Global"] = df["Score_Global"].round(2)

        # Alert level
        df["Niveau_Global"] = df["Score_Global"].apply(self._get_level)
        df["Couleur_Global"] = df["Score_Global"].apply(self._get_color)
        df["Action"]         = df["Score_Global"].apply(self._get_action)

        # Hedge signal — boolean flag
        df["Signal_Hedge"] = df["Score_Global"] >= THRESHOLD_CRITIQUE

        return df

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _get_level(score: float) -> str:
        if pd.isna(score) or score < THRESHOLD_VIGILANCE:
            return "🟢 Normal"
        if score < THRESHOLD_CRITIQUE:
            return "🟡 Vigilance"
        return "🔴 Critique"

    @staticmethod
    def _get_color(score: float) -> str:
        if pd.isna(score) or score < THRESHOLD_VIGILANCE:
            return "#2e7d32"
        if score < THRESHOLD_CRITIQUE:
            return "#f57f17"
        return "#c62828"

    @staticmethod
    def _get_action(score: float) -> str:
        if pd.isna(score) or score < THRESHOLD_VIGILANCE:
            return ALERT_CONFIG["normal"]["action"]
        if score < THRESHOLD_CRITIQUE:
            return ALERT_CONFIG["vigilance"]["action"]
        return ALERT_CONFIG["critique"]["action"]

    # ── Public Methods ────────────────────────────────────────────────────────

    def get_current_status(self) -> dict:
        """Full status for the latest trading day."""
        last = self.results.dropna(subset=["Score_Global"]).iloc[-1]
        return {
            "date":           last.name.strftime("%d/%m/%Y"),
            "score_stat":     round(float(last["Score_Stat"]), 1),
            "score_ml":       round(float(last["Score_ML"]) if not pd.isna(last["Score_ML"]) else 0, 1),
            "score_global":   round(float(last["Score_Global"]), 2),
            "niveau_global":  last["Niveau_Global"],
            "action":         last["Action"],
            "signal_hedge":   bool(last["Signal_Hedge"]),
            "regime":         last.get("Regime", "N/A"),
            "drawdown":       f"{last['Drawdown']*100:.2f}%",
            "volatilite":     f"{last['Vol_30j']*100:.2f}%",
            "var_99":         f"{last['VaR_99']*100:.2f}%",
            "z_score":        f"{last['Z_Score']:.2f}",
            "pc1_variance":   f"{last['PC1_Variance']*100:.1f}%" if not pd.isna(last.get("PC1_Variance", np.nan)) else "N/A",
        }

    def get_hedge_signals(self) -> pd.DataFrame:
        """Return days where hedge signal was triggered (Score_Global >= 7)."""
        hedges = self.results[self.results["Signal_Hedge"]].copy()
        if hedges.empty:
            return pd.DataFrame()
        return hedges[[
            "Rendement", "Score_Stat", "Score_ML",
            "Score_Global", "Niveau_Global", "Action",
            "Drawdown", "Vol_30j", "Regime",
        ]].copy()

    def get_alert_history(self, min_score: float = THRESHOLD_VIGILANCE) -> pd.DataFrame:
        """Return all days with Score_Global >= min_score."""
        return self.results[
            self.results["Score_Global"] >= min_score
        ].copy()

    def get_stress_periods(self, min_score: float = THRESHOLD_CRITIQUE) -> pd.DataFrame:
        """Identify contiguous stress periods above threshold."""
        alerts = self.results["Score_Global"] >= min_score
        periods, in_period, start = [], False, None

        for dt, is_alert in alerts.items():
            if is_alert and not in_period:
                in_period, start = True, dt
            elif not is_alert and in_period:
                in_period = False
                slice_ = self.results.loc[start:dt]
                periods.append({
                    "Début":          start.strftime("%d/%m/%Y"),
                    "Fin":            dt.strftime("%d/%m/%Y"),
                    "Durée (jours)":  len(slice_),
                    "Score max":      round(float(slice_["Score_Global"].max()), 2),
                    "Drawdown max":   f"{slice_['Drawdown'].min()*100:.2f}%",
                    "Régime dominant":slice_["Regime"].mode()[0] if "Regime" in slice_ else "N/A",
                    "Signal hedge":   "✅ Oui" if slice_["Signal_Hedge"].any() else "❌ Non",
                })

        if in_period and start:
            slice_ = self.results.loc[start:]
            periods.append({
                "Début":          start.strftime("%d/%m/%Y"),
                "Fin":            "En cours",
                "Durée (jours)":  len(slice_),
                "Score max":      round(float(slice_["Score_Global"].max()), 2),
                "Drawdown max":   f"{slice_['Drawdown'].min()*100:.2f}%",
                "Régime dominant":slice_["Regime"].mode()[0] if "Regime" in slice_ else "N/A",
                "Signal hedge":   "✅ Oui" if slice_["Signal_Hedge"].any() else "❌ Non",
            })

        return pd.DataFrame(periods) if periods else pd.DataFrame()

    def summary_stats(self) -> dict:
        """Global EWS statistics over the full period."""
        r = self.results.dropna(subset=["Score_Global"])
        n = len(r)
        return {
            "Jours analysés":       n,
            "Jours normaux":        int((r["Score_Global"] < THRESHOLD_VIGILANCE).sum()),
            "Jours vigilance":      int(((r["Score_Global"] >= THRESHOLD_VIGILANCE) & (r["Score_Global"] < THRESHOLD_CRITIQUE)).sum()),
            "Jours critiques":      int((r["Score_Global"] >= THRESHOLD_CRITIQUE).sum()),
            "% jours critiques":    f"{(r['Score_Global'] >= THRESHOLD_CRITIQUE).mean()*100:.1f}%",
            "Signaux hedge":        int(r["Signal_Hedge"].sum()),
            "Score global moyen":   f"{r['Score_Global'].mean():.2f}",
            "Score global max":     f"{r['Score_Global'].max():.2f}",
            "Drawdown max":         f"{r['Drawdown'].min()*100:.2f}%",
        }
