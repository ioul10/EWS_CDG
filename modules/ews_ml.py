"""
modules/ews_ml.py
─────────────────
EWS — Couche Machine Learning
Trois modèles complémentaires :

  1. Isolation Forest  → Détection d'anomalies journalières
  2. PCA               → Détection de stress systémique (corrélations)
  3. K-Means           → Classification des régimes de marché

Scoring ML :
  0 = Normal
  1 = Suspect
  2 = Anomalie confirmée

Score ML Total = moyenne pondérée des 3 signaux → sur 6
  0-2 : 🟢 Normal
  3-4 : 🟡 Suspect
  5-6 : 🔴 Anomalie

Author  : [Your Name]
Project : Early Warning System — CDG
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Optional


# ─── Config ───────────────────────────────────────────────────────────────────

# Isolation Forest
IF_CONTAMINATION  = 0.05     # Expected % of anomalies (~5%)
IF_N_ESTIMATORS   = 200
IF_RANDOM_STATE   = 42

# PCA
PCA_N_COMPONENTS  = 3
PCA_STRESS_THRESHOLD = 0.60  # PC1 variance > 60% → systemic stress
PCA_RECON_PERCENTILE = 95    # Reconstruction error threshold (95th pct)

# K-Means regimes
KMEANS_N_CLUSTERS  = 3
KMEANS_RANDOM_STATE = 42

# Feature engineering window
FEATURE_WINDOW = 20          # Rolling window for feature construction

# Regime labels (assigned after clustering based on volatility)
REGIME_LABELS = {
    "low":    {"label": "🟢 Calme",      "color": "#2e7d32", "score": 0},
    "mid":    {"label": "🟡 Transition", "color": "#f57f17", "score": 1},
    "high":   {"label": "🔴 Stress",     "color": "#c62828", "score": 2},
}


# ─── Feature Engineering ──────────────────────────────────────────────────────

def build_features(
    returns: pd.Series,
    ticker_returns: pd.DataFrame,
    window: int = FEATURE_WINDOW,
) -> pd.DataFrame:
    """
    Build feature matrix for ML models.

    Features per day:
      - Rolling volatility (portfolio)
      - Rolling mean return (portfolio)
      - Rolling max drawdown (portfolio)
      - Rolling skewness (portfolio)
      - Rolling kurtosis (portfolio)
      - Average cross-correlation between tickers
      - PC1 variance ratio (from rolling PCA on ticker returns)

    Parameters
    ----------
    returns        : pd.Series — daily portfolio returns
    ticker_returns : pd.DataFrame — daily returns per ticker
    window         : rolling window size

    Returns
    -------
    features : pd.DataFrame — one row per day, one column per feature
    """
    features = pd.DataFrame(index=returns.index)

    # Portfolio-level features
    features["vol"]      = returns.rolling(window).std() * np.sqrt(252)
    features["mean_rdt"] = returns.rolling(window).mean()
    features["skew"]     = returns.rolling(window).skew()
    features["kurt"]     = returns.rolling(window).kurt()

    # Drawdown feature
    rolling_max = (1 + returns).cumprod().rolling(window).max()
    cum_ret     = (1 + returns).cumprod()
    features["drawdown"] = (cum_ret - rolling_max) / rolling_max

    # Average pairwise correlation between tickers
    def avg_corr(x):
        if x.shape[0] < 5:
            return np.nan
        corr_matrix = x.corr()
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        vals = corr_matrix.where(mask).stack()
        return vals.mean() if len(vals) > 0 else np.nan

    # Rolling average correlation
    avg_corr_series = pd.Series(index=returns.index, dtype=float)
    for i in range(window, len(ticker_returns)):
        window_data = ticker_returns.iloc[i-window:i]
        avg_corr_series.iloc[i] = avg_corr(window_data)
    features["avg_corr"] = avg_corr_series

    return features.dropna()


# ─── 1. Isolation Forest ──────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Isolation Forest — Unsupervised anomaly detection.

    Learns what 'normal' looks like from the feature matrix.
    Flags days that are isolated quickly (few splits needed) as anomalies.

    Anomaly score ∈ [-1, 1] — closer to -1 means more anomalous.
    """

    def __init__(
        self,
        contamination: float = IF_CONTAMINATION,
        n_estimators: int = IF_N_ESTIMATORS,
        random_state: int = IF_RANDOM_STATE,
    ):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model  = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(self, features: pd.DataFrame) -> "AnomalyDetector":
        X = self.scaler.fit_transform(features.values)
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Returns DataFrame with:
          - IF_Score     : raw anomaly score (higher = more normal)
          - IF_Label     : 1=normal, -1=anomaly
          - IF_Anomaly   : bool
          - Score_IF     : EWS score 0/1/2
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self.scaler.transform(features.values)
        scores = self.model.score_samples(X)   # Higher = more normal
        labels = self.model.predict(X)         # 1=normal, -1=anomaly

        # Normalize score to [0,1] — lower = more anomalous
        score_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        # EWS scoring
        ews_score = pd.Series(0, index=features.index)
        ews_score[score_norm < 0.30] = 1   # Suspect
        ews_score[score_norm < 0.15] = 2   # Anomaly confirmed

        return pd.DataFrame({
            "IF_Score":   score_norm.round(4),
            "IF_Label":   labels,
            "IF_Anomaly": labels == -1,
            "Score_IF":   ews_score,
        }, index=features.index)


# ─── 2. PCA Stress Detector ───────────────────────────────────────────────────

class PCAStressDetector:
    """
    PCA — Systemic stress detection.

    Two complementary signals:
      1. PC1 variance ratio : when PC1 explains >60% of total variance,
         all assets move together → systemic stress signal.

      2. Reconstruction error : days with high reconstruction error
         are structurally different from the norm → anomaly signal.

    Both computed on a rolling basis (window days).
    """

    def __init__(
        self,
        n_components: int = PCA_N_COMPONENTS,
        stress_threshold: float = PCA_STRESS_THRESHOLD,
        recon_percentile: float = PCA_RECON_PERCENTILE,
        window: int = 60,
    ):
        self.n_components     = n_components
        self.stress_threshold = stress_threshold
        self.recon_percentile = recon_percentile
        self.window           = window

    def fit_transform(self, ticker_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling PCA signals.

        Returns DataFrame with:
          - PC1_Variance    : share of variance explained by PC1
          - Recon_Error     : reconstruction error of the day
          - PCA_Stress      : bool — PC1 > stress_threshold
          - Score_PCA       : EWS score 0/1/2
        """
        n = len(ticker_returns)
        results = pd.DataFrame(index=ticker_returns.index)
        results["PC1_Variance"] = np.nan
        results["Recon_Error"]  = np.nan

        scaler = StandardScaler()
        pca    = PCA(n_components=self.n_components)

        for i in range(self.window, n):
            window_data = ticker_returns.iloc[i-self.window:i]
            X_scaled    = scaler.fit_transform(window_data.values)

            # Fit PCA on window
            pca.fit(X_scaled)
            pc1_var = pca.explained_variance_ratio_[0]

            # Reconstruction error for current day
            today_scaled  = scaler.transform(ticker_returns.iloc[[i]].values)
            today_pca     = pca.transform(today_scaled)
            today_recon   = pca.inverse_transform(today_pca)
            recon_error   = np.mean((today_scaled - today_recon) ** 2)

            results.iloc[i, results.columns.get_loc("PC1_Variance")] = pc1_var
            results.iloc[i, results.columns.get_loc("Recon_Error")]  = recon_error

        # Stress signal from PC1
        results["PCA_Stress"] = results["PC1_Variance"] > self.stress_threshold

        # Reconstruction error threshold (rolling 95th percentile)
        recon_threshold = results["Recon_Error"].expanding().quantile(
            self.recon_percentile / 100
        )
        results["Recon_High"] = results["Recon_Error"] > recon_threshold

        # EWS scoring
        score = pd.Series(0, index=ticker_returns.index)
        score[results["PCA_Stress"] | results["Recon_High"]] = 1
        score[results["PCA_Stress"] & results["Recon_High"]] = 2
        results["Score_PCA"] = score

        return results.round(6)


# ─── 3. Market Regime Classifier ─────────────────────────────────────────────

class RegimeClassifier:
    """
    K-Means — Market regime classification.

    Clusters market days into 3 regimes based on:
      - Rolling volatility
      - Rolling mean return
      - Rolling drawdown
      - Average cross-correlation

    Regimes are labeled post-hoc by their volatility level:
      Low vol   → 🟢 Calme
      Mid vol   → 🟡 Transition
      High vol  → 🔴 Stress
    """

    def __init__(
        self,
        n_clusters: int = KMEANS_N_CLUSTERS,
        random_state: int = KMEANS_RANDOM_STATE,
    ):
        self.n_clusters   = n_clusters
        self.random_state = random_state
        self.scaler       = StandardScaler()
        self.model        = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=20,
        )
        self.regime_map   = {}   # cluster_id → regime key
        self.is_fitted    = False

    def fit(self, features: pd.DataFrame) -> "RegimeClassifier":
        X = self.scaler.fit_transform(features[["vol", "mean_rdt", "drawdown", "avg_corr"]].values)
        self.model.fit(X)

        # Label clusters by volatility level of centroids
        centers = self.scaler.inverse_transform(self.model.cluster_centers_)
        vol_idx = list(features.columns).index("vol") if "vol" in features.columns else 0
        vol_by_cluster = {i: centers[i][0] for i in range(self.n_clusters)}
        sorted_clusters = sorted(vol_by_cluster, key=vol_by_cluster.get)

        keys = ["low", "mid", "high"]
        self.regime_map = {cluster: key for cluster, key in zip(sorted_clusters, keys)}
        self.is_fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X       = self.scaler.transform(features[["vol", "mean_rdt", "drawdown", "avg_corr"]].values)
        labels  = self.model.predict(X)
        regimes = [self.regime_map.get(l, "mid") for l in labels]

        regime_labels  = [REGIME_LABELS[r]["label"] for r in regimes]
        regime_colors  = [REGIME_LABELS[r]["color"] for r in regimes]
        regime_scores  = [REGIME_LABELS[r]["score"] for r in regimes]

        return pd.DataFrame({
            "Cluster":       labels,
            "Regime":        regime_labels,
            "Regime_Color":  regime_colors,
            "Score_Regime":  regime_scores,
        }, index=features.index)


# ─── EWS ML Engine ───────────────────────────────────────────────────────────

class EWSML:
    """
    Early Warning System — Machine Learning Layer.

    Combines Isolation Forest + PCA + K-Means into a unified ML signal.

    Parameters
    ----------
    returns        : pd.Series — daily portfolio returns
    ticker_returns : pd.DataFrame — daily returns per ticker
    total_value    : pd.Series — portfolio total value (for drawdown)
    feature_window : rolling window for feature engineering
    pca_window     : rolling window for PCA computation
    """

    def __init__(
        self,
        returns: pd.Series,
        ticker_returns: pd.DataFrame,
        total_value: pd.Series,
        feature_window: int = FEATURE_WINDOW,
        pca_window: int = 60,
    ):
        self.returns        = returns
        self.ticker_returns = ticker_returns
        self.total_value    = total_value

        # Step 1 — Build features
        self.features = build_features(returns, ticker_returns, feature_window)

        # Step 2 — Isolation Forest
        self.anomaly_detector = AnomalyDetector()
        self.anomaly_detector.fit(self.features)
        self._if_results = self.anomaly_detector.predict(self.features)

        # Step 3 — PCA
        aligned_tickers = ticker_returns.reindex(self.features.index)
        self.pca_detector = PCAStressDetector(window=pca_window)
        self._pca_results = self.pca_detector.fit_transform(aligned_tickers)

        # Step 4 — K-Means Regimes
        self.regime_classifier = RegimeClassifier()
        self.regime_classifier.fit(self.features)
        self._regime_results = self.regime_classifier.predict(self.features)

        # Step 5 — Combine all signals
        self.results = self._combine()

    def _combine(self) -> pd.DataFrame:
        """Merge all ML results into a single DataFrame."""
        df = pd.concat([
            self.features,
            self._if_results,
            self._pca_results[["PC1_Variance", "Recon_Error", "PCA_Stress", "Score_PCA"]],
            self._regime_results,
        ], axis=1)

        # ML composite score (sum of 3 sub-scores, max = 6)
        score_cols = ["Score_IF", "Score_PCA", "Score_Regime"]
        df["Score_ML"] = df[score_cols].sum(axis=1)

        # ML alert level
        df["Niveau_ML"] = df["Score_ML"].apply(self._get_level)

        return df

    @staticmethod
    def _get_level(score: float) -> str:
        if score <= 2:  return "🟢 Normal"
        if score <= 4:  return "🟡 Suspect"
        return "🔴 Anomalie"

    # ── Public Methods ────────────────────────────────────────────────────────

    def get_current_status(self) -> dict:
        """Latest day ML status."""
        last = self.results.dropna(subset=["Score_ML"]).iloc[-1]
        return {
            "date":           last.name.strftime("%d/%m/%Y"),
            "score_if":       int(last["Score_IF"]),
            "score_pca":      int(last["Score_PCA"]),
            "score_regime":   int(last["Score_Regime"]),
            "score_ml":       int(last["Score_ML"]),
            "niveau_ml":      last["Niveau_ML"],
            "regime":         last["Regime"],
            "pc1_variance":   f"{last['PC1_Variance']*100:.1f}%",
            "if_score":       f"{last['IF_Score']:.3f}",
            "pca_stress":     bool(last["PCA_Stress"]),
        }

    def get_anomalies(self) -> pd.DataFrame:
        """Return days flagged as anomalies (Score_ML >= 3)."""
        return self.results[self.results["Score_ML"] >= 3].copy()

    def get_regime_distribution(self) -> pd.DataFrame:
        """Return regime distribution as a summary table."""
        r = self.results["Regime"].value_counts().reset_index()
        r.columns = ["Régime", "Nombre de jours"]
        r["Pourcentage"] = (r["Nombre de jours"] / len(self.results) * 100).round(1)
        return r

    def summary_stats(self) -> dict:
        r = self.results.dropna(subset=["Score_ML"])
        return {
            "Jours analysés":      len(r),
            "Anomalies IF":        int(self._if_results["IF_Anomaly"].sum()),
            "Jours stress PCA":    int(self._pca_results["PCA_Stress"].sum()),
            "% jours calmes":      f"{(r['Regime'].str.contains('Calme')).mean()*100:.1f}%",
            "% jours transition":  f"{(r['Regime'].str.contains('Transition')).mean()*100:.1f}%",
            "% jours stress":      f"{(r['Regime'].str.contains('Stress')).mean()*100:.1f}%",
            "Score ML max":        int(r["Score_ML"].max()),
        }
