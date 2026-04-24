"""
modules/markowitz.py
────────────────────
Modern Portfolio Theory — Markowitz Optimization.
Computes optimal portfolio weights via mean-variance optimization.

Three objectives available:
  1. Minimize Variance       → Minimum Risk Portfolio
  2. Maximize Sharpe Ratio   → Optimal Risk/Return Portfolio
  3. Target Return           → Efficient Frontier Point

Author  : [Your Name]
Project : Early Warning System — CDG
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional


# ─── Constants ────────────────────────────────────────────────────────────────

TRADING_DAYS    = 252
RISK_FREE_RATE  = 0.03      # 3% — approximation Bons du Trésor marocains
N_SIMULATIONS   = 5_000     # Monte Carlo simulations for frontier


# ─── Core Optimizer ───────────────────────────────────────────────────────────

class MarkowitzOptimizer:
    """
    Mean-Variance Portfolio Optimizer.

    Parameters
    ----------
    returns         : DataFrame of daily returns (Date x Tickers)
    risk_free_rate  : annualized risk-free rate (default 3%)
    max_weight      : maximum weight per asset (default 40%)
    min_weight      : minimum weight per asset (default 1%)
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = RISK_FREE_RATE,
        max_weight: float = 0.40,
        min_weight: float = 0.01,
    ):
        self.returns        = returns
        self.tickers        = list(returns.columns)
        self.n              = len(self.tickers)
        self.risk_free_rate = risk_free_rate
        self.max_weight     = max_weight
        self.min_weight     = min_weight

        # Annualized expected returns and covariance matrix
        self.mu    = returns.mean() * TRADING_DAYS
        self.sigma = returns.cov()  * TRADING_DAYS

    # ── Objective Functions ───────────────────────────────────────────────────

    def _portfolio_return(self, weights: np.ndarray) -> float:
        return float(np.dot(weights, self.mu))

    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        return float(np.sqrt(weights @ self.sigma.values @ weights))

    def _neg_sharpe(self, weights: np.ndarray) -> float:
        ret = self._portfolio_return(weights)
        vol = self._portfolio_volatility(weights)
        return -(ret - self.risk_free_rate) / vol if vol > 0 else 0.0

    # ── Constraints & Bounds ─────────────────────────────────────────────────

    def _get_constraints(self, target_return: Optional[float] = None) -> list:
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        if target_return is not None:
            constraints.append({
                "type": "eq",
                "fun": lambda w: self._portfolio_return(w) - target_return
            })
        return constraints

    def _get_bounds(self) -> list:
        return [(self.min_weight, self.max_weight)] * self.n

    def _initial_weights(self) -> np.ndarray:
        return np.array([1 / self.n] * self.n)

    # ── Optimization Methods ──────────────────────────────────────────────────

    def minimize_variance(self) -> dict:
        """
        Objective 1: Minimum Variance Portfolio.
        Finds weights that minimize total portfolio risk.
        """
        result = minimize(
            fun=self._portfolio_volatility,
            x0=self._initial_weights(),
            method="SLSQP",
            bounds=self._get_bounds(),
            constraints=self._get_constraints(),
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        return self._build_result(result.x, label="Variance Minimale")

    def maximize_sharpe(self) -> dict:
        """
        Objective 2: Maximum Sharpe Ratio Portfolio.
        Best risk-adjusted return — recommended default.
        """
        result = minimize(
            fun=self._neg_sharpe,
            x0=self._initial_weights(),
            method="SLSQP",
            bounds=self._get_bounds(),
            constraints=self._get_constraints(),
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        return self._build_result(result.x, label="Sharpe Maximum")

    def target_return(self, target: float) -> dict:
        """
        Objective 3: Target Return Portfolio.
        Minimum risk portfolio achieving a specific return target.

        Parameters
        ----------
        target : annualized target return (e.g. 0.10 for 10%)
        """
        if target > float(self.mu.max()):
            raise ValueError(
                f"Target return {target:.1%} exceeds maximum achievable return "
                f"{float(self.mu.max()):.1%}."
            )
        result = minimize(
            fun=self._portfolio_volatility,
            x0=self._initial_weights(),
            method="SLSQP",
            bounds=self._get_bounds(),
            constraints=self._get_constraints(target_return=target),
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        return self._build_result(result.x, label=f"Rendement Cible {target:.1%}")

    # ── Result Builder ────────────────────────────────────────────────────────

    def _build_result(self, weights: np.ndarray, label: str) -> dict:
        """Builds a standardized result dictionary from optimized weights."""
        weights = np.clip(weights, 0, 1)
        weights /= weights.sum()   # renormalize for numerical stability

        ret = self._portfolio_return(weights)
        vol = self._portfolio_volatility(weights)
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0.0

        return {
            "label":            label,
            "weights":          dict(zip(self.tickers, weights.round(6))),
            "expected_return":  round(ret, 6),
            "volatility":       round(vol, 6),
            "sharpe_ratio":     round(sharpe, 4),
        }

    # ── Efficient Frontier ────────────────────────────────────────────────────

    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """
        Compute the efficient frontier by solving for minimum variance
        at each target return level.

        Returns DataFrame with columns: [Return, Volatility, Sharpe]
        """
        min_ret = float(self.mu.min())
        max_ret = float(self.mu.max())
        targets = np.linspace(min_ret * 1.05, max_ret * 0.95, n_points)

        frontier = []
        for target in targets:
            try:
                result = self.target_return(target)
                frontier.append({
                    "Rendement":   result["expected_return"],
                    "Volatilité":  result["volatility"],
                    "Sharpe":      result["sharpe_ratio"],
                })
            except Exception:
                continue

        return pd.DataFrame(frontier)

    def monte_carlo_frontier(self) -> pd.DataFrame:
        """
        Monte Carlo simulation of random portfolios.
        Used to visualize the feasible set alongside the efficient frontier.

        Returns DataFrame with columns: [Return, Volatility, Sharpe, Weights...]
        """
        records = []
        for _ in range(N_SIMULATIONS):
            w = np.random.dirichlet(np.ones(self.n))
            ret = self._portfolio_return(w)
            vol = self._portfolio_volatility(w)
            sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0.0
            record = {"Rendement": ret, "Volatilité": vol, "Sharpe": sharpe}
            record.update(dict(zip(self.tickers, w)))
            records.append(record)

        return pd.DataFrame(records)
