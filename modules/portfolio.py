"""
modules/portfolio.py
────────────────────
Portfolio construction and valuation.
Takes prices + weights → builds daily portfolio value series.

Author  : [Your Name]
Project : Early Warning System — CDG
"""

import pandas as pd
import numpy as np
from typing import Optional


# ─── Portfolio Builder ────────────────────────────────────────────────────────

class Portfolio:
    """
    Represents a weighted equity portfolio built from MASI20 tickers.

    Parameters
    ----------
    prices          : DataFrame of adjusted close prices (Date x Tickers)
    weights         : dict {ticker: weight} — weights must sum to 1.0
    initial_value   : initial portfolio value in MAD (e.g. 100_000_000)
    name            : optional portfolio name for display
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        weights: dict,
        initial_value: float = 100_000_000,
        name: str = "Portefeuille CDG",
    ):
        self.name          = name
        self.initial_value = initial_value
        self.tickers       = list(weights.keys())
        self.weights       = weights

        # Align prices to selected tickers only
        self.prices = prices[self.tickers].copy()

        # Validate
        self._validate_weights()

        # Build
        self.quantities     = self._compute_quantities()
        self.values         = self._compute_values()
        self.total_value    = self.values.sum(axis=1).rename("Valeur_Totale")
        self.returns        = self._compute_returns()
        self.ticker_returns = self._compute_ticker_returns()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_weights(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Weights must sum to 1.0. Current sum: {total:.4f}"
            )
        for ticker, w in self.weights.items():
            if w < 0 or w > 1:
                raise ValueError(f"Weight for {ticker} must be between 0 and 1.")
            if ticker not in self.prices.columns:
                raise ValueError(f"Ticker {ticker} not found in price data.")

    # ── Construction ──────────────────────────────────────────────────────────

    def _compute_quantities(self) -> pd.Series:
        """
        Compute number of shares for each ticker based on initial allocation.
        Quantities are fixed at inception and remain constant throughout.
        """
        first_prices = self.prices.iloc[0]
        quantities = {}
        for ticker, weight in self.weights.items():
            allocated_capital = self.initial_value * weight
            quantities[ticker] = allocated_capital / first_prices[ticker]
        return pd.Series(quantities, name="Quantités")

    def _compute_values(self) -> pd.DataFrame:
        """
        Compute daily market value for each position.
        Value(t) = Quantity × Price(t)
        """
        values = pd.DataFrame(index=self.prices.index)
        for ticker in self.tickers:
            values[f"Val_{ticker}"] = self.quantities[ticker] * self.prices[ticker]
        return values

    def _compute_returns(self) -> pd.Series:
        """
        Compute daily simple returns of the total portfolio value.
        """
        returns = self.total_value.pct_change().dropna()
        returns.name = "Rdt_Portefeuille"
        return returns

    def _compute_ticker_returns(self) -> pd.DataFrame:
        """
        Compute daily simple returns for each individual ticker.
        """
        return self.prices.pct_change().dropna()

    # ── Statistics ────────────────────────────────────────────────────────────

    def annualized_return(self) -> float:
        """Annualized portfolio return."""
        total_return = (self.total_value.iloc[-1] / self.total_value.iloc[0]) - 1
        n_years = len(self.total_value) / 252
        return (1 + total_return) ** (1 / n_years) - 1

    def annualized_volatility(self) -> float:
        """Annualized portfolio volatility (std of daily returns × √252)."""
        return self.returns.std() * np.sqrt(252)

    def sharpe_ratio(self, risk_free_rate: float = 0.03) -> float:
        """
        Sharpe ratio = (annualized return - risk free rate) / annualized volatility.
        Default risk-free rate: 3% (approximation for Moroccan T-bills).
        """
        excess_return = self.annualized_return() - risk_free_rate
        vol = self.annualized_volatility()
        return excess_return / vol if vol != 0 else 0.0

    def max_drawdown(self) -> float:
        """Maximum drawdown of the portfolio over the full period."""
        cumulative = (1 + self.returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()

    def get_summary(self) -> dict:
        """Returns key statistics as a dictionary."""
        return {
            "Rendement annualisé":   f"{self.annualized_return() * 100:.2f}%",
            "Volatilité annualisée": f"{self.annualized_volatility() * 100:.2f}%",
            "Ratio de Sharpe":       f"{self.sharpe_ratio():.2f}",
            "Drawdown maximum":      f"{self.max_drawdown() * 100:.2f}%",
            "Valeur initiale (MAD)": f"{self.initial_value:,.0f}",
            "Valeur finale (MAD)":   f"{self.total_value.iloc[-1]:,.0f}",
            "Période":               f"{self.prices.index[0].strftime('%d/%m/%Y')} → {self.prices.index[-1].strftime('%d/%m/%Y')}",
            "Nombre de titres":      len(self.tickers),
        }

    def export_to_csv(self, path: str) -> None:
        """Export full portfolio data to CSV."""
        export_df = pd.concat([
            self.prices,
            self.values,
            self.total_value,
            self.returns,
        ], axis=1)
        export_df.to_csv(path)
        print(f"[Portfolio] Exported to {path}")


# ─── Weight Validator ─────────────────────────────────────────────────────────

def normalize_weights(weights: dict) -> dict:
    """
    Normalize weights so they sum to exactly 1.0.
    Useful when manual inputs don't sum perfectly.
    """
    total = sum(weights.values())
    if total == 0:
        raise ValueError("Cannot normalize: all weights are zero.")
    return {ticker: w / total for ticker, w in weights.items()}


def equal_weights(tickers: list) -> dict:
    """Returns equal-weight allocation for a list of tickers."""
    n = len(tickers)
    return {ticker: 1 / n for ticker in tickers}
