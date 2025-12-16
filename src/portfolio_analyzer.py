"""
portfolio_analyzer.py

PortfolioAnalyzer composes multiple PriceSeries objects and computes
portfolio level analytics like correlation and equal weight returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from src.price_series import PriceSeries, DataValidationError


@dataclass(frozen=True)
class PortfolioSummary:
    """Immutable summary object for easy printing/testing."""
    tickers: List[str]
    start_date: str
    end_date: str
    rows: int


class PortfolioAnalyzer:
    """
    Analyzes a group of PriceSeries objects (composition relationship).

    Attributes
    ----------
    series_map : Dict[str, PriceSeries]
        Mapping of ticker -> PriceSeries.
    start_date : Optional[str]
        Optional start date filter (e.g., "2015-01-01") applied to all series.
    """

    def __init__(self, series_list: List[PriceSeries], start_date: Optional[str] = None) -> None:
        if not series_list:
            raise DataValidationError("PortfolioAnalyzer requires at least one PriceSeries.")

        # dict comprehension (Part 2: comprehension)
        self.series_map: Dict[str, PriceSeries] = {ps.symbol: ps for ps in series_list}
        self.start_date = start_date

        if self.start_date is not None:
            self._apply_date_filter(self.start_date)

        self._validate_non_empty()

    def __str__(self) -> str:
        summary = self.summary()
        return f"PortfolioAnalyzer(tickers={summary.tickers}, rows={summary.rows}, range={summary.start_date}..{summary.end_date})"

    def _apply_date_filter(self, start_date: str) -> None:
        """Mutates each PriceSeries to keep rows >= start_date."""
        for ps in self.series_map.values():
            ps.df = ps.df.loc[start_date:]

    def _validate_non_empty(self) -> None:
        """Ensure every series has data after filtering."""
        empty = [t for t, ps in self.series_map.items() if ps.df.empty]
        if empty:
            raise DataValidationError(f"These tickers have no data after filtering: {empty}")

    def tickers(self) -> List[str]:
        """Return tickers in sorted order for consistent output."""
        return sorted(self.series_map.keys())

    def summary(self) -> PortfolioSummary:
        """Return a lightweight summary of the portfolio dataset overlap."""
        # Align by dates across all series
        aligned = self.aligned_close_prices()
        return PortfolioSummary(
            tickers=self.tickers(),
            start_date=str(aligned.index.min().date()),
            end_date=str(aligned.index.max().date()),
            rows=int(len(aligned)),
        )

    def aligned_close_prices(self) -> pd.DataFrame:
        """
        Return a DataFrame of aligned Close prices for all tickers.

        Rows are dates, columns are tickers. Uses inner join to keep only
        dates that exist for all series (clean overlap for fair comparisons).
        """
        closes = {}
        for t, ps in self.series_map.items():
            closes[t] = ps.close_prices()

        df = pd.DataFrame(closes).dropna(how="any")
        if df.empty:
            raise DataValidationError("No overlapping dates across tickers after alignment.")
        return df

    def returns_matrix(self) -> pd.DataFrame:
        """Daily percent returns for each ticker, aligned by date."""
        prices = self.aligned_close_prices()
        rets = prices.pct_change().dropna()
        if rets.empty:
            raise DataValidationError("Returns matrix is empty after pct_change.")
        return rets

    def correlation_matrix(self) -> pd.DataFrame:
        """Correlation of daily returns between tickers."""
        return self.returns_matrix().corr()

    def equal_weight_portfolio_returns(self) -> pd.Series:
        """
        Compute equal-weight portfolio daily returns.

        Uses numpy for vectorized averaging (advanced library use).
        """
        rets = self.returns_matrix()
        weights = np.ones(rets.shape[1]) / rets.shape[1]
        port = rets.values @ weights
        return pd.Series(port, index=rets.index, name="EqualWeight")

    def iter_ticker_stats(self) -> Iterator[tuple[str, float, float]]:
        """
        Generator yielding (ticker, max_drawdown, volatility20) for each series.
        (Part 2: generator function)
        """
        for t, ps in self.series_map.items():
            yield (t, ps.max_drawdown(), ps.volatility(20))