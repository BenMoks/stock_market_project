"""
price_series.py

Defines the PriceSeries class used to load, clean, and analyze historical stock prices.
Designed for Yahoo Finance-style CSV files (Date, Open, High, Low, Close, Adj Close, Volume).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import pandas as pd


class DataValidationError(ValueError):
    """Raised when an input dataset is missing required columns or contains invalid values."""


@dataclass(frozen=True)
class PriceRow:
    """
    Immutable row representation for generator-based iteration.

    Using a dataclass here makes each yielded row easy to inspect and test.
    """
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class PriceSeries:
    """
    Represents a single stock's historical price series.

    Attributes
    ----------
    symbol : str
        Stock ticker symbol (e.g., "AAPL").
    df : pd.DataFrame
        Cleaned price data indexed by Date.
    source_path : Optional[Path]
        The CSV file path (if loaded from disk).

    Notes
    -----
    This class intentionally focuses on *one* stock. Multi-stock analysis is handled
    by PortfolioAnalyzer (composition relationship in the overall project).
    """

    REQUIRED_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Volume"}

    def __init__(self, symbol: str, df: pd.DataFrame, source_path: Optional[Union[str, Path]] = None) -> None:
        """
        Construct a PriceSeries from a DataFrame.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        df : pd.DataFrame
            Raw or partially-cleaned dataframe.
        source_path : Optional[Union[str, Path]]
            Original source path if applicable.
        """
        self.symbol = symbol.strip().upper()
        self.source_path = Path(source_path) if source_path is not None else None

        if not self.symbol:
            raise DataValidationError("Symbol cannot be empty.")

        self.df = df.copy()
        self.clean_and_validate()

    def __str__(self) -> str:
        """Friendly summary for printing/logging."""
        start, end = None, None
        if not self.df.empty:
            start = self.df.index.min().date()
            end = self.df.index.max().date()
        return f"PriceSeries(symbol={self.symbol}, rows={len(self.df)}, range={start}..{end})"

    @classmethod
    def from_csv(cls, symbol: str, csv_path: Union[str, Path]) -> "PriceSeries":
        """
        Load a Stooq daily CSV/TXT file into a PriceSeries.

        Stooq files use columns like <DATE>, <OPEN>, ... and date format YYYYMMDD.
        """
        path = Path(csv_path)

        try:
            df = pd.read_csv(path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"CSV file not found: {path}") from e
        except pd.errors.ParserError as e:
            raise ValueError(f"Could not parse CSV (bad format): {path}") from e

        # Stooq -> standard columns
        df = df.rename(columns={
            "<DATE>": "Date",
            "<OPEN>": "Open",
            "<HIGH>": "High",
            "<LOW>": "Low",
            "<CLOSE>": "Close",
            "<VOL>": "Volume",
        })

        # Convert YYYYMMDD -> datetime (errors coerced just in case)
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")

        # Keep only the columns your validator expects
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        return cls(symbol=symbol, df=df, source_path=path)



    def clean_and_validate(self) -> None:
        """
        Clean the dataset:
        - Ensure required columns exist
        - Parse dates
        - Convert numeric columns
        - Drop missing values
        - Sort by Date and set Date index
        - Remove duplicate dates (keep last)
        """
        missing = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            # Exception Handling Approach #2:
            # Explicitly raise a validation exception when required data is missing.
            raise DataValidationError(f"{self.symbol}: missing required columns: {sorted(missing)}")

        # Parse Date
        self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")

        # Convert numeric columns safely
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:  # for-loop requirement satisfied here
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Drop rows with invalid date or required numeric values
        self.df = self.df.dropna(subset=["Date"] + numeric_cols)

        # Set index to Date for time series operations
        self.df = self.df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").set_index("Date")

        # Basic sanity check: prices should be > 0
        if (self.df["Close"] <= 0).any():
            raise DataValidationError(f"{self.symbol}: Close contains non-positive values.")

    def iter_rows(self) -> Iterator[PriceRow]:
        """
        Generator that yields rows one-by-one as immutable PriceRow objects.
        (Part 2: generator function)
        """
        for idx, row in self.df.iterrows():
            yield PriceRow(
                date=idx,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            )

    def close_prices(self) -> pd.Series:
        """Return the Close price series."""
        return self.df["Close"].copy()

    def daily_returns(self) -> pd.Series:
        """
        Compute daily percent returns from Close prices.
        """
        closes = self.close_prices()
        return closes.pct_change().dropna()

    def cumulative_returns(self) -> pd.Series:
        """
        Compute cumulative returns from daily returns.
        """
        r = self.daily_returns()
        return (1 + r).cumprod() - 1

    def moving_average(self, window: int = 20) -> pd.Series:
        """
        Compute moving average of Close prices.

        Parameters
        ----------
        window : int
            Rolling window size in days.
        """
        if window <= 0:
            raise ValueError("window must be a positive integer.")
        return self.close_prices().rolling(window=window).mean()

    def volatility(self, window: int = 20) -> float:
        """
        Compute rolling volatility (std dev) of daily returns, then return latest value.
        """
        if self.df.empty:
            return float("nan")
        r = self.daily_returns()
        vol = r.rolling(window=window).std()
        return float(vol.dropna().iloc[-1]) if not vol.dropna().empty else float("nan")

    def max_drawdown(self) -> float:
        """
        Compute max drawdown from Close prices.

        Drawdown = (price / rolling_max) - 1
        Most negative drawdown is returned.
        """
        closes = self.close_prices()
        rolling_max = closes.cummax()
        drawdown = (closes / rolling_max) - 1.0
        return float(drawdown.min())

    def rsi(self, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI) for this price series.
        
        Uses the indicators module to compute RSI on Close prices.
        
        Parameters
        ----------
        window : int, default 14
            Number of periods for RSI calculation.
        
        Returns
        -------
        pd.Series
            RSI values indexed by date.
        """
        from src.indicators import rsi as calc_rsi
        return calc_rsi(self.close_prices(), window=window)

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence) for this price series.
        
        Uses the indicators module to compute MACD on Close prices.
        
        Parameters
        ----------
        fast : int, default 12
            Period for fast EMA.
        slow : int, default 26
            Period for slow EMA.
        signal : int, default 9
            Period for signal line EMA.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 'MACD', 'Signal', 'Histogram'
        """
        from src.indicators import macd as calc_macd
        return calc_macd(self.close_prices(), fast=fast, slow=slow, signal=signal)

    def bollinger_bands(self, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands for this price series.
        
        Uses the indicators module to compute Bollinger Bands on Close prices.
        
        Parameters
        ----------
        window : int, default 20
            Period for moving average.
        num_std : float, default 2.0
            Number of standard deviations for bands.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 'Upper', 'Middle', 'Lower'
        """
        from src.indicators import bollinger_bands as calc_bb
        return calc_bb(self.close_prices(), window=window, num_std=num_std)

    def __add__(self, other: "PriceSeries") -> "PriceSeries":
        """
        Operator overloading: merge two PriceSeries objects into one.

        Rules:
        - Symbols are combined as "AAA+BBB"
        - DataFrames are concatenated
        - Clean/validate again to ensure consistent index and remove duplicates
        """
        if not isinstance(other, PriceSeries):
            return NotImplemented

        merged_symbol = f"{self.symbol}+{other.symbol}"
        merged_df = pd.concat([self.df.reset_index(), other.df.reset_index()], ignore_index=True)
        return PriceSeries(symbol=merged_symbol, df=merged_df)
