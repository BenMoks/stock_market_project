"""
indicators.py

Technical analysis indicators for stock price series.
Provides RSI, MACD, and Bollinger Bands calculations.
"""

from typing import Optional
import pandas as pd
import numpy as np


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI measures momentum on a scale of 0-100.
    Values above 70 typically indicate overbought conditions,
    while values below 30 indicate oversold conditions.
    
    Parameters
    ----------
    prices : pd.Series
        Price series (typically Close prices).
    window : int, default 14
        Number of periods for RSI calculation.
    
    Returns
    -------
    pd.Series
        RSI values indexed by date.
    """
    if window <= 0:
        raise ValueError("window must be a positive integer.")
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    MACD is a trend-following momentum indicator that shows
    the relationship between two moving averages of prices.
    
    Parameters
    ----------
    prices : pd.Series
        Price series (typically Close prices).
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
    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError("All periods must be positive integers.")
    if fast >= slow:
        raise ValueError("fast period must be less than slow period.")
    
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram
    })


def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Bollinger Bands consist of a middle band (SMA) and two outer bands
    that are standard deviations away from the middle band.
    
    Parameters
    ----------
    prices : pd.Series
        Price series (typically Close prices).
    window : int, default 20
        Period for moving average.
    num_std : float, default 2.0
        Number of standard deviations for bands.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'Upper', 'Middle', 'Lower'
    """
    if window <= 0:
        raise ValueError("window must be a positive integer.")
    if num_std <= 0:
        raise ValueError("num_std must be positive.")
    
    middle = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return pd.DataFrame({
        'Upper': upper,
        'Middle': middle,
        'Lower': lower
    })
