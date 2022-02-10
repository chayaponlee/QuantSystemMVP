"""
A calculator for indicators

We use talib as of now

But why do we need a script for indicators, can't we just use talib directly in backtests?
This is called seperation of concerns.
If talib goes out of production, you can just change the implementation inside here without affecting the other code
For example you can change sma_series to return pd.DataFrame(series).rolling(n).mean()
"""

import talib


def adx_series(high, low, close, n):
    """
    Creates a series for Average Direction Index series from lookback period

    :param high: pd.Series, array
    :param low: pd.Series, array
    :param close: pd.Series, array
    :param n: int (lookback period)
    :return: pd.Series, array
    """
    return talib.ADX(high, low, close, timeperiod=n)


def ema_series(series, n):
    """
    Creates an Exponential Moving Average series from lookback period

    :param series: pd.Series, array
    :param n: int (lookback period)
    :return: pd.Series, array
    """
    return talib.EMA(series, timeperiod=n)


def sma_series(series, n):
    """
    Creates a Simple Moving Average series from lookback period

    :param series: pd.Series, array
    :param n: int (lookback period)
    :return: pd.Series, array
    """
    return talib.SMA(series, timeperiod=n)
