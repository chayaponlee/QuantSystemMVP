"""
A calculator for indicators

We use talib as of now

But why do we need a script for indicators, can't we just use talib directly in backtests?
This is called seperation of concerns.
If talib goes out of production, you can just change the implementation inside here without affecting the other code
For example you can change sma_series to return pd.DataFrame(series).rolling(n).mean()
"""

import talib
import pandas as pd


def calc_ret(prices: pd.Series, n: int = 1) -> pd.Series:
    """
    Create returns series based on prices
    :param prices: prices
    :param n: number of days ago to calculate returns
    :return: pd.Series
    """
    return prices / prices.shift(n) - 1


def calc_roll_std(prices: pd.Series, n: int = 20) -> pd.Series:
    """
    Create rolling daily returns volatility (std) based on prices
    :param prices: prices
    :param n: number of days ago to specify range for calculating return volatility
    :return: pd.Series
    """

    ret = prices / prices.shift(1)

    return ret.rolling(n).std()



# talib functions

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
