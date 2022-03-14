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
import numpy as np
from typing import List, Set, Dict, Tuple


def calc_ret(prices: pd.Series, n: int = 1) -> pd.Series:
    """
    Create returns series based on prices
    :param prices: prices
    :param n: number of days ago to calculate returns
    :return: pd.Series
    """
    return prices / prices.shift(n) - 1


def roll_std(prices: pd.Series, n: int = 20) -> pd.Series:
    """
    Create rolling daily returns volatility (std) based on prices
    :param prices: prices
    :param n: number of days ago to specify range for calculating return volatility
    :return: rolling std series
    """

    ret = calc_ret(prices, 1)

    return ret.rolling(n).std()


def delta(ts: pd.Series, n: int = 1) -> pd.Series:
    """
    Computes delta (change) in time series value n days ago
    Formula: value(t) - value(t-n)
    :param ts: time series of values
    :param n: lookback days
    :return: delta value series
    """

    delta = ts - ts.shift(n)

    return delta


def delta_volume(volume: pd.Series, n: int = 1, is_log: bool = True):
    """
    Computes delta (change) in (log) volume n days ago
    Formula: delta(log(volume), 2)
    :param volume: volume time series
    :param n: lookback days
    :param is_log: if True, log volume is used
    :return: delta volume series
    """
    if is_log:
        volume = np.log(volume)

    delta_volume = delta(volume, n=n)

    return delta_volume


def pct_change(new: pd.Series, old: pd.Series):
    """
    Computes percentage change between two Series
    :param new: new (current) series
    :param old: old (previous) series
    :return: percentage changes series
    """
    return new / old - 1


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
