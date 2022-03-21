import pandas as pd
import numpy as np
from typing import Union, List
from realgam.quantlib.engineer.interface import BaseEngineer, GroupBaseEngineer
import talib


class TalibEngineer(BaseEngineer):
    """
    Class for Engineering Technical Analysis indicators based on talib, contains methods and operations to create
    techincal indicators

    """

    def __init__(self, financial_df: pd.DataFrame):
        # we are actually sorting our dataframe reference, if we wish to not manipulate out input, do copy instead
        financial_df.sort_values('date', inplace=True)
        super().__init__(financial_df)

    @property
    def df(self):
        return super().df()

    def set_df(self, financial_df: pd.DataFrame):
        # we are actually sorting our dataframe reference, if we wish to not manipulate out input, do copy instead
        financial_df.sort_values('date', inplace=True)
        super().set_df(financial_df)

    def adx(self, n: int, adjust: bool = True, inplace: bool = False):
        """
        Creates a series for Average Direction Index series from lookback period

        :param high: pd.Series, array
        :param low: pd.Series, array
        :param close: pd.Series, array
        :param n: int (lookback period)
        :return: pd.Series, array
        """

        if adjust:
            eng_values = talib.ADX(self.df.highadj, self.df.lowadj, self.df.closeadj, timeperiod=n)

        else:
            eng_values = talib.ADX(self.df.high, self.df.low, self.df.close, timeperiod=n)

        if inplace:
            self.df[f'adx_{n}'] = eng_values.values
        else:
            return eng_values

    def ema(self, col: str, n: int, inplace: bool = True):
        """
        Creates an Exponential Moving Average series from lookback period

        :param series: pd.Series, array
        :param n: int (lookback period)
        :return: pd.Series, array
        """
        eng_values = talib.EMA(self.df[col], timeperiod=n)
        if inplace:
            self.df[f'ema_{col}_{n}'] = eng_values.values
        else:
            return eng_values

    def ma(self, col: str, n: int, inplace: bool = True):
        """
        Creates a Simple Moving Average series from lookback period

        :param series: pd.Series, array
        :param n: int (lookback period)
        :return: pd.Series, array
        """
        eng_values = talib.MA(self.df[col], timeperiod=n)
        if inplace:
            self.df[f'ma_{col}_{n}'] = eng_values.values
        else:
            return eng_values


class GroupTalibEngineer(GroupBaseEngineer):
    """
    Class for Engineering Technical Analysis indicators based on talib, contains methods and operations to create
    techincal indicators

    """

    def __init__(self, financial_df: pd.DataFrame, groupby_col: Union[str, List]):
        if isinstance(groupby_col, str):
            sort_cols = [groupby_col, 'date']
        else:
            sort_cols = groupby_col.append('date')
        # we are actually sorting our dataframe reference, if we wish to not manipulate out input, do copy instead
        financial_df.sort_values(sort_cols, inplace=True)
        super().__init__(financial_df, groupby_col)

    @property
    def df(self):
        return super().df()

    @property
    def groupby_col(self):
        return super().groupby_col()

    def set_df(self, financial_df: pd.DataFrame):
        groupby_col = self.groupby_col
        if isinstance(groupby_col, str):
            sort_cols = [groupby_col, 'date']
        else:
            sort_cols = groupby_col.append('date')
        # we are actually sorting our dataframe reference, if we wish to not manipulate out input, do copy instead
        financial_df.sort_values(sort_cols, inplace=True)
        super().set_df(financial_df)

    def adx(self, n: int, adjust: bool = True, inplace: bool = False):
        """
        Creates a series for Average Direction Index series from lookback period

        :param high: pd.Series, array
        :param low: pd.Series, array
        :param close: pd.Series, array
        :param n: int (lookback period)
        :return: pd.Series, array
        """

        if adjust:
            def func(df):
                return talib.ADX(df.highadj, df.lowadj, df.closeadj, timeperiod=n)
        else:
            def func(df):
                return talib.ADX(df.high, df.low, df.close, timeperiod=n)

        eng_values = self.df.groupby(self.groupby_col).apply(func)
        if inplace:
            self.df[f'adx_{n}'] = eng_values.values
        else:
            return eng_values

    def ema(self, col: str, n: int, inplace: bool = True):
        """
        Creates an Exponential Moving Average series from lookback period

        :param series: pd.Series, array
        :param n: int (lookback period)
        :return: pd.Series, array
        """

        def func(series):
            return talib.EMA(series, timeperiod=n)

        eng_values = self.df.groupby(self.groupby_col)[col].apply(func)
        if inplace:
            self.df[f'ema_{col}_{n}'] = eng_values.values
        else:
            return eng_values

    def ma(self, col: str, n: int, inplace: bool = True):
        """
        Creates a Simple Moving Average series from lookback period

        :param series: pd.Series, array
        :param n: int (lookback period)
        :return: pd.Series, array
        """
        def func(series):
            return talib.MA(series, timeperiod=n)

        eng_values = self.df.groupby(self.groupby_col)[col].apply(func)
        if inplace:
            self.df[f'ma_{col}_{n}'] = eng_values.values
        else:
            return eng_values
