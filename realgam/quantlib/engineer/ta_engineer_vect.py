import pandas as pd
import numpy as np
from typing import Union, List
from realgam.quantlib.engineer.interface import BaseEngineer, GroupBaseEngineer
import talib
from joblib import Parallel, delayed


class TalibEngineerV(BaseEngineer):
    """
    Class for Engineering Technical Analysis indicators based on talib, contains methods and operations to create
    techincal indicators

    """

    def __init__(self, financial_df: pd.DataFrame):
        if list(financial_df.index.names) != ['ticker', 'date']:
            raise Exception("OpEngineerV object requires input dataframe to have multiindex strictly in "
                            "the following hierarchy: ['ticker', 'date']")

        # ensure that the reference dataframe also is sorted incase we want to do
        # financial_df['alpha1'] = aev.alpha1(5, 20)
        financial_df.sort_values(['ticker', 'date'], inplace=True)
        super().__init__(financial_df)

    @property
    def df(self):
        return super().df()

    def set_df(self, financial_df: pd.DataFrame):
        if list(financial_df.index.names) != ['ticker', 'date']:
            raise Exception("OpEngineerV object requires input dataframe to have multiindex strictly in "
                            "the following hierarchy: ['ticker', 'date']")

        financial_df.sort_values(['ticker', 'date'], inplace=True)
        super().__init__(financial_df)

    def daily_vwap(self, adjusted: bool = True, wide: bool = False, inplace: bool = False):
        """
        Calculates daily vwap
        Formula: (open + high + low + close) / 4
        :param adjusted: if adjusted, use adjusted prices
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """

        if adjusted:
            adj_str = 'adj'
        else:
            adj_str = ''

        open = self.df[f'open{adj_str}'].unstack('ticker')
        high = self.df[f'high{adj_str}'].unstack('ticker')
        low = self.df[f'low{adj_str}'].unstack('ticker')
        close = self.df[f'close{adj_str}'].unstack('ticker')

        ta_values = (open + high + low + close).divide(4)

        if inplace:
            self.df['vwapadj'] = ta_values.stack().swaplevel()
        else:
            if wide:
                return ta_values
            else:
                return ta_values.stack().swaplevel()

    def adv(self, n: int, wide: bool = False, inplace: bool = False):
        """
        Calculates avg daily volume
        :param n: window
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """

        volume = self.df['volume'].unstack('ticker')
        ta_values = volume.apply(lambda x: talib.MA(x, timeperiod=n))

        if inplace:
            self.df[f'adv{n}'] = ta_values.stack().swaplevel()
        else:
            if wide:
                return ta_values
            else:
                return ta_values.stack().swaplevel()

