import pandas as pd
import numpy as np
from typing import Union, List
from realgam.quantlib.engineer.interface import BaseEngineer, GroupBaseEngineer


class FactorEngineer(BaseEngineer):
    """
    Class for Engineering Factors, contains methods and operations to create factors
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

    def calc_ret(self, price_col: str, inplace: bool = False):
        """
        Calculate daily return or t-1 return given a price column
        :param price_col:prices
        :param inplace:if True, creates column to current dataframe attribute, otherwise returns the series
        :return:
        """

        eng_values = self.df[price_col] / self.df[price_col].shift(1) - 1
        if inplace:
            self.df['ret'] = eng_values.values
        else:
            return eng_values

    def calc_retn(self, price_col: str, n: int = 2, inplace: bool = False):
        """
        Calculate return for t-n given price column
        :param price_col: prices
        :param n: lags
        :param inplace: if True, creates column to current dataframe attribute, otherwise returns the series
        :return:
        """

        eng_values = self.df[price_col] / self.df[price_col].shift(1) - 1

        if inplace:
            self.df[f'ret_lag{n}'] = eng_values.values
        else:
            return eng_values

    def roll_vol(self, price_col: str, n: int = 20, inplace: bool = False):
        """
        Rolling volatility of prices
        :param price_col: prices
        :param n: lags
        :param inplace: if True, creates column to current dataframe attribute, otherwise returns the series
        :return:
        """

        eng_values = self.calc_ret(price_col).rolling(n).std()

        if inplace:
            self.df[f'vol_{n}'] = eng_values.values
        else:
            return eng_values

    def delta(self, col, n: int = 1, inplace: bool = False) -> pd.Series:
        """
        Equivalent to x(t) - x(t-n)
        :param col: desired values
        :param n: lag
        :param inplace: if True, creates column to current dataframe attribute, otherwise returns the series
        :return:
        """

        eng_values = self.df[col] - self.df[col].shift(n)

        if inplace:
            self.df[f'delta_{col}_{n}'] = eng_values.values
        else:
            return eng_values

    def pct_change_cols(self, current_col: str, prev_col: str, inplace: bool = False):
        """
        Percentage change between two columns or series
        (Groupby mode won't effect this function's output)
        :param current_col: current series
        :param prev_col: previous series
        :param inplace: if True, creates column to current dataframe attribute, otherwise returns the series
        :return:
        """
        eng_values = self.df[current_col] / self.df[prev_col] - 1
        if inplace:
            self.df[f'pct_change_{current_col}_{prev_col}'] = eng_values.values
        else:
            return eng_values

    def roll_tsargmax(self, col: str, n: int, inplace: bool = False):
        """
        Rolling time series argmax based on given col
        :param col: series
        :param n: lag
        :param inplace: if True, creates column to current dataframe attribute, otherwise returns the series
        :return:
        """
        eng_values = self.df[col].rolling(n).apply(np.argmax, engine='cython', raw=True)

        if inplace:
            self.df[f'tsargmax_{col}'] = eng_values.values
        else:
            return eng_values

    def roll_tsargmin(self, col: str, n: int, inplace: bool = False):
        """
        Rolling time series argmin based on given col
        :param col: series
        :param n: lag
        :param inplace: if True, creates column to current dataframe attribute, otherwise returns the series
        :return:
        """
        eng_values = self.df[col].rolling(n).apply(np.argmin, engine='cython', raw=True)

        if inplace:
            self.df[f'tsargmin_{col}'] = eng_values.values
        else:
            return eng_values

    def rank(self, col: str, inplace: bool = False, **kwargs):
        """
        Ranking by given column
        :param col: series
        :param inplace:
        :return:
        """
        eng_values = self.df[col].rank(**kwargs)

        if inplace:
            self.df[f'rank_{col}'] = eng_values.values
        else:
            return eng_values

    def pctrank(self, col: str, inplace: bool = False, **kwargs):
        """
        Percentile ranking by given column
        :param col: series
        :param inplace:
        :return:
        """
        eng_values = self.df[col].rank(pct=True, **kwargs)

        if inplace:
            self.df[f'pctrank_{col}'] = eng_values.values
        else:
            return eng_values

    def roll_tsrank(self, col: str, n: int, inplace: bool = False, **kwargs):
        """
        Rolling time series rank based on given col
        :param col: series
        :param n: rollling window
        :param inplace: apply to dataframe attr
        :return:
        """
        eng_values = self.df[col].rolling(n).apply(lambda x: pd.Series(x).rank(**kwargs).iloc[-1],
                                                   engine='cython', raw=True)

        if inplace:
            self.df[f'tsrank_{col}'] = eng_values.values
        else:
            return eng_values

    def roll_tspctrank(self, col: str, n: int, inplace: bool = False, **kwargs):
        """
        Rolling time series percentile rank based on given col
        :param col: series
        :param n: rollling window
        :param inplace: apply to dataframe attr
        :return:
        """
        eng_values = self.df[col].rolling(n).apply(lambda x: pd.Series(x).rank(pct=True, **kwargs).iloc[-1],
                                                   engine='cython', raw=True)

        if inplace:
            self.df[f'tspctrank_{col}'] = eng_values.values
        else:
            return eng_values


class GroupFactorEngineer(GroupBaseEngineer):
    """
    Class for Factor Engineering at a group level, for example factor engineering by tickers
    """

    def __init__(self, financial_df: pd.DataFrame, groupby_col: Union[str, List]):
        if isinstance(groupby_col, str):
            sort_cols = [groupby_col, 'date']
        else:
            sort_cols = groupby_col.append('date')
        # we are actually sorting our dataframe reference, if we wish to not manipulate out input, do copy instead
        financial_df.sort_values(sort_cols, inplace=True)
        super().__init__(financial_df.copy(), groupby_col)

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

    def calc_ret(self, price_col: str, inplace: bool = False):
        """
        Calculate daily return or t-1 return given a price column
        :param price_col:prices
        :param inplace:if True, creates column to current dataframe attribute, otherwise returns the series
        :return:
        """
        func = lambda x: x / x.shift(1) - 1
        eng_values = self.df.groupby(self.groupby_col)[price_col].apply(func)
        if inplace:
            self.df['ret'] = eng_values.values
        else:
            return eng_values

    def calc_retn(self, price_col: str, n: int = 2, inplace: bool = False):
        """
        Calculate return for t-n given price column
        :param price_col: prices
        :param n: lags
        :param inplace: if True, creates column to current dataframe attribute, otherwise returns the series
        :return:
        """
        func = lambda x: x / x.shift(n) - 1
        eng_values = self.df.groupby(self.groupby_col)[price_col].apply(func)
        if inplace:
            self.df[f'ret_lag{n}'] = eng_values.values
        else:
            return eng_values

    def roll_vol(self, price_col: str, n: int = 20, inplace: bool = False):
        """
        Rolling volatility of prices
        :param price_col: prices
        :param n: lags
        :param inplace: if True, creates column to current dataframe attribute, otherwise returns the series
        :return:
        """

        df = self.df.copy()
        df['ret'] = self.calc_ret(price_col)

        eng_values = df.groupby(self.groupby_col)['ret'].rolling(n).std()
        if inplace:
            self.df[f'vol_{n}'] = eng_values.values
        else:
            return eng_values

    def delta(self, col, n: int = 1, inplace: bool = False) -> pd.Series:
        """
        Equivalent to x(t) - x(t-n)
        :param col: desired values
        :param n: lag
        :param inplace: if True, creates column to current dataframe attribute, other wise returns the series
        :return:
        """

        func = lambda x: x - x.shift(n)
        eng_values = self.df.groupby(self.groupby_col)[col].apply(func)
        if inplace:
            self.df[f'delta_{col}_{n}'] = eng_values.values
        else:
            return eng_values

    def pct_change_cols(self, current_col: str, prev_col: str, inplace: bool = False):
        """
        Percentage change between two columns or series
        (Groupby mode won't effect this function's output)
        :param current_col: current series
        :param prev_col: previous series
        :param inplace: if True, creates column to current dataframe attribute, otherwise returns the series
        :return:
        """
        eng_values = self.df[current_col] / self.df[prev_col] - 1
        if inplace:
            self.df[f'pct_change_{current_col}_{prev_col}'] = eng_values.values
        else:
            return eng_values

    def roll_tsargmax(self, col: str, n: int, inplace: bool = False):
        """
        Rolling time series argmax based on given col
        :param col: series
        :param n: lag
        :param inplace: if True, creates column to current dataframe attribute, otherwise returns the series
        :return:
        """
        eng_values = self.df.groupby(self.groupby_col)[col].rolling(n).apply(np.argmax, engine='cython', raw=True)

        if inplace:
            self.df[f'tsargmax_{col}'] = eng_values.values
        else:
            return eng_values

    def roll_tsargmin(self, col: str, n: int, inplace: bool = False):
        """
        Rolling time series argmin based on given col
        :param col: series
        :param n: lag
        :param inplace: if True, creates column to current dataframe attribute, otherwise returns the series
        :return:
        """
        eng_values = self.df.groupby(self.groupby_col)[col].rolling(n).apply(np.argmin, engine='cython', raw=True)

        if inplace:
            self.df[f'tsargmin_{col}'] = eng_values.values
        else:
            return eng_values

    def rank(self, col: str, inplace: bool = False, **kwargs):
        """
        Ranking by given column groupby dates
        :param col: series
        :param inplace:
        :return:
        """
        eng_values = self.df.groupby('date')[col].rank(**kwargs)

        if inplace:
            self.df[f'rank_{col}'] = eng_values.values
        else:
            return eng_values

    def pctrank(self, col: str, inplace: bool = False, **kwargs):
        """
        Percentile ranking by given column groupby dates
        :param col: series
        :param inplace:
        :return:
        """
        eng_values = self.df.groupby('date')[col].rank(pct=True, **kwargs)

        if inplace:
            self.df[f'pctrank_{col}'] = eng_values.values
        else:
            return eng_values

    def roll_tsrank(self, col: str, n: int, inplace: bool = False, **kwargs):
        """
        Rolling time series rank based on given col
        :param col: series
        :param n: rollling window
        :param inplace: apply to dataframe attr
        :return:
        """
        eng_values = self.df.groupby(self.groupby_col)[col].rolling(n).apply(
            lambda x: pd.Series(x).rank(**kwargs).iloc[-1], engine='cython', raw=True)

        if inplace:
            self.df[f'tsrank_{col}'] = eng_values.values
        else:
            return eng_values

    def roll_tspctrank(self, col: str, n: int, inplace: bool = False, **kwargs):
        """
        Rolling time series pctrank based on given col
        :param col: series
        :param n: rollling window
        :param inplace: apply to dataframe attr
        :return:
        """
        eng_values = self.df.groupby(self.groupby_col)[col].rolling(n).apply(
            lambda x: pd.Series(x).rank(pct=True, **kwargs).iloc[-1], engine='cython', raw=True)

        if inplace:
            self.df[f'tspctrank_{col}'] = eng_values.values
        else:
            return eng_values
