import pandas as pd
import numpy as np
from typing import Union, List
from realgam.quantlib.engineer.interface import BaseEngineer, GroupBaseEngineer
from realgam.quantlib.engineer.utils import rank_last_value


class OpEngineer(BaseEngineer):
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

    def ts_ret(self, price_col: str, inplace: bool = False):
        """
        Calculate daily return or t-1 return given a price column
        :param price_col:prices
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df[price_col].pct_change()

        if inplace:
            self.df[f'ts_ret_{price_col}'] = eng_values.values
        else:
            return eng_values

    def ts_retn(self, price_col: str, n: int, inplace: bool = False):
        """
        Calculate return for t-n given price column
        :param price_col: prices
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df[price_col].pct_change(n)

        if inplace:
            self.df[f'ts_retn{n}_{price_col}'] = eng_values.values
        else:
            return eng_values

    def ts_std(self, price_col: str, n: int, inplace: bool = False):
        """
        Rolling volatility of prices
        :param price_col: prices
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.ts_ret(price_col).rolling(n).std()

        if inplace:
            self.df[f'ts_std{n}_{price_col}'] = eng_values.values
        else:
            return eng_values

    def ts_lag(self, col: str, n: int, inplace: bool = False):
        """
        Rolling lagged value of given column
        :param col: desired column
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df[col].shift(n)

        if inplace:
            self.df[f'ts_lag{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def ts_sum(self, col: str, n: int, inplace: bool = False):
        """
        Rolling summed value of given column
        :param col: desired column
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df[col].rolling(n).sum()

        if inplace:
            self.df[f'ts_sum{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def ts_product(self, col: str, n: int, inplace: bool = False):
        """
        Rolling product value of given column
        :param col: desired column
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df[col].rolling(n).apply(np.prod, engine='cython', raw=True)

        if inplace:
            self.df[f'ts_product{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def ts_delta(self, col, n: int, inplace: bool = False):
        """
        Equivalent to x(t) - x(t-n)
        :param col: desired values
        :param n: lag
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df[col].diff(n)

        if inplace:
            self.df[f'ts_delta{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def pct_change_cols(self, current_col: str, prev_col: str, inplace: bool = False):
        """
        Percentage change between two columns or series
        (Groupby mode won't effect this function's output)
        :param current_col: current series
        :param prev_col: previous series
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df[current_col] / self.df[prev_col] - 1

        if inplace:
            self.df[f'pct_change_cols_{current_col}_{prev_col}'] = eng_values.values
        else:
            return eng_values

    def ts_max(self, col: str, n: int, inplace: bool = False):
        """
        Rolling time series max based on given col
        :param col: desired column
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df[col].rolling(n).max()

        if inplace:
            self.df[f'ts_max{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def ts_min(self, col: str, n: int, inplace: bool = False):
        """
        Rolling time series min based on given col
        :param col: desired column
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df[col].rolling(n).min()

        if inplace:
            self.df[f'ts_min{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def ts_argmax(self, col: str, n: int, inplace: bool = False):
        """
        Rolling time series argmax based on given col
        :param col: series
        :param n: lag
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df[col].rolling(n).apply(np.argmax, engine='cython', raw=True).add(1)

        if inplace:
            self.df[f'ts_argmax{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def ts_argmin(self, col: str, n: int, inplace: bool = False):
        """
        Rolling time series argmin based on given col
        :param col: series
        :param n: lag
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df[col].rolling(n).apply(np.argmin, engine='cython', raw=True).add(1)

        if inplace:
            self.df[f'ts_argmax{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def cs_rank(self, col: str, inplace: bool = False, **kwargs):
        """
        Ranking by given column
        :param col: series
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df[col].rank(**kwargs).add(1)

        if inplace:
            self.df[f'cs_rank_{col}'] = eng_values.values
        else:
            return eng_values

    def cs_pctrank(self, col: str, inplace: bool = False, **kwargs):
        """
        Percentile ranking by given column
        :param col: series
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df[col].rank(pct=True, **kwargs)

        if inplace:
            self.df[f'cs_pctrank_{col}'] = eng_values.values
        else:
            return eng_values

    def ts_rank(self, col: str, n: int, inplace: bool = False, **kwargs):
        """
        Rolling time series rank based on given col
        :param col: series
        :param n: rollling window
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df[col].rolling(n).apply(rank_last_value, raw=True).add(1)

        if inplace:
            self.df[f'ts_rank_{col}'] = eng_values.values
        else:
            return eng_values

    def ts_pctrank(self, col: str, n: int, inplace: bool = False, **kwargs):
        """
        Rolling time series percentile rank based on given col
        :param col: series
        :param n: rollling window
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df[col].rolling(n).apply(rank_last_value, args=(2, True), raw=True)

        if inplace:
            self.df[f'ts_pctrank_{col}'] = eng_values.values
        else:
            return eng_values


class GroupOpEngineer(GroupBaseEngineer):
    """
    Class for Factor Engineering at a group level, for example factor engineering by tickers
    """

    def __init__(self, financial_df: pd.DataFrame, groupby_col: Union[str, List]):
        if isinstance(groupby_col, str):
            sort_cols = [groupby_col, 'date']
        else:
            sort_cols = groupby_col + ['date']
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
            sort_cols = groupby_col + ['date']
        # we are actually sorting our dataframe reference, if we wish to not manipulate out input, do copy instead
        financial_df.sort_values(sort_cols, inplace=True)
        super().set_df(financial_df)

    def ts_ret(self, price_col: str, inplace: bool = False):
        """
        Calculate daily return or t-1 return given a price column
        :param price_col:prices
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        # func = lambda x: x.pct_change()
        eng_values = self.df.groupby(self.groupby_col)[price_col].pct_change()
        if inplace:
            self.df[f'ts_ret_{price_col}'] = eng_values.values
        else:
            return eng_values

    def ts_retn(self, price_col: str, n: int, inplace: bool = False):
        """
        Calculate return for t-n given price column
        :param price_col: prices
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        # func = lambda x: x / x.shift(n) - 1
        eng_values = self.df.groupby(self.groupby_col)[price_col].pct_change(n)
        if inplace:
            self.df[f'ts_retn{n}_{price_col}'] = eng_values.values
        else:
            return eng_values

    def ts_std(self, price_col: str, n: int, inplace: bool = False):
        """
        Rolling volatility of prices
        :param price_col: prices
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        df = self.df.copy()
        df['ret'] = self.ts_ret(price_col)

        eng_values = df.groupby(self.groupby_col)['ret'].rolling(n).std()
        if inplace:
            self.df[f'ts_std{n}_{price_col}'] = eng_values.values
        else:
            return eng_values

    def ts_lag(self, col: str, n: int, inplace: bool = False):
        """
        Rolling lagged value of given column
        :param col: desired column
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df.groupby(self.groupby_col)[col].shift(n)

        if inplace:
            self.df[f'ts_lag{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def ts_sum(self, col: str, n: int, inplace: bool = False):
        """
        Rolling summed value of given column
        :param col: desired column
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df.groupby(self.groupby_col)[col].rolling(n).sum()

        if inplace:
            self.df[f'ts_sum{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def ts_product(self, col: str, n: int, inplace: bool = False):
        """
        Rolling product value of given column
        :param col: desired column
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df.groupby(self.groupby_col)[col].rolling(n).apply(np.prod, engine='cython', raw=True)

        if inplace:
            self.df[f'ts_product{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def ts_delta(self, col, n: int, inplace: bool = False):
        """
        Equivalent to x(t) - x(t-n)
        :param col: desired values
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        # func = lambda x: x - x.shift(n)
        eng_values = self.df.groupby(self.groupby_col)[col].diff(n)
        if inplace:
            self.df[f'ts_delta{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def pct_change_cols(self, current_col: str, prev_col: str, inplace: bool = False):
        """
        Percentage change between two columns or series
        (Groupby mode won't effect this function's output)
        :param current_col: current series
        :param prev_col: previous series
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df[current_col] / self.df[prev_col] - 1
        if inplace:
            self.df[f'pct_change_cols_{current_col}_{prev_col}'] = eng_values.values
        else:
            return eng_values

    def ts_max(self, col: str, n: int, inplace: bool = False):
        """
        Rolling time series max based on given col
        :param col: desired column
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df.groupby(self.groupby_col)[col].rolling(n).max()

        if inplace:
            self.df[f'ts_max{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def ts_min(self, col: str, n: int, inplace: bool = False):
        """
        Rolling time series min based on given col
        :param col: desired column
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df.groupby(self.groupby_col)[col].rolling(n).min()

        if inplace:
            self.df[f'ts_min{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def ts_argmax(self, col: str, n: int, inplace: bool = False):
        """
        Rolling time series argmax based on given col
        :param col: series
        :param n: lag
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df.groupby(self.groupby_col)[col].rolling(n).apply(np.argmax, engine='cython', raw=True).add(
            1)
        if inplace:
            self.df[f'ts_argmax{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def ts_argmin(self, col: str, n: int, inplace: bool = False):
        """
        Rolling time series argmin based on given col
        :param col: series
        :param n: lag
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df.groupby(self.groupby_col)[col].rolling(n).apply(np.argmin, engine='cython', raw=True).add(
            1)
        if inplace:
            self.df[f'ts_argmin{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def cs_rank(self, col: str, inplace: bool = False, **kwargs):
        """
        Ranking by given column groupby dates
        :param col: series
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df.groupby('date')[col].rank(**kwargs).add(1)
        if inplace:
            self.df[f'cs_rank_{col}'] = eng_values.values
        else:
            return eng_values

    def cs_pctrank(self, col: str, inplace: bool = False, **kwargs):
        """
        Percentile ranking by given column groupby dates
        :param col: series
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df.groupby('date')[col].rank(pct=True, **kwargs)
        if inplace:
            self.df[f'cs_pctrank_{col}'] = eng_values.values
        else:
            return eng_values

    def ts_rank(self, col: str, n: int, inplace: bool = False, **kwargs):
        """
        Rolling time series rank based on given col
        :param col: series
        :param n: rollling window
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df.groupby(self.groupby_col)[col].rolling(n).apply(rank_last_value, raw=True).add(1)

        if inplace:
            self.df[f'ts_rank{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def ts_pctrank(self, col: str, n: int, inplace: bool = False, **kwargs):
        """
        Rolling time series pctrank based on given col
        :param col: series
        :param n: rollling window
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df.groupby(self.groupby_col)[col].rolling(n).apply(rank_last_value, args=(2, True), raw=True)

        if inplace:
            self.df[f'ts_pctrank{n}_{col}'] = eng_values.values
        else:
            return eng_values

    def ts_corr(self, col1: str, col2: str, n: int, inplace: bool = False):
        """
        Rolling correlation
        :param col1: column1
        :param col2: column2
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df.groupby(self.groupby_col)[[col1, col2]].rolling(n).corr().iloc[0::2, -1]

        if inplace:
            self.df[f'ts_corr{n}_{col1}_{col2}'] = eng_values.values
        else:
            return eng_values

    def ts_cov(self, col1: str, col2: str, n: int, inplace: bool = False):
        """
        Rolling covariance
        :param col1: column1
        :param col2: column2
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        eng_values = self.df.groupby(self.groupby_col)[[col1, col2]].rolling(n).cov().iloc[0::2, -1]
        if inplace:
            self.df[f'ts_cov{n}_{col1}_{col2}'] = eng_values.values
        else:
            return eng_values
