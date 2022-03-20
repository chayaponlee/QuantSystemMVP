import pandas as pd
import numpy as np
from typing import Union, List
from realgam.quantlib.engineer.interface import BaseEngineer, GroupBaseEngineer
from realgam.quantlib.engineer.utils import rank_last_value


class OpEngineerV(BaseEngineer):
    """
    Class for Engineering Factors, contains methods and operations to create factors via vectorization
    Requires input dataframe to have multiindex strictly in the following hierarchy: ['ticker', 'date']
    """

    def __init__(self, financial_df: pd.DataFrame):
        # we are actually sorting our dataframe reference, if we wish to not manipulate out input, do copy instead
        if list(financial_df.index.names) != ['ticker', 'date']:
            raise Exception("OpEngineerV object requires input dataframe to have multiindex strictly in "
                            "the following hierarchy: ['ticker', 'date']")

        financial_df.sort_values(['ticker', 'date'], inplace=True)
        super().__init__(financial_df)

    @property
    def df(self):
        return super().df()

    def set_df(self, financial_df: pd.DataFrame):
        # we are actually sorting our dataframe reference, if we wish to not manipulate out input, do copy instead
        financial_df.sort_values('date', inplace=True)
        super().set_df(financial_df)

    def ts_ret(self, col: str, inplace: bool = False):
        """
        Calculate daily return or t-1 return given a price column
        :param price_col:prices
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        price = self.df[col].unstack('ticker')
        eng_values = price.pct_change().stack(dropna=False).swaplevel()

        if inplace:
            self.df[f'ts_ret_{col}'] = eng_values
        else:
            return eng_values

    def ts_retn(self, col: str, n: int, inplace: bool = False):
        """
        Calculate return for t-n given price column
        :param price_col: prices
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        price = self.df[col].unstack('ticker')
        eng_values = price.pct_change(n).stack(dropna=False).swaplevel()

        if inplace:
            self.df[f'ts_retn{n}_{col}'] = eng_values
        else:
            return eng_values

    def ts_std(self, col: str, n: int, inplace: bool = False):
        """
        Rolling volatility of prices
        :param price_col: prices
        :param n: lags
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        price = self.df[col].unstack('ticker')
        eng_values = price.rolling(n).std().stack(dropna=False).swaplevel()

        if inplace:
            self.df[f'ts_std{n}_{col}'] = eng_values
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
        unstacked = self.df[col].unstack('ticker')
        eng_values = unstacked.shift(n).stack(dropna=False).swaplevel()

        if inplace:
            self.df[f'ts_lag{n}_{col}'] = eng_values
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
        unstacked = self.df[col].unstack('ticker')
        eng_values = unstacked.rolling(n).sum().stack(dropna=False).swaplevel()

        if inplace:
            self.df[f'ts_sum{n}_{col}'] = eng_values
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
        unstacked = self.df[col].unstack('ticker')
        eng_values = unstacked.shift(n).rolling(n).apply(np.prod, engine='cython', raw=True).stack(dropna=False).swaplevel()

        if inplace:
            self.df[f'ts_product{n}_{col}'] = eng_values
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
        unstacked = self.df[col].unstack('ticker')
        eng_values = unstacked.diff(n).stack().swaplevel()

        if inplace:
            self.df[f'ts_delta{n}_{col}'] = eng_values
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
        unstacked = self.df[col].unstack('ticker')
        eng_values = unstacked.rolling(n).max().stack().swaplevel()

        if inplace:
            self.df[f'ts_max{n}_{col}'] = eng_values
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
        unstacked = self.df[col].unstack('ticker')
        eng_values = unstacked.rolling(n).min().stack().swaplevel()

        if inplace:
            self.df[f'ts_min{n}_{col}'] = eng_values
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
        unstacked = self.df[col].unstack('ticker')
        eng_values = unstacked.rolling(n).apply(np.argmax, engine='cython', raw=True).add(1).stack().swaplevel()

        if inplace:
            self.df[f'ts_argmax{n}_{col}'] = eng_values
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
        unstacked = self.df[col].unstack('ticker')
        eng_values = unstacked.rolling(n).apply(np.argmin, engine='cython', raw=True).add(1).stack().swaplevel()

        if inplace:
            self.df[f'ts_argmax{n}_{col}'] = eng_values
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

    def ts_rank(self, col: str, n: int, inplace: bool = False):
        """
        Rolling time series rank based on given col
        :param col: series
        :param n: rollling window
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        unstacked = self.df[col].unstack('ticker')
        eng_values = unstacked.rolling(n).apply(rank_last_value, raw=True).add(1).stack().swaplevel()

        if inplace:
            self.df[f'ts_rank_{col}'] = eng_values
        else:
            return eng_values

    def ts_pctrank(self, col: str, n: int, inplace: bool = False):
        """
        Rolling time series percentile rank based on given col
        :param col: series
        :param n: rollling window
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        unstacked = self.df[col].unstack('ticker')
        eng_values = unstacked.rolling(n).apply(rank_last_value, args=(2, True), raw=True).stack().swaplevel()

        if inplace:
            self.df[f'ts_pctrank_{col}'] = eng_values
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
        unstacked1 = self.df[col1].unstack('ticker')
        unstacked2 = self.df[col2].unstack('ticker')
        eng_values = unstacked1.rolling(n).corr(unstacked2).stack().swaplevel()

        if inplace:
            self.df[f'ts_corr{n}_{col1}_{col2}'] = eng_values
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
        unstacked1 = self.df[col1].unstack('ticker')
        unstacked2 = self.df[col2].unstack('ticker')
        eng_values = unstacked1.rolling(n).cov(unstacked2).stack().swaplevel()

        if inplace:
            self.df[f'ts_corr{n}_{col1}_{col2}'] = eng_values
        else:
            return eng_values

