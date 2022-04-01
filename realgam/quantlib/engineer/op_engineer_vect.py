import pandas as pd
import numpy as np
from typing import Union, List
from realgam.quantlib.engineer.interface import BaseEngineer, GroupBaseEngineer
from realgam.quantlib.engineer.utils import rank_last_value, rolling_rank
from joblib import Parallel, delayed

PRIMARY_KEY = 'permaticker'


class OpEngineerV(BaseEngineer):
    """
    Class for Engineering Factors, contains methods and operations to create factors via vectorization
    Requires input dataframe to have multiindex strictly in the following hierarchy: ['ticker', 'date']
    """

    def __init__(self, financial_df: pd.DataFrame):
        # we are actually sorting our dataframe reference, if we wish to not manipulate out input, do copy instead
        if list(financial_df.index.names) != [PRIMARY_KEY, 'date']:
            raise Exception("OpEngineerV object requires input dataframe to have multiindex strictly in "
                            "the following hierarchy: [PRIMARY_KEY, 'date']")

        # ensure that the reference dataframe also is sorted incase we want to do
        # financial_df['alpha1'] = aev.alpha1(5, 20)
        financial_df.sort_values([PRIMARY_KEY, 'date'], inplace=True)
        super().__init__(financial_df)

    @property
    def df(self):
        return super().df()

    def set_df(self, financial_df: pd.DataFrame):
        if list(financial_df.index.names) != [PRIMARY_KEY, 'date']:
            raise Exception("OpEngineerV object requires input dataframe to have multiindex strictly in "
                            "the following hierarchy: [PRIMARY_KEY, 'date']")

        financial_df.sort_values([PRIMARY_KEY, 'date'], inplace=True)
        super().__init__(financial_df)

    def ts_lag(self, col: str, n: int = 1, wide: bool = False, inplace: bool = False):
        """
        Shifts the desired data column
        :param col: desired column
        :param n: window
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")

        unstacked = self.df[col].unstack(PRIMARY_KEY)
        eng_values = unstacked.shift(n)

        if inplace:
            self.df[f'ts_lag{n}_{col}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def ts_retfwd(self, future: str = 'closeadj', previous: str = 'openadj', n: int = 5, wide: bool = False,
                  inplace: bool = False):
        """
        Calculate forward returns. The formula use avoids biases. It is not wise to do close to close returns because
        we compute signals at the closing price, so it's not wise to use that closing price as the buy price. Instead,
        we might use the next day open price.
        Formula: Future price(t + n) / Previous price(t + 1) - 1

        :param future: Future Price
        :param previous: Previous Price
        :param n: lookforward window
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")
        price1 = self.df[future].unstack(PRIMARY_KEY)
        price2 = self.df[previous].unstack(PRIMARY_KEY)

        eng_values = price1.shift(-n).div(price2.shift(-1)).subtract(1)

        if inplace:
            self.df[f'ts_retfwd_{future}_{previous}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def ts_ret(self, col: str = 'closeadj', wide: bool = False, inplace: bool = False):
        """
        Calculate daily return or t-1 return given a price column
        :param col :prices
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")
        price = self.df[col].unstack(PRIMARY_KEY)
        eng_values = price.pct_change()

        if inplace:
            self.df[f'ts_ret_{col}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def ts_retn(self, col: str, n: int, wide: bool = False, inplace: bool = False):
        """
        Calculate return for t-n given price column
        :param col: prices
        :param n: lags
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")
        price = self.df[col].unstack(PRIMARY_KEY)
        eng_values = price.pct_change(n)

        if inplace:
            self.df[f'ts_retn{n}_{col}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def ts_std(self, col: str, n: int, wide: bool = False, inplace: bool = False):
        """
        Rolling volatility of prices
        :param col: prices
        :param n: lags
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")
        price = self.df[col].unstack(PRIMARY_KEY)
        eng_values = price.rolling(n).std()

        if inplace:
            self.df[f'ts_std{n}_{col}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def ts_lag(self, col: str, n: int, wide: bool = False, inplace: bool = False):
        """
        Rolling lagged value of given column
        :param col: desired column
        :param n: lags
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")
        unstacked = self.df[col].unstack(PRIMARY_KEY)
        eng_values = unstacked.shift(n)

        if inplace:
            self.df[f'ts_lag{n}_{col}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def ts_sum(self, col: str, n: int, wide: bool = False, inplace: bool = False):
        """
        Rolling summed value of given column
        :param col: desired column
        :param n: lags
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")
        unstacked = self.df[col].unstack(PRIMARY_KEY)
        eng_values = unstacked.rolling(n).sum()

        if inplace:
            self.df[f'ts_sum{n}_{col}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def ts_product(self, col: str, n: int, wide: bool = False, inplace: bool = False):
        """
        Rolling product value of given column
        :param col: desired column
        :param n: lags
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")

        def func(series):
            return series.rolling(n).apply(np.prod, engine='cython', raw=True)

        unstacked = self.df[col].unstack(PRIMARY_KEY)
        eng_values = pd.concat(Parallel(n_jobs=-1)(delayed(func)(unstacked[col]) for col in unstacked), axis=1)

        # eng_values = unstacked.rolling(n).apply(np.prod, engine='cython', raw=True)
        if inplace:
            self.df[f'ts_product{n}_{col}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def ts_delta(self, col, n: int, wide: bool = False, inplace: bool = False):
        """
        Equivalent to x(t) - x(t-n)
        :param col: desired values
        :param n: lag
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")
        unstacked = self.df[col].unstack(PRIMARY_KEY)
        eng_values = unstacked.diff(n)

        if inplace:
            self.df[f'ts_delta{n}_{col}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def pct_change_cols(self, current_col: str, prev_col: str, wide: bool = False, inplace: bool = False):
        """
        Percentage change between two columns or series
        (Groupby mode won't effect this function's output)
        :param current_col: current series
        :param prev_col: previous series
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")
        eng_values = self.df[current_col] / self.df[prev_col] - 1

        if inplace:
            self.df[f'pct_change_cols_{current_col}_{prev_col}'] = eng_values.values
        else:
            if wide:
                return eng_values.unstack(PRIMARY_KEY)
            else:
                return eng_values

    def ts_max(self, col: str, n: int, wide: bool = False, inplace: bool = False):
        """
        Rolling time series max based on given col
        :param col: desired column
        :param n: lags
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")
        unstacked = self.df[col].unstack(PRIMARY_KEY)
        eng_values = unstacked.rolling(n).max()

        if inplace:
            self.df[f'ts_max{n}_{col}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def ts_min(self, col: str, n: int, wide: bool = False, inplace: bool = False):
        """
        Rolling time series min based on given col
        :param col: desired column
        :param n: lags
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")
        unstacked = self.df[col].unstack(PRIMARY_KEY)
        eng_values = unstacked.rolling(n).min()

        if inplace:
            self.df[f'ts_min{n}_{col}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def ts_argmax(self, col: str, n: int, wide: bool = False, inplace: bool = False):
        """
        Rolling time series argmax based on given col
        :param col: series
        :param n: lag
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")

        def func(series):
            return series.rolling(n).apply(np.argmax, engine='cython', raw=True).add(1)

        unstacked = self.df[col].unstack(PRIMARY_KEY)
        eng_values = pd.concat(Parallel(n_jobs=-1)(delayed(func)(unstacked[col]) for col in unstacked), axis=1)
        # eng_values = unstacked.rolling(n).apply(np.argmax, engine='cython', raw=True).add(1)

        if inplace:
            self.df[f'ts_argmax{n}_{col}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def ts_argmin(self, col: str, n: int, wide: bool = False, inplace: bool = False):
        """
        Rolling time series argmin based on given col
        :param col: series
        :param n: lag
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")

        def func(series):
            return series.rolling(n).apply(np.argmin, engine='cython', raw=True).add(1)

        unstacked = self.df[col].unstack(PRIMARY_KEY)
        eng_values = pd.concat(Parallel(n_jobs=-1)(delayed(func)(unstacked[col]) for col in unstacked), axis=1)
        # eng_values = unstacked.rolling(n).apply(np.argmin, engine='cython', raw=True).add(1)

        if inplace:
            self.df[f'ts_argmax{n}_{col}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def cs_rank(self, col: str, wide: bool = False, inplace: bool = False, **kwargs):
        """
        Ranking by given column
        :param col: series
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")
        unstacked = self.df[col].unstack(PRIMARY_KEY)
        eng_values = unstacked.rank(axis=1, **kwargs)

        if inplace:
            self.df[f'cs_rank_{col}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def cs_pctrank(self, col: str, wide: bool = False, inplace: bool = False, **kwargs):
        """
        Percentile ranking by given column
        :param col: series
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")
        unstacked = self.df[col].unstack(PRIMARY_KEY)
        eng_values = unstacked.rank(axis=1, pct=True, **kwargs)

        if inplace:
            self.df[f'cs_pctrank_{col}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def ts_rank(self, col: str, n: int, wide: bool = False, inplace: bool = False):
        """
        Rolling time series rank based on given col
        :param col: series
        :param n: rollling window
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")
        unstacked = self.df[col].unstack(PRIMARY_KEY)
        eng_values = pd.DataFrame(rolling_rank(unstacked, n), index=unstacked.index, columns=unstacked.columns)
        # eng_values = unstacked.rolling(n).apply(rank_last_value, raw=True)

        if inplace:
            self.df[f'ts_rank_{col}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def ts_pctrank(self, col: str, n: int, wide: bool = False, inplace: bool = False):
        """
        Rolling time series percentile rank based on given col
        :param col: series
        :param n: rollling window
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")
        unstacked = self.df[col].unstack(PRIMARY_KEY)
        eng_values = pd.DataFrame(rolling_rank(unstacked, n, pct=True), index=unstacked.index,
                                  columns=unstacked.columns)
        # eng_values = unstacked.rolling(n).apply(rank_last_value, args=(2, True), raw=True)

        if inplace:
            self.df[f'ts_pctrank_{col}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def ts_corr(self, col1: str, col2: str, n: int, wide: bool = False, inplace: bool = False):
        """
        Rolling correlation
        :param col1: column1
        :param col2: column2
        :param n: lags
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")
        unstacked1 = self.df[col1].unstack(PRIMARY_KEY)
        unstacked2 = self.df[col2].unstack(PRIMARY_KEY)
        eng_values = unstacked1.rolling(n).corr(unstacked2)

        if inplace:
            self.df[f'ts_corr{n}_{col1}_{col2}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()

    def ts_cov(self, col1: str, col2: str, n: int, wide: bool = False, inplace: bool = False):
        """
        Rolling covariance
        :param col1: column1
        :param col2: column2
        :param n: lags
        :param wide: if wide return a wide dataframe
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        if not isinstance(wide, bool):
            raise Exception(f"'wide' argument should be a bool, received {type(wide)}")
        unstacked1 = self.df[col1].unstack(PRIMARY_KEY)
        unstacked2 = self.df[col2].unstack(PRIMARY_KEY)
        eng_values = unstacked1.rolling(n).cov(unstacked2)

        if inplace:
            self.df[f'ts_corr{n}_{col1}_{col2}'] = eng_values.stack().swaplevel()
        else:
            if wide:
                return eng_values
            else:
                return eng_values.stack().swaplevel()
