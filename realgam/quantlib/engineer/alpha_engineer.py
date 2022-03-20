import pandas as pd
import numpy as np
from typing import List, Union
from realgam.quantlib.engineer.interface import BaseEngineer, GroupBaseEngineer
from realgam.quantlib.engineer.op_engineer import OpEngineer, GroupOpEngineer


class AlphaEngineer(GroupBaseEngineer):
    """
    Class for engineering Alpha formulas, apply at groupby level tickers
    """

    def __init__(self, financial_df: pd.DataFrame, groupby_col: Union[str, List]):
        if isinstance(groupby_col, str):
            sort_cols = [groupby_col, 'date']
        else:
            sort_cols = groupby_col + ['date']
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

    def alpha1(self, n_ts_arg_max: int = 5, n_std: int = 20, inplace: bool = False):
        """
        Runtime: 58 sec
        Formula: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, n_std) : close), 2.), n_ts_arg_max)) - 0.5)
        :param n_std: lookback range for calculating return vol
        :param n_ts_arg_max: lookback range for calculating ts argmax
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")

        df = self.df.copy()[['closeadj']]
        goe = GroupOpEngineer(df, 'ticker')
        df['ts_ret_closeadj'] = goe.ts_ret('closeadj').values
        df[f'ts_std{n_std}_closeadj'] = goe.ts_std('closeadj', n_std).values
        # here the closeadj column will be replaced by the conditional values
        df.loc[df['ts_ret_closeadj'] < 0, 'closeadj'] = df['ts_std20_closeadj']
        df.loc[df['ts_ret_closeadj'].isnull() | df['ts_std20_closeadj'].isnull(), 'closeadj'] = np.nan
        df['closeadj'] = np.power(df.closeadj, 2)

        goe.set_df(df)
        df[f'ts_argmax{n_ts_arg_max}_closeadj'] = goe.ts_argmax('closeadj', n_ts_arg_max).values

        # here is where i'm in doubt, since we have only 5 unique values from ts_argmax (1-5) the ranking method
        # will be questionable. ideally since there's only 5 unique values normal rank would yield the same value
        # since the enitre universe at one date would have a value between 1-5. so i suggest using pct rank, since
        # the value will be ranked by percentile which takes into account the frequency of each value occurring in
        # the entire universe
        goe.set_df(df)
        df[f'cs_pctrank_ts_argmax{n_ts_arg_max}_closeadj'] = goe.cs_pctrank(f'ts_argmax{n_ts_arg_max}_closeadj')
        # the minus 0.5 i'm not really sure why it's there
        alpha_values = df[f'cs_pctrank_ts_argmax{n_ts_arg_max}_closeadj'] - 0.5

        if inplace:
            self.df['alpha1'] = alpha_values.values
        else:
            return alpha_values

    def alpha2(self, n_delta: int = 2, n_corr: int = 6, inplace: bool = False) -> Union[None, pd.Series]:
        """
        Runtime: 78 min
        Formula: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
        :param n_delta: range for calculating delta volume`
        :param n_corr: correlation lookback window
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        df = self.df.copy()

        df['logvolume'] = np.log(df.volume)
        gfe = GroupOpEngineer(df, 'ticker')
        df[f'delta_logvolume_{n_delta}'] = gfe.ts_delta('logvolume', 2).values
        df['ret_intraday'] = gfe.pct_change_cols('closeadj', 'openadj').values
        gfe.set_df(df)
        df[f'pctrank_delta_logvolume_{n_delta}'] = gfe.cs_pctrank(f'delta_logvolume_{n_delta}').values
        df[f'pctrank_ret_intraday'] = gfe.cs_pctrank('ret_intraday').values

        tickers = df.ticker.unique()
        corr_stack = []

        for i, ticker in enumerate(tickers):
            temp_hist = df[df.ticker == ticker]
            corr_stack.append(temp_hist[f'pctrank_delta_logvolume_{n_delta}'].rolling(n_corr,
                                                                                      ).corr(
                temp_hist['pctrank_ret_intraday'], engine='cython', raw=True))

        alpha_values = -1 * pd.concat(corr_stack)

        if inplace:
            self.df['alpha2'] = alpha_values.values
        else:
            return alpha_values

    def alpha3(self, n_corr: int = 10, inplace: bool = False) -> Union[None, pd.Series]:
        """
        Formula: (-1 * correlation(rank(open), rank(volume), 10))
        :param n_corr: correlation lookback window
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        df = self.df.copy()

        gfe = GroupOpEngineer(df, 'ticker')
        df['pctrank_openadj'] = gfe.cs_pctrank('openadj').values
        df['pctrank_volume'] = gfe.cs_pctrank('volume').values

        tickers = df.ticker.unique()
        corr_stack = []
        for i, ticker in enumerate(tickers):
            temp_hist = df[df.ticker == ticker]
            corr_stack.append(temp_hist['pctrank_openadj'].rolling(n_corr).corr(
                temp_hist['pctrank_volume'], engine='cython', raw=True))

        alpha_values = -1 * pd.concat(corr_stack)

        if inplace:
            self.df['alpha3'] = alpha_values.values
        else:
            return alpha_values

    def alpha4(self, n_tsrank: int = 9, inplace: bool = True) -> Union[None, pd.Series]:
        """
        Formula: (-1 * Ts_Rank(rank(low), 9))
        :param n_tsrank: rolling window for tsrank
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        df = self.df.copy()

        gfe = GroupOpEngineer(df, 'ticker')
        df['pctrank_lowadj'] = gfe.cs_pctrank('lowadj').values
        gfe.set_df(df)
        df['tsrank_rank_lowadj'] = gfe.ts_rank('pctrank_lowadj', n_tsrank).values

        alpha_values = -1 * df['tsrank_rank_lowadj']

        if inplace:
            self.df['alpha4'] = alpha_values.values
        else:
            return alpha_values

    # def alpha5(self, ):
    """
    Uses vwap, will skip for now since unsure how to calculate vwap
    """

    def alpha6(self, n_corr: int = 10, inplace: bool = True) -> Union[None, pd.Series]:
        """
        Formula: (-1 * correlation(open, volume, 10))
        :param n_corr: window for correlation
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        df = self.df.copy()

        tickers = df.ticker.unique()
        corr_stack = []
        for i, ticker in enumerate(tickers):
            temp_hist = df[df.ticker == ticker]
            corr_stack.append(temp_hist['openadj'].rolling(n_corr).corr(
                temp_hist['volume'], engine='cython', raw=True))

        alpha_values = -1 * pd.concat(corr_stack)

        if inplace:
            self.df['alpha6'] = alpha_values.values
        else:
            return alpha_values

    # def alpha7(self, ):
