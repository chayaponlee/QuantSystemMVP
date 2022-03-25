import pandas as pd
import numpy as np
from typing import List, Union
from realgam.quantlib.engineer.interface import BaseEngineer, GroupBaseEngineer
from realgam.quantlib.engineer.op_engineer_vect import OpEngineerV
from realgam.quantlib.engineer.ta_engineer_vect import TalibEngineerV
import time


class AlphaEngineerV(BaseEngineer):
    """
    Class for engineering Alpha formulas, apply at groupby level tickers
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

    def alpha1(self, n_ts_arg_max: int = 5, n_std: int = 20, inplace: bool = False):
        """
        Runtime: ~4 min
        Formula: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, n_std) : close), 2.), n_ts_arg_max)) - 0.5)
        :param n_std: lookback range for calculating return vol
        :param n_ts_arg_max: lookback range for calculating ts argmax
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")

        df = self.df.copy()[['closeadj']]
        close = df['closeadj'].unstack('ticker')
        oe = OpEngineerV(df)
        returns = oe.ts_ret('closeadj', wide=True)

        close[returns < 0] = oe.ts_std('closeadj', n_std, wide=True)
        df['closeadj_square'] = np.power(close, 2).stack().swaplevel()
        oe.set_df(df)
        oe.ts_argmax('closeadj_square', n_ts_arg_max, inplace=True)
        alpha_values = oe.cs_pctrank(f'ts_argmax{n_ts_arg_max}_closeadj_square', wide=True).mul(
            -.5).stack().swaplevel()

        if inplace:
            self.df['alpha1'] = alpha_values
        else:
            return alpha_values

    def alpha2(self, n_delta: int = 2, n_corr: int = 6, inplace: bool = False):
        """
        Runtime: 5 min 14 sec
        Formula: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
        :param n_delta: range for calculating delta volume`
        :param n_corr: correlation lookback window
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")

        df = self.df.copy()[['closeadj', 'openadj', 'volume']]
        df['log_volume'] = np.log(df.volume)
        oe = OpEngineerV(df)
        oe.ts_delta('log_volume', n_delta, inplace=True)
        oe.pct_change_cols('closeadj', 'openadj', inplace=True)
        oe.cs_pctrank(f'ts_delta{n_delta}_log_volume', inplace=True)
        oe.cs_pctrank('pct_change_cols_closeadj_openadj', inplace=True)

        alpha_values = -1 * oe.ts_corr(f'cs_pctrank_ts_delta{n_delta}_log_volume',
                                       'cs_pctrank_pct_change_cols_closeadj_openadj',
                                       n_corr).replace([-np.inf, np.inf], np.nan)

        if inplace:
            self.df['alpha2'] = alpha_values
        else:
            return alpha_values

    def alpha3(self, n_corr: int = 10, inplace: bool = False):
        """
        Formula: (-1 * correlation(rank(open), rank(volume), 10))
        :param n_corr: correlation lookback window
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        df = self.df.copy()[['openadj', 'volume']]
        oe = OpEngineerV(df)
        oe.cs_pctrank('openadj', inplace=True)
        oe.cs_pctrank('volume', inplace=True)

        alpha_values = -1 * oe.ts_corr('cs_pctrank_openadj', 'cs_pctrank_volume',
                                       n_corr).replace([-np.inf, np.inf], np.nan)

        if inplace:
            self.df['alpha3'] = alpha_values
        else:
            return alpha_values

    def alpha4(self, n_tsrank: int = 9, inplace: bool = False):
        """
        Formula: (-1 * Ts_Rank(rank(low), 9))
        :param n_tsrank: rolling window for tsrank
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        df = self.df.copy()[['lowadj']]

        oe = OpEngineerV(df)
        oe.cs_pctrank('lowadj', inplace=True)

        alpha_values = -1 * oe.ts_rank('cs_pctrank_lowadj', n_tsrank)

        if inplace:
            self.df['alpha4'] = alpha_values
        else:
            return alpha_values

    def alpha5(self, n_sum: int = 10, inplace: bool = False):
        """
        Formula: (rank((open - ts_mean(vwap, 10))) * (-1 * abs(rank((close - vwap)))))
        :param n_sum: window
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        df = self.df.copy()
        open = df['openadj'].unstack('ticker')

        tae = TalibEngineerV(df)
        df['vwap'] = tae.daily_vwap()

        oe = OpEngineerV(df)
        # term1 = open - (sum(vwap, 10) / 10
        df['term1'] = (open - oe.ts_sum('vwap', n_sum, wide=True).divide(n_sum)).stack().swaplevel()
        df['close-vwap'] = df['closeadj'] - df['vwap']
        oe.set_df(df)
        alpha_values = oe.cs_pctrank('term1', wide=True).mul(
            oe.cs_pctrank('close-vwap', wide=True).abs().mul(-1)).stack().swaplevel()

        if inplace:
            self.df['alpha5'] = alpha_values
        else:
            return alpha_values

    def alpha6(self, n_corr: int = 10, inplace: bool = False) -> Union[None, pd.Series]:
        """
        Formula: (-1 * correlation(open, volume, 10))
        :param n_corr: window for correlation
        :param inplace: if True, add engineered column to dataframe attr
        :return: if inplace == False, return pd.Series
        """
        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")
        df = self.df.copy()[['openadj', 'volume']]

        oe = OpEngineerV(df)
        alpha_values = -1 * oe.ts_corr('openadj', 'volume', n_corr)

        if inplace:
            self.df['alpha6'] = alpha_values
        else:
            return alpha_values

    def alpha7(self, n_adv: int = 20, n_delta: int = 7, n_ts_rank: int = 60, inplace: bool = False):
        """
        (adv20 < volume)
        ? ((-ts_rank(abs(ts_delta(close, 7)), 60)) * sign(ts_delta(close, 7)))
        : -1
        :param n_adv:
        :param n_delta:
        :param n_ts_rank:
        :return:
        """
        # (adv20 < volume)
        # ? ((-ts_rank(abs(ts_delta(close, 7)), 60)) * sign(ts_delta(close, 7)))
        # : -1

        if not isinstance(inplace, bool):
            raise Exception(f"'inplace' argument should be a bool, received {type(inplace)}")

        df = self.df.copy()[['closeadj', 'volume']]
        # df['alpha'] = -1

        tae = TalibEngineerV(df)
        adv = tae.adv(n_adv, wide=True)
        volume = df['volume'].unstack('ticker')

        oe = OpEngineerV(df)

        df[f'ts_delta{n_delta}_closeadj'] = oe.ts_delta('closeadj', n_delta)

        df[f'ts_delta{n_delta}_closeadj_abs'] = df[f'ts_delta{n_delta}_closeadj'].abs()

        oe.set_df(df)

        # term_2 = ts_delta(close, 7)
        term_2 = df[f'ts_delta{n_delta}_closeadj'].unstack('ticker')

        alpha_values = -1 * oe.ts_rank(f'ts_delta{n_delta}_closeadj_abs', n_ts_rank, wide=True).mul(
            np.sign(term_2)).where(adv < volume).stack().swaplevel()

        if inplace:
            self.df['alpha7'] = alpha_values
        else:
            return alpha_values
