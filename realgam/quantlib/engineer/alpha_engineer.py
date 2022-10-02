import pandas as pd
import numpy as np
from typing import List, Union, Dict
from realgam.quantlib.engineer.interface import BaseEngineer, GroupBaseEngineer
from realgam.quantlib.engineer.op_engineer import OpEngineer, GroupOpEngineer
from realgam.quantlib.engineer.op_functions import *


class AlphaEngineer(BaseEngineer):
    """
    Class for engineering Alpha formulas, apply at groupby level tickers
    """

    def __init__(self, financial_df: pd.DataFrame, primary_key: str, date_key: str, candle_names_dict: Dict):
        if list(financial_df.index.names) != [primary_key, date_key]:
            raise Exception("OpEngineerV object requires input dataframe to have multiindex strictly in "
                            "the following hierarchy: [PRIMARY_KEY, 'date']")

        financial_df.sort_values([primary_key, date_key], inplace=True)
        self.primary_key = primary_key
        self.date_key = date_key
        o = financial_df[candle_names_dict['o']].unstack(primary_key).sort_index()
        h = financial_df[candle_names_dict['h']].unstack(primary_key).sort_index()
        l = financial_df[candle_names_dict['l']].unstack(primary_key).sort_index()
        c = financial_df[candle_names_dict['c']].unstack(primary_key).sort_index()
        v = financial_df[candle_names_dict['v']].unstack(primary_key).sort_index()
        vwap = o.add(h).add(l).add(c).div(4).sort_index()
        adv20 = v.rolling(20).mean()
        r = financial_df[candle_names_dict['r']].unstack(primary_key).sort_index()
        self.o = o
        self.h = h
        self.l = l
        self.c = c
        self.v = v
        self.vwap = vwap
        self.adv20 = adv20
        self.r = r
        super().__init__(financial_df)

    @property
    def df(self):
        return super().df()

    def set_df(self, financial_df: pd.DataFrame):
        if list(financial_df.index.names) != [self.primary_key, self.date_key]:
            raise Exception("OpEngineerV object requires input dataframe to have multiindex strictly in "
                            "the following hierarchy: [self.primary_key, self.date_key]")

        financial_df.sort_values([self.primary_key, self.date_key], inplace=True)
        super().__init__(financial_df)

    def alpha1(self):
        """(rank(ts_argmax(power(((returns < 0)
            ? ts_std(returns, 20)
            : close), 2.), 5)) -0.5)"""
        c = self.c.copy()
        r = self.r.copy()
        c[r < 0] = ts_std(r, 20)
        return (rank(ts_argmax(power(c, 2), 5)).mul(-.5)
                .stack().swaplevel())

    def alpha2(self):
        """(-1 * ts_corr(rank(ts_delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
        s1 = rank(ts_delta(log(self.v), 2))
        s2 = rank((self.c / self.o) - 1)
        alpha = -ts_corr(s1, s2, 6)
        return alpha.stack(self.primary_key).swaplevel().replace([-np.inf, np.inf], np.nan)

    def alpha3(self):
        """(-1 * ts_corr(rank(open), rank(volume), 10))"""

        return (-ts_corr(rank(self.o), rank(self.v), 10)
                .stack(self.primary_key)
                .swaplevel()
                .replace([-np.inf, np.inf], np.nan))

    def alpha4(self):
        """(-1 * Ts_Rank(rank(low), 9))"""
        return (-ts_rank(rank(self.l), 9)
                .stack(self.primary_key)
                .swaplevel())

    def alpha5(self):
        """(rank((open - ts_mean(vwap, 10))) * (-1 * abs(rank((close - vwap)))))"""
        return (rank(self.o.sub(ts_mean(self.vwap, 10)))
                .mul(rank(self.c.sub(self.vwap)).mul(-1).abs())
                .stack(self.primary_key)
                .swaplevel())

    def alpha6(self):
        """(-ts_corr(open, volume, 10))"""
        return (-ts_corr(self.o, self.v, 10)
                .stack(self.primary_key)
                .swaplevel())

    def alpha7(self):
        """(adv20 < volume)
            ? ((-ts_rank(abs(ts_delta(close, 7)), 60)) * sign(ts_delta(close, 7)))
            : -1
        """

        delta7 = ts_delta(self.c, 7)
        return (-ts_rank(abs(delta7), 60)
                .mul(sign(delta7))
                .where(self.adv20 < self.v, -1)
                .stack(self.primary_key)
                .swaplevel())

    def alpha8(self):
        """-rank(((ts_sum(open, 5) * ts_sum(returns, 5)) -
            ts_lag((ts_sum(open, 5) * ts_sum(returns, 5)),10)))
        """
        return (-(rank(((ts_sum(self.o, 5) * ts_sum(self.r, 5)) -
                        ts_lag((ts_sum(self.o, 5) * ts_sum(self.r, 5)), 10))))
                .stack(self.primary_key)
                .swaplevel())

    def alpha9(self):
        """(0 < ts_min(ts_delta(close, 1), 5)) ? ts_delta(close, 1)
        : ((ts_max(ts_delta(close, 1), 5) < 0)
        ? ts_delta(close, 1) : (-1 * ts_delta(close, 1)))
        """
        close_diff = ts_delta(self.c, 1)
        alpha = close_diff.where(ts_min(close_diff, 5) > 0,
                                 close_diff.where(ts_max(close_diff, 5) < 0,
                                                  -close_diff))
        return (alpha
                .stack(self.primary_key)
                .swaplevel())

    def alpha10(self):
        """rank(((0 < ts_min(ts_delta(close, 1), 4))
            ? ts_delta(close, 1)
            : ((ts_max(ts_delta(close, 1), 4) < 0)
                ? ts_delta(close, 1)
                : (-1 * ts_delta(close, 1)))))
        """
        close_diff = ts_delta(self.c, 1)
        alpha = close_diff.where(ts_min(close_diff, 4) > 0,
                                 close_diff.where(ts_min(close_diff, 4) > 0,
                                                  -close_diff))

        return (rank(alpha)
                .stack(self.primary_key)
                .swaplevel())

    def alpha11(self):
        """(rank(ts_max((vwap - close), 3)) +
            rank(ts_min(vwap - close), 3)) *
            rank(ts_delta(volume, 3))
            """
        return (rank(ts_max(self.vwap.sub(self.c), 3))
                .add(rank(ts_min(self.vwap.sub(self.c), 3)))
                .mul(rank(ts_delta(self.v, 3)))
                .stack(self.primary_key)
                .swaplevel())

    def alpha12(self):
        """(sign(ts_delta(volume, 1)) *
                (-1 * ts_delta(close, 1)))
            """
        return (sign(ts_delta(self.v, 1)).mul(-ts_delta(self.c, 1))
                .stack(self.primary_key)
                .swaplevel())

    def alpha13(self):
        """-rank(ts_cov(rank(close), rank(volume), 5))"""
        return (-rank(ts_cov(rank(self.c), rank(self.v), 5))
                .stack(self.primary_key)
                .swaplevel())

    def alpha14(self):
        """
        (-rank(ts_delta(returns, 3))) * ts_corr(open, volume, 10))
        """

        alpha = -rank(ts_delta(self.r, 3)).mul(ts_corr(self.o, self.v, 10)
                                               .replace([-np.inf,
                                                         np.inf],
                                                        np.nan))
        return (alpha
                .stack(self.primary_key)
                .swaplevel())

    def alpha15(self):
        """(-1 * ts_sum(rank(ts_corr(rank(high), rank(volume), 3)), 3))"""
        alpha = (-ts_sum(rank(ts_corr(rank(self.h), rank(self.v), 3)
                              .replace([-np.inf, np.inf], np.nan)), 3))
        return (alpha
                .stack(self.primary_key)
                .swaplevel())

    def alpha16(self):
        """(-1 * rank(ts_cov(rank(high), rank(volume), 5)))"""
        return (-rank(ts_cov(rank(self.h), rank(self.v), 5))
                .stack(self.primary_key)
                .swaplevel())

    def alpha17(self):
        """(((-1 * rank(ts_rank(close, 10))) * rank(ts_delta(ts_delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))
            """
        adv20 = ts_mean(self.v, 20)
        return (-rank(ts_rank(self.c, 10))
                .mul(rank(ts_delta(ts_delta(self.c, 1), 1)))
                .mul(rank(ts_rank(self.v.div(adv20), 5)))
                .stack(self.primary_key)
                .swaplevel())

    def alpha18(self):
        """-rank((ts_std(abs((close - open)), 5) + (close - open)) +
                ts_corr(close, open,10))
        """
        return (-rank(ts_std(self.c.sub(self.o).abs(), 5)
                      .add(self.c.sub(self.o))
                      .add(ts_corr(self.c, self.o, 10)
                           .replace([-np.inf,
                                     np.inf],
                                    np.nan)))
                .stack(self.primary_key)
                .swaplevel())

    def alpha19(self):
        """((-1 * sign(((close - ts_lag(close, 7)) + ts_delta(close, 7)))) *
        (1 + rank((1 + ts_sum(returns,250)))))
        """
        return (-sign(ts_delta(self.c, 7) + ts_delta(self.c, 7))
                .mul(1 + rank(1 + ts_sum(self.r, 250)))
                .stack(self.primary_key)
                .swaplevel())

    def alpha20(self):
        """-rank(open - ts_lag(high, 1)) *
            rank(open - ts_lag(close, 1)) *
            rank(open -ts_lag(low, 1))"""
        return (rank(self.o - ts_lag(self.h, 1))
                .mul(rank(self.o - ts_lag(self.c, 1)))
                .mul(rank(self.o - ts_lag(self.l, 1)))
                .mul(-1)
                .stack(self.primary_key)
                .swaplevel())

    def alpha21(self):
        """ts_mean(close, 8) + ts_std(close, 8) < ts_mean(close, 2)
            ? -1
            : (ts_mean(close,2) < ts_mean(close, 8) - ts_std(close, 8)
                ? 1
                : (volume / adv20 < 1
                    ? -1
                    : 1))
        """
        sma2 = ts_mean(self.c, 2)
        sma8 = ts_mean(self.c, 8)
        std8 = ts_std(self.c, 8)

        cond_1 = sma8.add(std8) < sma2
        cond_2 = sma8.add(std8) > sma2
        cond_3 = self.v.div(ts_mean(self.v, 20)) < 1

        # val = np.ones_like(self.c)
        alpha = pd.DataFrame(np.select(condlist=[cond_1, cond_2, cond_3],
                                       choicelist=[-1, 1, -1], default=1),
                             index=self.c.index,
                             columns=self.c.columns)

        return (alpha
                .stack(self.primary_key)
                .swaplevel())

    def alpha22(self):
        """-(ts_delta(ts_corr(high, volume, 5), 5) *
            rank(ts_std(close, 20)))
        """

        return (ts_delta(ts_corr(self.h, self.v, 5)
                         .replace([-np.inf,
                                   np.inf],
                                  np.nan), 5)
                .mul(rank(ts_std(self.c, 20)))
                .mul(-1)
                .stack(self.primary_key)
                .swaplevel())

    def alpha23(self):
        """((ts_mean(high, 20) < high)
                ? (-1 * ts_delta(high, 2))
                : 0
            """

        return (ts_delta(self.h, 2)
                .mul(-1)
                .where(ts_mean(self.h, 20) < self.h, 0)
                .stack(self.primary_key)
                .swaplevel())

    def alpha24(self):
        """((((ts_delta((ts_mean(close, 100)), 100) / ts_lag(close, 100)) <= 0.05)
            ? (-1 * (close - ts_min(close, 100)))
            : (-1 * ts_delta(close, 3)))
        """
        cond = ts_delta(ts_mean(self.c, 100), 100) / ts_lag(self.c, 100) <= 0.05

        return (self.c.sub(ts_min(self.c, 100)).mul(-1).where(cond, -ts_delta(self.c, 3))
                .stack(self.primary_key)
                .swaplevel())

    def alpha25(self):
        """rank((-1 * returns) * adv20 * vwap * (high - close))"""
        return (rank(-self.r.mul(self.adv20)
                     .mul(self.vwap)
                     .mul(self.h.sub(self.c)))
                .stack(self.primary_key)
                .swaplevel())

    def alpha26(self):
        """(-1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))"""
        return (ts_max(ts_corr(ts_rank(self.v, 5),
                               ts_rank(self.h, 5), 5)
                       .replace([-np.inf, np.inf], np.nan), 3)
                .mul(-1)
                .stack(self.primary_key)
                .swaplevel())

    def alpha27(self):
        """((0.5 < rank(ts_mean(ts_corr(rank(volume), rank(vwap), 6), 2)))
                ? -1
                : 1)"""
        cond = rank(ts_mean(ts_corr(rank(self.v),
                                    rank(self.vwap), 6), 2))
        alpha = cond.notnull().astype(float)
        return (alpha.where(cond <= 0.5, -alpha)
                .stack(self.primary_key)
                .swaplevel())

    def alpha28(self):
        """scale(((ts_corr(adv20, low, 5) + (high + low) / 2) - close))"""
        return (scale(ts_corr(self.adv20, self.l, 5)
                      .replace([-np.inf, np.inf], 0)
                      .add(self.h.add(self.l).div(2).sub(self.c)))
                .stack(self.primary_key)
                .swaplevel())

    def alpha29(self):
        """(ts_min(ts_product(rank(rank(scale(log(ts_sum(ts_min(rank(rank((-1 *
                rank(ts_delta((close - 1),5))))), 2), 1))))), 1), 5)
            + ts_rank(ts_lag((-1 * returns), 6), 5))
        """
        return (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-rank(ts_delta((self.c - 1), 5)))), 2))))), 5)
                .add(ts_rank(ts_lag((-1 * self.r), 6), 5))
                .stack(self.primary_key)
                .swaplevel())

    def alpha30(self):
        """(((1.0 - rank(((sign((close - ts_lag(close, 1))) +
                sign((ts_lag(close, 1) - ts_lag(close, 2)))) +
                sign((ts_lag(close, 2) - ts_lag(close, 3)))))) *
                ts_sum(volume, 5)) / ts_sum(volume, 20))"""
        close_diff = ts_delta(self.c, 1)
        return (rank(sign(close_diff)
                     .add(sign(ts_lag(close_diff, 1)))
                     .add(sign(ts_lag(close_diff, 2))))
                .mul(-1).add(1)
                .mul(ts_sum(self.v, 5))
                .div(ts_sum(self.v, 20))
                .stack(self.primary_key)
                .swaplevel())

    def alpha31(self):
        """((rank(rank(rank(ts_weighted_mean((-1 * rank(rank(ts_delta(close, 10)))), 10)))) +
            rank((-1 * ts_delta(close, 3)))) + sign(scale(ts_corr(adv20, low, 12))))
        """
        return (rank(rank(rank(ts_weighted_mean(rank(rank(ts_delta(self.c, 10))).mul(-1), 10))))
                .add(rank(ts_delta(self.c, 3).mul(-1)))
                .add(sign(scale(ts_corr(self.adv20, self.l, 12)
                                .replace([-np.inf, np.inf],
                                         np.nan))))
                .stack(self.primary_key)
                .swaplevel())

    def alpha32(self):
        """scale(ts_mean(close, 7) - close) +
            (20 * scale(ts_corr(vwap, ts_lag(close, 5),230)))"""
        return (scale(ts_mean(self.c, 7).sub(self.c))
                .add(20 * scale(ts_corr(self.vwap,
                                        ts_lag(self.c, 5), 230)))
                .stack(self.primary_key)
                .swaplevel())

    def alpha33(self):
        """rank(-(1 - (open / close)))"""
        return (rank(self.o.div(self.c).mul(-1).add(1).mul(-1))
                .stack(self.primary_key)
                .swaplevel())

    def alpha34(self):
        """rank(((1 - rank((ts_std(returns, 2) / ts_std(returns, 5)))) + (1 - rank(ts_delta(close, 1)))))"""

        return (rank(rank(ts_std(self.r, 2).div(ts_std(self.r, 5))
                          .replace([-np.inf, np.inf],
                                   np.nan))
                     .mul(-1)
                     .sub(rank(ts_delta(self.c, 1)))
                     .add(2))
                .stack(self.primary_key)
                .swaplevel())

    def alpha35(self):
        """((ts_Rank(volume, 32) *
            (1 - ts_Rank(((close + high) - low), 16))) *
            (1 -ts_Rank(returns, 32)))
        """
        return (ts_rank(self.v, 32)
                .mul(1 - ts_rank(self.c.add(self.h).sub(self.l), 16))
                .mul(1 - ts_rank(self.r, 32))
                .stack(self.primary_key)
                .swaplevel())

    def alpha36(self):
        """2.21 * rank(ts_corr((close - open), ts_lag(volume, 1), 15)) +
            0.7 * rank((open- close)) +
            0.73 * rank(ts_Rank(ts_lag(-1 * returns, 6), 5)) +
            rank(abs(ts_corr(vwap,adv20, 6))) +
            0.6 * rank(((ts_mean(close, 200) - open) * (close - open)))
        """

        return (rank(ts_corr(self.c.sub(self.o), ts_lag(self.v, 1), 15)).mul(2.21)
                .add(rank(self.o.sub(self.c)).mul(.7))
                .add(rank(ts_rank(ts_lag(-self.r, 6), 5)).mul(0.73))
                .add(rank(abs(ts_corr(self.vwap, self.adv20, 6))))
                .add(rank(ts_mean(self.c, 200).sub(self.o).mul(self.c.sub(self.o))).mul(0.6))
                .stack(self.primary_key)
                .swaplevel())

    def alpha37(self):
        """(rank(ts_corr(ts_lag((open - close), 1), close, 200)) + rank((open - close)))"""
        return (rank(ts_corr(ts_lag(self.o.sub(self.c), 1), self.c, 200))
                .add(rank(self.o.sub(self.c)))
                .stack(self.primary_key)
                .swaplevel())

    def alpha38(self):
        """"-1 * rank(ts_rank(close, 10)) * rank(close / open)"""
        return (rank(ts_rank(self.o, 10))
                .mul(rank(self.c.div(self.o).replace([-np.inf, np.inf], np.nan)))
                .mul(-1)
                .stack(self.primary_key)
                .swaplevel())

    def alpha39(self):
        """-rank(ts_delta(close, 7) * (1 - rank(ts_weighted_mean(volume / adv20, 9)))) *
                (1 + rank(ts_sum(returns, 250)))"""
        return (rank(ts_delta(self, 7).mul(rank(ts_weighted_mean(self.v.div(self.adv20), 9)).mul(-1).add(1))).mul(-1)
                .mul(rank(ts_mean(self.r, 250).add(1)))
                .stack(self.primary_key)
                .swaplevel())

    def alpha40(self):
        """((-1 * rank(ts_std(high, 10))) * ts_corr(high, volume, 10))
        """
        return (rank(ts_std(self.h, 10))
                .mul(ts_corr(self.h, self.v, 10))
                .mul(-1)
                .stack(self.primary_key)
                .swaplevel())

    def alpha41(self):
        """power(high * low, 0.5 - vwap"""
        return (power(self.h.mul(self.l), 0.5)
                .sub(self.vwap)
                .stack(self.primary_key)
                .swaplevel())

    def alpha42(self):
        """rank(vwap - close) / rank(vwap + close)"""
        return (rank(self.vwap.sub(self.c))
                .div(rank(self.vwap.add(self.c)))
                .stack(self.primary_key)
                .swaplevel())

    def alpha43(self):
        """(ts_rank((volume / adv20), 20) * ts_rank((-1 * ts_delta(close, 7)), 8))"""

        return (ts_rank(self.v.div(self.adv20), 20)
                .mul(ts_rank(ts_delta(self.c, 7).mul(-1), 8))
                .stack(self.primary_key)
                .swaplevel())

    def alpha44(self):
        """-ts_corr(high, rank(volume), 5)"""

        return (ts_corr(self.h, rank(self.v), 5)
                .replace([-np.inf, np.inf], np.nan)
                .mul(-1)
                .stack(self.primary_key)
                .swaplevel())

    def alpha45(self):
        """-(rank((ts_mean(ts_lag(close, 5), 20)) *
            ts_corr(close, volume, 2)) *
            rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2)))"""

        return (rank(ts_mean(ts_lag(self.c, 5), 20))
                .mul(ts_corr(self.c, self.v, 2)
                     .replace([-np.inf, np.inf], np.nan))
                .mul(rank(ts_corr(ts_sum(self.c, 5),
                                  ts_sum(self.c, 20), 2)))
                .mul(-1)
                .stack(self.primary_key)
                .swaplevel())

    def alpha46(self):
        """0.25 < ts_lag(ts_delta(close, 10), 10) / 10 - ts_delta(close, 10) / 10
                ? -1
                : ((ts_lag(ts_delta(close, 10), 10) / 10 - ts_delta(close, 10) / 10 < 0)
                    ? 1
                    : -ts_delta(close, 1))
        """

        cond = ts_lag(ts_delta(self.c, 10), 10).div(10).sub(ts_delta(self.c, 10).div(10))
        alpha = pd.DataFrame(-np.ones_like(cond),
                             index=self.c.index,
                             columns=self.c.columns)
        alpha[cond.isnull()] = np.nan
        return (cond.where(cond > 0.25,
                           -alpha.where(cond < 0,
                                        -ts_delta(self.c, 1)))
                .stack(self.primary_key)
                .swaplevel())

    def alpha47(self):
        """((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) /
            (ts_sum(high, 5) /5))) - rank((vwap - ts_lag(vwap, 5))))"""

        return (rank(self.c.pow(-1)).mul(self.v).div(self.adv20)
                .mul(self.h.mul(rank(self.h.sub(self.c))
                           .div(ts_mean(self.h, 5)))
                     .sub(rank(ts_delta(self.vwap, 5))))
                .stack(self.primary_key)
                .swaplevel())

    # def alpha48(c, industry):
    #     """(indneutralize(((ts_corr(ts_delta(close, 1), ts_delta(ts_lag(close, 1), 1), 250) *
    #         ts_delta(close, 1)) / close), IndClass.subindustry) /
    #         ts_sum(((ts_delta(close, 1) / ts_lag(close, 1))^2), 250))"""
    #     pass

    def alpha48(self):
        """(indneutralize(((ts_corr(ts_delta(close, 1), ts_delta(ts_lag(close, 1), 1), 250) *
            ts_delta(close, 1)) / close), IndClass.subindustry) /
            ts_sum(((ts_delta(close, 1) / ts_lag(close, 1))^2), 250))"""
        return np.nan

    def alpha49(self):
        """ts_delta(ts_lag(close, 10), 10).div(10).sub(ts_delta(close, 10).div(10)) < -0.1 * c
            ? 1
            : -ts_delta(close, 1)"""
        cond = (ts_delta(ts_lag(self.c, 10), 10).div(10)
                .sub(ts_delta(self.c, 10).div(10)) >= -0.1 * self.c)
        return (-ts_delta(self.c, 1)
                .where(cond, 1)
                .stack(self.primary_key)
                .swaplevel())

    def alpha50(self):
        """-ts_max(rank(ts_corr(rank(volume), rank(vwap), 5)), 5)"""
        return (ts_max(rank(ts_corr(rank(self.v),
                                    rank(self.vwap), 5)), 5)
                .mul(-1)
                .stack(self.primary_key)
                .swaplevel())

    def alpha51(self):
        """ts_delta(ts_lag(close, 10), 10).div(10).sub(ts_delta(close, 10).div(10)) < -0.05 * c
            ? 1
            : -ts_delta(close, 1)"""
        cond = (ts_delta(ts_lag(self.c, 10), 10).div(10)
                .sub(ts_delta(self.c, 10).div(10)) >= -0.05 * self.c)
        return (-ts_delta(self.c, 1)
                .where(cond, 1)
                .stack(self.primary_key)
                .swaplevel())

    def alpha52(self):
        """(ts_lag(ts_min(low, 5), 5) - ts_min(low, 5)) *
            rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220) *
            ts_rank(volume, 5)
        """
        return (ts_delta(ts_min(self.l, 5), 5)
                .mul(rank(ts_sum(self.r, 240)
                          .sub(ts_sum(self.r, 20))
                          .div(220)))
                .mul(ts_rank(self.v, 5))
                .stack(self.primary_key)
                .swaplevel())

    def alpha53(self):
        """-1 * ts_delta(1 - (high - close) / (close - low), 9)"""
        inner = (self.c.sub(self.l)).add(1e-6)
        return (ts_delta(self.h.sub(self.c)
                         .mul(-1).add(1)
                         .div(self.c.sub(self.l)
                              .add(1e-6)), 9)
                .mul(-1)
                .stack(self.primary_key)
                .swaplevel())

    def alpha54(self):
        """-(low - close) * power(open, 5) / ((low - high) * power(close, 5))"""
        return (self.l.sub(self.c).mul(self.o.pow(5)).mul(-1)
                .div(self.l.sub(self.h).replace(0, -0.0001).mul(self.c ** 5))
                .stack(self.primary_key)
                .swaplevel())

    def alpha55(self):
        """(-1 * ts_corr(rank(((close - ts_min(low, 12)) /
                                (ts_max(high, 12) - ts_min(low,12)))),
                        rank(volume), 6))"""

        return (ts_corr(rank(self.c.sub(ts_min(self.l, 12))
                             .div(ts_max(self.h, 12).sub(ts_min(self.l, 12))
                                  .replace(0, 1e-6))),
                        rank(self.v), 6)
                .replace([-np.inf, np.inf], np.nan)
                .mul(-1)
                .stack(self.primary_key)
                .swaplevel())

    def alpha56(self):
        """-rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) *
            rank((returns * cap))
        """
        return np.nan

    def alpha57(self):
        """-(close - vwap) / ts_weighted_mean(rank(ts_argmax(close, 30)), 2)"""
        return (self.c.sub(self.vwap.add(1e-5))
                .div(ts_weighted_mean(rank(ts_argmax(self.c, 30)))).mul(-1)
                .stack(self.primary_key)
                .swaplevel())

    def alpha58(self):
        """(-1 * ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(vwap, IndClass.sector), volume, 3), 7), 5))"""
        return np.nan

    def alpha59(self):
        """-ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(vwap, IndClass.industry), volume, 4), 16), 8)"""
        return np.nan

    def alpha60(self):
        """-((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))"""
        return (scale(rank(self.c.mul(2).sub(self.l).sub(self.h)
                           .div(self.h.sub(self.l).replace(0, 1e-5))
                           .mul(self.v))).mul(2)
                .sub(scale(rank(ts_argmax(self.c, 10)))).mul(-1)
                .stack(self.primary_key)
                .swaplevel())
