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

    def daily_vwap(self, inplace: bool = False):

        df = self.df.copy()
        vwap = (df.openadj + df.highadj + df.lowadj + df.open).divide(4)

        if inplace:
            self.df['vwapadj'] = vwap
        else:
            return vwap

