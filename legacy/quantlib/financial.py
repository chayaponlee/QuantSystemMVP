from pandas import DataFrame, Series
from typing import List, Callable, Dict


class FinancialSeries(Series):
    @property
    def _constructor(self):
        return FinancialSeries

    @property
    def _constructor_expanddim(self):
        return FinancialDF


class FinancialDF(DataFrame):

    @property
    def _constructor(self):
        return FinancialDF

    @property
    def _constructor_sliced(self):
        return FinancialSeries

    def cross_rank(self, column_to_rank: str, **kwargs) -> Series:
        return self.groupby('date')[column_to_rank].rank(**kwargs)

    def cross_pctrank(self, column_to_rank: str, **kwargs) -> Series:
        return self.groupby('date')[column_to_rank].rank(pct=True, **kwargs)

    def engineer(self, func: Callable, engineer_col: str, groupby_cols='ticker', **kwargs) -> Series:
        return self.groupby(groupby_cols)[engineer_col].apply(func, **kwargs)

    def engineer_s(self, func: Callable, engineer_col: str, **kwargs) -> Series:
        return self[engineer_col].apply(func, **kwargs)

    def calc_alpha(self, alpha_func: Callable, **kwargs) -> Series:
        return alpha_func(self, **kwargs)
