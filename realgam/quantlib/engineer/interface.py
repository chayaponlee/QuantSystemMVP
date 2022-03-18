from abc import ABC, abstractmethod
import pandas as pd
from typing import Union, List

pd.options.mode.chained_assignment = None


class BaseEngineer(ABC):

    def __init__(self, financial_df: pd.DataFrame):
        self.__df = financial_df.copy()

    @abstractmethod
    def df(self):
        return self.__df

    @abstractmethod
    def set_df(self, financial_df: pd.DataFrame):
        self.__df = None
        self.__df = financial_df.copy()


class GroupBaseEngineer(ABC):

    def __init__(self, financial_df: pd.DataFrame, groupby_col: Union[str, List]):
        self.__df = financial_df.copy()
        self.__groupby_col = groupby_col

    @abstractmethod
    def df(self):
        return self.__df

    @abstractmethod
    def set_df(self, financial_df: pd.DataFrame):
        self.__df = financial_df.copy()

    @abstractmethod
    def groupby_col(self):
        return self.__groupby_col
