from abc import ABC, abstractmethod
import json
import pandas as pd


class Engineer(ABC):

    def __init__(self, financial_df):
        self.f_df = financial_df

