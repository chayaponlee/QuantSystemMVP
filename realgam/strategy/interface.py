from abc import ABC, abstractmethod
import json
import pandas as pd


class BaseStrategy(ABC):
    """
       Interface for strategies
       Defines the mandatory structure must follow any custom strategies
    """

    def __init__(self, config: str) -> None:

        with open(config) as f:
            self.config = json.load(f)
        # super().__init__(config)

    @abstractmethod
    def extend_historicals(self, historical_data: pd.DataFrame) -> pd.DataFrame:

        return historical_data

    @abstractmethod
    def run_backtest(self, historical_data: pd.DataFrame) -> pd.DataFrame:

        return historical_data

    @abstractmethod
    def get_subsys_pos(self) -> None:

        pass

