from datetime import datetime as dt
import json
import numpy as np
import os
from backtest import Backtest
from datahandler.handler import SharadarDataHandler
from event.event import SignalEvent
from execution.sim_execution import SimulatedExecutionHandler
from portfolio import Portfolio
from strategy import Strategy

from realgam.quantlib import qlogger
import logging
logger = qlogger.init(__file__, logging.INFO)

PROJECT_PATH = os.getenv('QuantSystemMVP')
DATA_PATH = f'{PROJECT_PATH}/Data/historical/stock_hist_perma.obj'

STRAT_CONFIG_PATH = f'{PROJECT_PATH}/realgam/backtester/strategy_config.json'
STRAT_CONFIG = json.load(open(STRAT_CONFIG_PATH))


class MovingAverageCrossStrategy(Strategy):
    """
    Carries out a basic Moving Average Crossover strategy with a
    short/long simple weighted moving average. Default short/long
    windows are 100/400 periods respectively.
    """

    def __init__(self, bars, events, short_window=100, long_window=400):
        """
        Initialises the buy and hold strategy.

        Parameters:
        bars - The DataHandler object that provides bar information
        events - The Event Queue object.
        short_window - The short moving average lookback.
        long_window - The long moving average lookback.
        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.short_window = short_window
        self.long_window = long_window

        # Set to True if a symbol is in the market
        self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to 'OUT'.
        """
        bought = {}
        for s in self.symbol_list:
            bought[s] = 'OUT'
        return bought

    def calculate_signals(self, event):
        """
        Generates a new set of signals based on the MAC
        SMA with the short window crossing the long window
        meaning a long entry and vice versa for a short entry.    

        Parameters
        event - A MarketEvent object. 
        """
        if event.type == 'MARKET':
            for symbol in self.symbol_list:
                bars = self.bars.get_latest_bars_values(symbol, "closeadj", N=self.long_window)

                if bars is not None and bars != []:
                    short_sma = np.mean(bars[-self.short_window:])
                    long_sma = np.mean(bars[-self.long_window:])

                    dt = self.bars.get_latest_bar_datetime(symbol)
                    sig_dir = ""
                    strength = 1.0
                    strategy_id = 1

                    if short_sma > long_sma and self.bought[symbol] == "OUT":
                        sig_dir = 'LONG'
                        signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength)
                        self.events.put(signal)
                        self.bought[symbol] = 'LONG'

                    elif short_sma < long_sma and self.bought[symbol] == "LONG":
                        sig_dir = 'EXIT'
                        signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength)
                        self.events.put(signal)
                        self.bought[symbol] = 'OUT'


if __name__ == "__main__":
    # data_dir = DATA_PATH
    symbol_list = STRAT_CONFIG['symbol_list']
    initial_capital = 100000.0
    start_date = dt(2018, 1, 1, 0, 0, 0)
    heartbeat = 0.0

    backtest = Backtest(DATA_PATH,
                        symbol_list,
                        initial_capital,
                        heartbeat,
                        start_date,
                        SharadarDataHandler,
                        SimulatedExecutionHandler,
                        Portfolio,
                        MovingAverageCrossStrategy)

    backtest.simulate_trading()
