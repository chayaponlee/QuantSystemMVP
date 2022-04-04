from datetime import datetime as dt
import json
import numpy as np
import pandas as pd
import math
import os
from backtest import Backtest, Backtest2
from realgam.backtester.datahandler import SharadarDataHandler
from realgam.backtester.event import SignalEvent, OrderEvent, OrderEvent2, FillEvent
from realgam.backtester.execution import SimulatedExecutionHandler
from realgam.backtester.portfolio import NaivePortfolio, MaturePortfolio
from realgam.backtester.interface import IStrategy, IOrderManager, IExecutionHandler
from realgam.backtester.order_manager import NaiveOrderManager

from realgam.quantlib import qlogger
import logging

logger = qlogger.init(__file__, logging.INFO)

PROJECT_PATH = os.getenv('QuantSystemMVP')
DATA_PATH = f'{PROJECT_PATH}/Data/historical/stock_hist_perma.obj'

PORT_CONFIG_PATH = f'{PROJECT_PATH}/realgam/backtester/portfolio/p_config.json'
PORT_CONFIG = json.load(open(PORT_CONFIG_PATH))

STRAT_CONFIG_PATH = f'{PROJECT_PATH}/realgam/backtester/strategy_config.json'
STRAT_CONFIG = json.load(open(STRAT_CONFIG_PATH))

# ======================
# config
# ======================
MAX_OPEN_TRADES = 5
TRADABLE_BALANCE_RATIO = 0.95


class MovingAverageCrossStrategy(IStrategy):
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
        print(short_window)
        print(long_window)
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
        # ensure that enough data to calculate smas and previous smas
        data_lookback_threshold = self.long_window + 2
        if event.type == 'MARKET':
            for symbol in self.symbol_list:
                bars = self.bars.get_latest_bars_values(symbol, "closeadj", N=data_lookback_threshold)

                if bars is not None and len(bars) >= data_lookback_threshold:

                    previous_short_sma = np.mean(bars[-self.short_window - 1: -1])
                    previous_long_sma = np.mean(bars[-self.long_window - 1: -1])
                    short_sma = np.mean(bars[-self.short_window:])
                    long_sma = np.mean(bars[-self.long_window:])

                    dt = self.bars.get_latest_bar_datetime(symbol)
                    # sig_dir = ""
                    strength = 1.0
                    strategy_id = 1

                    if short_sma > long_sma and previous_short_sma < previous_long_sma and self.bought[symbol] == "OUT":
                        sig_dir = 'LONG'
                        signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength)
                        self.events.put(signal)
                        self.bought[symbol] = 'LONG'

                    elif short_sma < long_sma and previous_short_sma > previous_long_sma and self.bought[
                        symbol] == "LONG":
                        sig_dir = 'EXIT'
                        signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength)
                        self.events.put(signal)
                        self.bought[symbol] = 'OUT'


class MACOrderManager(NaiveOrderManager):
    """
    Flexible Class for Creating a variability of orders by
    listening to Signals and generating orders by sending it to the
    Execution Handler
    """

    def __init__(self, events, portfolio):
        super().__init__(events, portfolio)

    def generate_order(self, signal):
        """
        Simply files an Order object as a constant quantity
        sizing of the signal object, without risk management or
        position sizing considerations.

        Parameters:
        signal - The tuple containing Signal information.
        """
        order = None

        symbol = signal.symbol
        direction = signal.signal_type
        strength = signal.strength

        # mkt_quantity = 100
        cur_quantity = self.portfolio.current_positions[symbol]
        order_type = 'ND_MKT_CLOSE'

        if direction == 'LONG' and cur_quantity == 0:
            order = OrderEvent2(symbol, order_type, 'BUY')
        if direction == 'SHORT' and cur_quantity == 0:
            order = OrderEvent2(symbol, order_type, 'SELL')

        if direction == 'EXIT' and cur_quantity > 0:
            order = OrderEvent2(symbol, order_type, 'SELL')
        if direction == 'EXIT' and cur_quantity < 0:
            order = OrderEvent2(symbol, order_type, 'BUY')
        return order


class MACExecutionHandler(SimulatedExecutionHandler):
    """
    The simulated execution handler simply converts all order
    objects into their equivalent fill objects automatically
    without latency, slippage or fill-ratio issues.

    This allows a straightforward "first go" test of any strategy,
    before implementation with a more sophisticated execution
    handler.
    """

    def __init__(self, events, portfolio, bars):
        """
        Initialises the handler, setting the event queues
        up internally.

        Parameters:
        events - The Queue of Event objects.
        """
        super().__init__(events)
        self.portfolio = portfolio
        self.current_open_trades = 0
        self.max_open_trades = MAX_OPEN_TRADES
        self.tradable_balance_ratio = TRADABLE_BALANCE_RATIO
        self.bars = bars
        self.pending_orders = []

    def manage_risk_order(self, event, price_type):
        """
        Currently for position sizing
        :param event:
        :return:
        """
        current_cash_balance = self.portfolio.current_holdings['cash']
        if self.max_open_trades > self.current_open_trades:
            trade_cash_quantity = (current_cash_balance * self.tradable_balance_ratio) / \
                                  (self.max_open_trades - self.current_open_trades)
            latest_price = self.bars.get_latest_bar_value(
                event.symbol, price_type
            )
            trade_quantity = math.floor(trade_cash_quantity / latest_price)
            return trade_quantity

        else:
            return 0

    def execute_buy_order(self, event, trade_quantity):
        """
        Simply converts Order objects into Fill objects naively,
        i.e. without any latency, slippage or fill ratio problems.

        Parameters:
        event - Contains an Event object with order information.
        """

        fill_event = FillEvent(
            dt.utcnow(), event.symbol,
            'ARCA', trade_quantity, event.direction, None
        )
        logger.info(f'Buy {event.symbol} at {trade_quantity}')
        self.current_open_trades += 1
        self.events.put(fill_event)

    def execute_sell_order(self, event, trade_quantity):
        """
        Simply converts Order objects into Fill objects naively,
        i.e. without any latency, slippage or fill ratio problems.

        Parameters:
        event - Contains an Event object with order information.
        """

        fill_event = FillEvent(
            dt.utcnow(), event.symbol,
            'ARCA', trade_quantity, event.direction, None
        )
        logger.info(f'Sell {event.symbol} at {trade_quantity}')
        self.current_open_trades -= 1
        self.events.put(fill_event)

    def excecute_order(self, event):

        if event.direction == 'BUY':

            if event.order_type in (['ND_MKT_CLOSE', 'MKT_ON_CLOSE']):
                trade_quantity = self.manage_risk_order(event, PORT_CONFIG['close_column_name'])
            elif event.order_type == 'ND_MKT_OPEN':
                trade_quantity = self.manage_risk_order(event, PORT_CONFIG['open_column_name'])
            else:
                raise Exception("Invalid type of order")
            if trade_quantity > 0:
                self.execute_buy_order(event, trade_quantity)
            else:
                logger.info("Reached maximum allowed trades: NO FILL")

        elif event.direction == 'SELL':

            cur_quantity = self.portfolio.current_positions[event.symbol]
            trade_quantity = abs(cur_quantity)
            self.execute_sell_order(event, trade_quantity)

    def execute_pending_order(self):

        if len(self.pending_orders) > 0:
            for event in self.pending_orders:
                self.excecute_order(event)

            self.pending_orders = []

    def manage_order(self, event):
        if event.type == 'ORDER':
            if event.order_type in (['ND_MKT_CLOSE', 'ND_MKT_OPEN']):
                logger.info("Received pending order")
                self.pending_orders.append(event)
            else:
                self.excecute_order(event)


if __name__ == "__main__":
    # data_dir = DATA_PATH
    symbol_list = STRAT_CONFIG['symbol_list']
    initial_capital = 100000.0
    start_date = pd.to_datetime('2016-01-01')
    heartbeat = 0.0

    ma_short = 50
    ma_long = 200
    backtest_file_name = f'MAC_{ma_short}_{ma_long}'

    backtest = Backtest2(DATA_PATH,
                         backtest_file_name,
                         symbol_list,
                         initial_capital,
                         heartbeat,
                         start_date,
                         SharadarDataHandler,
                         MACExecutionHandler,
                         MaturePortfolio,
                         MACOrderManager,
                         MovingAverageCrossStrategy, short_window=ma_short, long_window=ma_long)

    backtest.simulate_trading()
