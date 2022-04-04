# backtest.py

import pprint
import time

try:
    import Queue as queue
except ImportError:
    import queue

from realgam.quantlib import qlogger
import logging

logger = qlogger.init(__file__, logging.INFO)


class Backtest:
    """
    Enscapsulates the settings and components for carrying out
    an event-driven backtest.
    """

    def __init__(
            self, data_dir, strategy_output_name, symbol_list, initial_capital,
            heartbeat, start_date, data_handler,
            execution_handler, portfolio, strategy, **kwargs
    ):
        """
        Initialises the backtest.

        Parameters:
        data_dir - The hard root to the CSV data directory.
        symbol_list - The list of symbol strings.
        intial_capital - The starting capital for the portfolio.
        heartbeat - Backtest "heartbeat" in seconds
        start_date - The start datetime of the strategy.
        data_handler - (Class) Handles the market data feed.
        execution_handler - (Class) Handles the orders/fills for trades.
        portfolio - (Class) Keeps track of portfolio current and prior positions.
        strategy_output_name - Name of backtest file which is based on our strategy
        strategy - (Class) Generates signals based on market data.
        **kwargs - strategy parameters
        """
        self.data_dir = data_dir
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date

        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy
        self.strategy_output_name = strategy_output_name
        self.strategy_args = kwargs
        self.events = queue.Queue()

        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1

        self._generate_trading_instances()

    def _generate_trading_instances(self):
        """
        Generates the trading instance objects from 
        their class types.
        """
        logger.info(
            "Creating DataHandler, Strategy, Portfolio and ExecutionHandler"
        )
        self.data_handler = self.data_handler_cls(self.events, self.data_dir, self.symbol_list, self.start_date)
        self.strategy = self.strategy_cls(self.data_handler, self.events, **self.strategy_args)
        self.portfolio = self.portfolio_cls(self.data_handler, self.events, self.start_date,
                                            self.initial_capital)
        self.execution_handler = self.execution_handler_cls(self.events)

    def _run_backtest(self):
        """
        Executes the backtest.
        """
        i = 0
        while True:
            # first while loop for iterating over the days
            i += 1
            logger.info(i)
            # Update the market bars
            if self.data_handler.continue_backtest == True:
                self.data_handler.update_bars()
            else:
                break

            # Handle the events
            while True:
                # try getting all the events happening within one time period (day) after one available data point (bar)
                # after checking for all the symbols and each symbol's possible events and executing them, the while
                # loop breaks, and we move on to the next bar
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    break
                else:
                    if event is not None:
                        if event.type == 'MARKET':
                            self.strategy.calculate_signals(event)
                            self.portfolio.update_timeindex(event)

                        elif event.type == 'SIGNAL':
                            self.signals += 1
                            self.portfolio.update_signal(event)

                        elif event.type == 'ORDER':
                            self.orders += 1
                            self.execution_handler.execute_order(event)

                        elif event.type == 'FILL':
                            self.fills += 1
                            self.portfolio.update_fill(event)

            time.sleep(self.heartbeat)

    def _output_performance(self):
        """
        Outputs the strategy performance from the backtest.
        """
        self.portfolio.create_equity_curve_dataframe()

        logger.info("Creating summary stats...")
        stats = self.portfolio.output_summary_stats(self.strategy_output_name)

        logger.info("Creating equity curve...")
        logger.info(self.portfolio.equity_curve.tail(10))
        pprint.pprint(stats)

        logger.info("Signals: %s" % self.signals)
        logger.info("Orders: %s" % self.orders)
        logger.info("Fills: %s" % self.fills)

    def simulate_trading(self):
        """
        Simulates the backtest and outputs portfolio performance.
        """
        self._run_backtest()
        self._output_performance()


class Backtest2:
    """
    Enscapsulates the settings and components for carrying out
    an event-driven backtest.
    """

    def __init__(
            self, data_dir, strategy_output_name, symbol_list, initial_capital,
            heartbeat, start_date, data_handler,
            execution_handler, portfolio, order_manager, strategy, **kwargs
    ):
        """
        Initialises the backtest.

        Parameters:
        data_dir - The hard root to the CSV data directory.
        symbol_list - The list of symbol strings.
        intial_capital - The starting capital for the portfolio.
        heartbeat - Backtest "heartbeat" in seconds
        start_date - The start datetime of the strategy.
        data_handler - (Class) Handles the market data feed.
        execution_handler - (Class) Handles the orders/fills for trades.
        portfolio - (Class) Keeps track of portfolio current and prior positions.
        strategy_output_name - Name of backtest file which is based on our strategy
        strategy - (Class) Generates signals based on market data.
        **kwargs - strategy parameters
        """
        self.data_dir = data_dir
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date

        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy
        self.order_manager_cls = order_manager
        self.strategy_output_name = strategy_output_name
        self.strategy_args = kwargs
        self.events = queue.Queue()

        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1

        self._generate_trading_instances()

    def _generate_trading_instances(self):
        """
        Generates the trading instance objects from
        their class types.
        """
        logger.info(
            "Creating DataHandler, Strategy, Portfolio and ExecutionHandler"
        )
        self.data_handler = self.data_handler_cls(self.events, self.data_dir, self.symbol_list, self.start_date)
        self.strategy = self.strategy_cls(self.data_handler, self.events, **self.strategy_args)
        self.portfolio = self.portfolio_cls(self.data_handler, self.events, self.start_date,
                                            self.initial_capital)
        self.order_manager = self.order_manager_cls(self.events, self.portfolio)
        self.execution_handler = self.execution_handler_cls(self.events, self.portfolio, self.data_handler)

    def _run_backtest(self):
        """
        Executes the backtest.
        """
        i = 0
        while True:
            # first while loop for iterating over the days
            i += 1
            logger.info(i)
            # Update the market bars
            if self.data_handler.continue_backtest == True:
                self.data_handler.update_bars()
            else:
                break

            # Handle the events
            while True:
                # try getting all the events happening within one time period (day) after one available data point (bar)
                # after checking for all the symbols and each symbol's possible events and executing them, the while
                # loop breaks, and we move on to the next bar
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    break
                else:
                    if event is not None:
                        if event.type == 'MARKET':
                            self.strategy.calculate_signals(event)
                            self.portfolio.update_timeindex(event)
                            self.execution_handler.execute_pending_order()

                        elif event.type == 'SIGNAL':
                            self.signals += 1
                            self.order_manager.update_signal(event)

                        elif event.type == 'ORDER':
                            self.orders += 1
                            self.execution_handler.manage_order(event)

                        elif event.type == 'FILL':
                            self.fills += 1
                            self.portfolio.update_fill(event)

            time.sleep(self.heartbeat)

    def _output_performance(self):
        """
        Outputs the strategy performance from the backtest.
        """
        self.portfolio.create_equity_curve_dataframe()

        logger.info("Creating summary stats...")
        stats = self.portfolio.output_summary_stats(self.strategy_output_name)

        logger.info("Creating equity curve...")
        logger.info(self.portfolio.equity_curve.tail(10))
        pprint.pprint(stats)

        logger.info("Signals: %s" % self.signals)
        logger.info("Orders: %s" % self.orders)
        logger.info("Fills: %s" % self.fills)

    def simulate_trading(self):
        """
        Simulates the backtest and outputs portfolio performance.
        """
        self._run_backtest()
        self._output_performance()
