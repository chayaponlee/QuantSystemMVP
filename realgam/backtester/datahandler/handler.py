from realgam.backtester.interface import IDataHandler
import os

import numpy as np
import pandas as pd

from realgam.backtester.event.event import MarketEvent
from realgam.quantlib import general_utils as gu, qlogger

import logging

logger = qlogger.init(__file__, logging.INFO)


class HistoricCSVDataHandler(IDataHandler):
    """
    HistoricCSVDataHandler is designed to read CSV files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface.
    """

    def __init__(self, events, data_dir, symbol_list):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.

        It will be assumed that all files are of the form
        'symbol.csv', where symbol is a string in the list.

        Parameters:
        events - The Event Queue.
        data_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol strings.
        """
        self.events = events
        self.data_dir = data_dir
        self.symbol_list = symbol_list

        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self.bar_index = 0

        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.

        For this handler it will be assumed that the data is
        taken from AlphaVantage. Thus its format will be respected.
        """
        comb_index = None
        for s in self.symbol_list:
            # Load the CSV file with no header information, indexed on date
            self.symbol_data[s] = pd.read_csv(
                os.path.join(self.data_dir, '%s.csv' % s),
                header=0, index_col=0, parse_dates=True,
                names=[
                    'datetime', 'open', 'high',
                    'low', 'close', 'volume', 'adj_close',
                ]
            )
            self.symbol_data[s].sort_index(inplace=True)

            # Combine the index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_data[s].index
            else:
                comb_index.union(self.symbol_data[s].index)

            # Set the latest symbol_data to None
            self.latest_symbol_data[s] = []

        for s in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].reindex(
                index=comb_index, method='pad'
            )
            self.symbol_data[s]["returns"] = self.symbol_data[s]["adj_close"].pct_change().dropna()
            self.symbol_data[s] = self.symbol_data[s].iterrows()

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed.
        """
        for b in self.symbol_data[symbol]:
            yield b

    def get_latest_bar(self, symbol):
        """
        Returns the last bar from the latest_symbol list.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            logger.info("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1]

    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            logger.info("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-N:]

    def get_latest_bar_datetime(self, symbol):
        """
        Returns a Python datetime object for the last bar.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            logger.info("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1][0]

    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the pandas Bar series object.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            logger.info("That symbol is not available in the historical data set.")
            raise
        else:
            return getattr(bars_list[-1][1], val_type)

    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        try:
            bars_list = self.get_latest_bars(symbol, N)
        except KeyError:
            logger.info("That symbol is not available in the historical data set.")
            raise
        else:
            return np.array([getattr(b[1], val_type) for b in bars_list])

    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for s in self.symbol_list:
            try:
                bar = next(self._get_new_bar(s))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
        self.events.put(MarketEvent())


class SharadarDataHandler(IDataHandler):
    """
    SharadarDataHandler is designed to read in historical symbols data based on Sharadar's data format
    Currently the type of data is a pickle .obj format
    Data has the following columns: ['permaticker', 'date', 'ticker', 'open', 'high', 'low', 'close', \
    'openadj', 'highadj', 'lowadj', 'closeadj', 'volume']
    Primary symbol key will be permaticker
    """

    def __init__(self, events, data_dir, symbol_list, start_date):
        """
        Initialises the Sharadar data handler by requesting
        for location of the pickle object and loading the whole symbol universe data


        Parameters:
        events - The Event Queue.
        data_dir - Absolute directory path to the Pickle file
        symbol_list - A list of symbol strings.
        symbol_name_convert -Dictionary conversion between permaticker and ticker (ticker is used for executing orders
        and permaticker for internal computations)
        """
        self.events = events
        self.data_dir = data_dir
        self.symbol_list = symbol_list
        self.start_date = start_date
        self.symbol_name_converter = {}
        self.symbol_data = {}
        self.symbol_data_cols = ['date', 'openadj', 'highadj', 'lowadj', 'closeadj', 'volume']
        self.returns_proxy = 'closeadj'
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self.bar_index = 0

        self._open_convert_pickle_file()

    def _open_convert_pickle_file(self):
        """
        Converts pickle file into dataframe and then dictionary format of symbol_data
        """
        comb_index = None
        logger.info(f'Loading data from {self.data_dir}')
        stocks_df, _, _ = gu.load_file(self.data_dir)
        stocks_df = stocks_df.reset_index()
        stocks_df = stocks_df[stocks_df.date >= self.start_date]
        available_tickers = stocks_df[['permaticker', 'ticker']].drop_duplicates(keep='last')
        latest_ticker_universe_pair = available_tickers.drop_duplicates('permaticker', keep='last')
        latest_permatickers = list(latest_ticker_universe_pair['permaticker'])
        latest_tickers = list(latest_ticker_universe_pair['ticker'])

        logger.info("Creating symbol name converter")
        for permaticker, ticker in zip(latest_permatickers, latest_tickers):
            self.symbol_name_converter[permaticker] = ticker
            self.symbol_name_converter[ticker] = permaticker

        if self.symbol_list is None:
            self.symbol_list = latest_permatickers
        else:
            symbols_non_include = [sym for sym in self.symbol_list if sym not in latest_permatickers]
            if len(symbols_non_include) > 0:
                logger.info(f'Symbols not available in data: {symbols_non_include}')
                temp_sym_list = [sym for sym in self.symbol_list if sym in latest_permatickers]
                # remove duplicates
                self.symbol_list = list(dict.fromkeys(temp_sym_list))
            else:
                self.symbol_list = list(dict.fromkeys(self.symbol_list))
                logger.info("All symbols in symbol list are available to backtest")

        logger.info(f'Number of symbols in universe: {len(self.symbol_list)}')

        logger.info("Iterating data onto symbol data ")
        for s in self.symbol_list:
            # Loop through desired symbol list

            self.symbol_data[s] = stocks_df[stocks_df.permaticker == s][self.symbol_data_cols]
            self.symbol_data[s].set_index('date', inplace=True)
            self.symbol_data[s].sort_index(inplace=True)

            # Combine the index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_data[s].index
            else:
                comb_index.union(self.symbol_data[s].index)

            # Set the latest symbol_data to None
            self.latest_symbol_data[s] = []

        logger.info("Reindexing data for each symbol")
        for s in self.symbol_list:
            # print(s)
            # if s == '199059':
            #
            #     print(self.symbol_data[s])
            self.symbol_data[s] = self.symbol_data[s].reindex(
                index=comb_index, method='pad'
            )
            self.symbol_data[s]["returns"] = self.symbol_data[s][self.returns_proxy].pct_change().dropna()
            self.symbol_data[s] = self.symbol_data[s].iterrows()

        logger.info("!!!!!Finish Initializing SharadarDataHandler!!!!!")

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed.
        """
        for b in self.symbol_data[symbol]:
            yield b

    def get_latest_bar(self, symbol):
        """
        Returns the last bar from the latest_symbol list.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            logger.info("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1]

    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            logger.info("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-N:]

    def get_latest_bar_datetime(self, symbol):
        """
        Returns a Python datetime object for the last bar.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            logger.info("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1][0]

    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the pandas Bar series object.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            logger.info("That symbol is not available in the historical data set.")
            raise
        else:
            return getattr(bars_list[-1][1], val_type)

    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        try:
            bars_list = self.get_latest_bars(symbol, N)
        except KeyError:
            logger.info("That symbol is not available in the historical data set.")
            raise
        else:
            return np.array([getattr(b[1], val_type) for b in bars_list])

    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for s in self.symbol_list:
            try:
                bar = next(self._get_new_bar(s))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
        self.events.put(MarketEvent())
