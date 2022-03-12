import pandas as pd
import nasdaqdatalink as ndl
import time
import datetime
import logging
from requests.exceptions import ChunkedEncodingError
from realgam.quantlib import qlogger
import json
import os
from dateutil.relativedelta import relativedelta
from typing import List, Set, Dict, Tuple

# log.basicConfig(level='INFO')
logger = qlogger.init(__file__, logging.INFO)

PROJECT_PATH = os.getenv('QuantSystemMVP')
NDL_CONFIG_PATH = f'{PROJECT_PATH}/realgam/quantlib/nasdaq_dl_config.json'
NDL_CONFIG = json.load(open(NDL_CONFIG_PATH))

"""
when obtaining data from numerous sources, we want to standardize communication units.
in other words, we want object types to be the same. for instance, things like dataframe index 'type' or 'class'
should be the same
"""


def format_date(dates) -> datetime.date:
    """
    Standardizes the date value in a particular date format yy-mm-dd

    :param dates: str, datetime object
    :return: datetime object
    """
    yymmdd = list(map(lambda x: int(x), str(dates).split(" ")[0].split("-")))
    # what this does is take a list of dates in [yy--mm--dd {other stuff} format
    # strips away the other stuff , then returns the datetime object
    return datetime.date(yymmdd[0], yymmdd[1], yymmdd[2])


"""
Retrieve American stock data

As of 2022-02-09, we are currently subscribed to 10 years of historical data so we would have atmost data since 
2012.

Configuring nasdaqdatalink API: depending on your directory, you have to enter the API key in to the config file
For MacAir: you can configure the file at 
"/opt/homebrew/Caskroom/miniforge/base/envs/quant/lib/python3.9/site-packages/nasdaqdatalink/api_config.py"
"""


def retrieve_historical_stocks(start_date: str = '2012-01-01') -> pd.DataFrame:
    """
    Retrieve historical stocks via Nasdaq Data Link based on American Stocks

    Exchanges: ['NYSE', 'NASDAQ', 'NYSEMKT']

    Type of Stocks: ['Domestic Common Stock', 'ADR Common Stock',
                  'Domestic Common Stock Primary Class', 'Canadian Common Stock',
                  'ADR Common Stock Primary Class',
                  'Canadian Common Stock Primary Class',
                  'ADR Common Stock Secondary Class',
                  'Domestic Common Stock Secondary Class']

    By default, data will be save as a tuple format via pickle I/O functions
    Tuple Format: (data_long_format, data_wide_format, available tickers)

    data_long_format columns: ['ticker', 'date', 'open', 'high', 'low', 'close', 'openadj', 'highadj',
                                    'lowadj', 'closeadj', 'volume']]

    data_wide_format columns: ['date', 'open', 'high', 'low', 'close',
                            'openadj', 'highadj', 'lowadj', 'closeadj', 'volume']

    :return:  tuple(pd.DataFrame, pd.DataFrame, list, list)
    """

    # Get ticket information: list of tickers
    logger.info("Fetching ticker metadata")
    tickers_metadata = ndl.get_table('SHARADAR/TICKERS', paginate=True)
    logger.info("Done")

    # Filtering exchanges and categories of stocks
    filter_exchange = NDL_CONFIG['exchanges']

    filter_cat = NDL_CONFIG['stock_categories']
    # Filter stocks that we focus on
    focused_stocks_ticker = tickers_metadata[
        (tickers_metadata.exchange.isin(filter_exchange)) & (
            tickers_metadata.category.isin(filter_cat))].ticker.unique()

    n_total_tickers = len(focused_stocks_ticker)
    logger.info(f"Total tickers: {n_total_tickers}")

    # Retrieve data via for loop (might be a more efficient way to do this)
    stacked_data = []
    # stacked_dict = {}
    s_time_chunk = time.time()
    logger.info("Fetching historical data")

    batch_size = 500
    for i in range(0, n_total_tickers, batch_size):

        fetch_list = focused_stocks_ticker[i:i + batch_size]

        try_cnt = 0
        while try_cnt <= 20:
            try:
                data = ndl.get_table('SHARADAR/SEP', ticker=fetch_list,
                                     paginate=True, date={'gte': start_date})
                break
            # in case of network failure, try again after 5 seconds for 20 tries
            except ChunkedEncodingError as ex:
                if try_cnt > 20:
                    logger.info(f'ChunkedEncodingError (after trying multiple times): {ex}')
                    logger.info(f"Failed to fetch {fetch_list}")
                    break
                else:
                    try_cnt += 1
                    time.sleep(5)
                    logger.info("Failed: Retrying")

            except Exception as ex:
                if try_cnt > 20:
                    logger.info(f'All Other Exception Occurred (after trying multiple times): {ex}')
                    logger.info(f"Failed to fetch {fetch_list}")
                    break
                else:
                    try_cnt += 1
                    time.sleep(5)
                    logger.info("Failed: Retrying")

        data['adj_multiplier'] = data.closeadj / data.close
        data['openadj'] = data.open * data.adj_multiplier
        data['highadj'] = data.high * data.adj_multiplier
        data['lowadj'] = data.low * data.adj_multiplier
        data = data[
            ['ticker', 'date', 'open', 'high', 'low', 'close', 'openadj', 'highadj', 'lowadj', 'closeadj', 'volume']]
        data.sort_values('date', inplace=True)
        data.set_index('date', inplace=True)
        stacked_data.append(data)

        # condition for unavailable data, as some of them are probably not available before 2012 due to subscription
        # package limits
        # if data.shape[0] > 0:
        #     stacked_dict[ticker] = data

        if i % 100 == 0:
            logger.info(f'Tickers iterated: {i}, Progress: {round(i / n_total_tickers * 100, 2)}%')

    e_time_chunk = time.time()

    logger.info("Done")
    logger.info(f"Total download time: {e_time_chunk - s_time_chunk} sec")

    # Concat into one dataframe
    stacked_hist = pd.concat(stacked_data)
    stacked_hist = stacked_hist.reset_index().sort_values(['ticker', 'date']).set_index('date')

    return stacked_hist


def update_historical_data(existing_hist: pd.DataFrame) -> pd.DataFrame:
    logger.info("Updating Data")
    latest_data_date = existing_hist.index.max()
    next_avai_date = latest_data_date + relativedelta(days=1)

    new_hist = retrieve_historical_stocks(next_avai_date.strftime("%Y-%m-%d"))

    # if no new data, exit the code
    if new_hist.shape[0] == 0:
        raise Exception("No New Data Available, Data Update Process Stopped")
    new_stacked_hist = pd.concat([existing_hist, new_hist])
    new_stacked_hist = new_stacked_hist.reset_index().sort_values(['ticker', 'date']).set_index('date')

    return new_stacked_hist


def process_historical_dataframe(stacked_hist: pd.DataFrame) -> Tuple:
    logger.info("Processing Data")
    ohlcv_columns = ['open', 'high', 'low', 'close', 'openadj', 'highadj', 'lowadj',
                     'closeadj', 'volume', 'ret', 'retvol']

    stacked_hist_wide = stacked_hist.reset_index().pivot(index='date', columns='ticker', values=ohlcv_columns)
    stacked_hist_wide.columns = [f'{j}_{i}' for i, j in stacked_hist_wide.columns]

    available_tickers = stacked_hist.ticker.unique()

    return stacked_hist, stacked_hist_wide, available_tickers


def extend_dataframe(stacked_hist: pd.DataFrame) -> Tuple:
    """
    Extends the historical data (wide format) to have other stats like returns, vol, and is_active values

    :param traded: list
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """

    stacked_hist = stacked_hist.reset_index().sort_values(['ticker', 'date']).set_index('date')
    logger.info("Extending DataFrame")

    # lets get return statistics using closing prices
    # and volatility statistics using rolling standard deviations of 25 day window
    # lets also see if a stock is being actively traded, by seeing if closing price today != yesterday
    stacked_hist['ret'] = stacked_hist.groupby("ticker")['closeadj'].apply(lambda x: x / x.shift() - 1)
    stacked_hist['retvol'] = stacked_hist.groupby("ticker").ret.rolling(25).std().values
    stacked_hist, stacked_hist_wide, available_tickers = process_historical_dataframe(stacked_hist)
    # get is_active columns
    close_df = stacked_hist.reset_index().pivot(index='date', columns='ticker', values='closeadj')
    is_active = ~close_df.isnull()
    is_active.columns = [f'{col}_active' for col in is_active.columns]
    stacked_hist_wide = pd.concat([stacked_hist_wide, is_active], axis=1)

    traded = available_tickers
    # formats date
    stacked_hist_wide.index = pd.Series(stacked_hist_wide.index).apply(lambda x: format_date(x))
    open_cols = list(map(lambda x: f'{x}_open', traded))
    high_cols = list(map(lambda x: f'{x}_high', traded))
    low_cols = list(map(lambda x: f'{x}_low', traded))
    close_cols = list(map(lambda x: f'{x}_close', traded))
    openadj_cols = list(map(lambda x: f'{x}_openadj', traded))
    highadj_cols = list(map(lambda x: f'{x}_highadj', traded))
    lowadj_cols = list(map(lambda x: f'{x}_lowadj', traded))
    closeadj_cols = list(map(lambda x: f'{x}_closeadj', traded))
    volume_cols = list(map(lambda x: f'{x}_volume', traded))
    ret_cols = list(map(lambda x: f'{x}_ret', traded))
    retvol_cols = list(map(lambda x: f'{x}_retvol', traded))
    active_cols = list(map(lambda x: f'{x}_active', traded))

    # Pay attention to how we fill the data with nan values, for example we bfill and ffill ohlcv but we bfill and ffill
    # ret and retvol with 0, the function code is different from HanggukQuant's, and we are not sure if this would
    # affect the performance results or not, but we hope it does not. Hopefully, the active columns should be a good
    # indicator in preventing us from selecting inactive stocks
    historical_data = stacked_hist_wide.copy()
    historical_ohlcv_data = historical_data[open_cols + high_cols + low_cols + close_cols + openadj_cols +
                                            highadj_cols + lowadj_cols + closeadj_cols + volume_cols]

    historical_retvol_data = historical_data[ret_cols + retvol_cols]
    historical_active_data = historical_data[active_cols]

    historical_retvol_data.fillna(0, inplace=True)

    historical_ohlcv_data.fillna(method="ffill", inplace=True)

    historical_data = pd.concat([historical_ohlcv_data, historical_retvol_data, historical_active_data], axis=1)

    historical_data.fillna(method="bfill", inplace=True)

    return stacked_hist, historical_data, available_tickers
