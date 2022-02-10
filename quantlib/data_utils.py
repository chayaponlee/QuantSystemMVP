import pandas as pd
import nasdaqdatalink as ndl
import time
import datetime
import logging
from requests.exceptions import ChunkedEncodingError
from quantlib import qlogger

# log.basicConfig(level='INFO')
logger = qlogger.init(__file__, logging.INFO)

"""
when obtaining data from numerous sources, we want to standardize communication units.
in other words, we want object types to be the same. for instance, things like dataframe index 'type' or 'class'
should be the same
"""


def format_date(dates):
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


def retrieve_historical_stocks():
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
    Tuple Format: (data_long_format, data_wide_format, ticker universe, actual available tickers)

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
    filter_exchange = ['NYSE', 'NASDAQ', 'NYSEMKT']

    filter_cat = ['Domestic Common Stock', 'ADR Common Stock',
                  'Domestic Common Stock Primary Class', 'Canadian Common Stock',
                  'ADR Common Stock Primary Class',
                  'Canadian Common Stock Primary Class',
                  'ADR Common Stock Secondary Class',
                  'Domestic Common Stock Secondary Class']

    # Filter stocks that we focus on
    focused_stocks_ticker = tickers_metadata[
        (tickers_metadata.exchange.isin(filter_exchange)) & (
            tickers_metadata.category.isin(filter_cat))].ticker.unique()

    n_total_tickers = len(focused_stocks_ticker)
    logger.info(f"Total tickers: {n_total_tickers}")

    # Retrieve data via for loop (might be a more efficient way to do this)
    stacked_data = []
    stacked_dict = {}
    s_time_chunk = time.time()
    logger.info("Fetching historical data")
    for i, ticker in enumerate(focused_stocks_ticker):

        try_cnt = 0
        while try_cnt <= 20:
            try:
                data = ndl.get_table('SHARADAR/SEP', ticker=ticker, paginate=True, date={'gte': '2012-01-01'})
                break
            # in case of network failure, try again after 5 seconds for 20 tries
            except ChunkedEncodingError as ex:
                if try_cnt > 20:
                    logger.info(f'ChunkedEncodingError (after trying multiple times): {ex}')
                    logger.info(f"Failed to fetch {ticker}")
                    break
                else:
                    try_cnt += 1
                    time.sleep(5)
                    logger.info("Failed: Retrying")

            except Exception as ex:
                if try_cnt > 20:
                    logger.info(f'All Other Exception Occurred (after trying multiple times): {ex}')
                    logger.info(f"Failed to fetch {ticker}")
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
        if data.shape[0] > 0:
            stacked_dict[ticker] = data

        if (i + 1) % 100 == 0:
            logger.info(f'Tickers iterated: {i + 1}, Progress: {round((i + 1) / n_total_tickers * 100, 2)}%')

    e_time_chunk = time.time()

    logger.info("Done")
    logger.info(f"Total download time: {e_time_chunk - s_time_chunk} sec")

    # Concat into one dataframe
    stacked_hist = pd.concat(stacked_data)

    # We also create a wide dataframe
    stacked_hist_wide = pd.DataFrame(index=stacked_hist[stacked_hist.ticker == 'AMZN'].index)
    stacked_hist_wide.index.name = "date"
    # actual downloaded tickers, as some of them are probably not available before 2012 due to subscription package
    # limits
    available_tickers = list(stacked_dict.keys())

    for ticker in available_tickers:
        inst_df = stacked_dict[ticker][['open', 'high', 'low', 'close',
                                        'openadj', 'highadj', 'lowadj', 'closeadj', 'volume']]
        # add an identifier to the columns
        columns = list(map(lambda x: f'{ticker}_{x}', inst_df.columns))
        # this adds the instrument name to each column
        stacked_hist_wide[columns] = inst_df

    return stacked_hist, stacked_hist_wide, available_tickers


def extend_dataframe(traded, df):
    """
    Extends the historical data (wide format) to have other stats like returns, vol, and is_active values

    :param traded: list
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    # formats date
    df.index = pd.Series(df.index).apply(lambda x: format_date(x))
    open_cols = list(map(lambda x: f'{x}_open', traded))
    high_cols = list(map(lambda x: f'{x}_high', traded))
    low_cols = list(map(lambda x: f'{x}_low', traded))
    close_cols = list(map(lambda x: f'{x}_close', traded))
    openadj_cols = list(map(lambda x: f'{x}_openadj', traded))
    highadj_cols = list(map(lambda x: f'{x}_highadj', traded))
    lowadj_cols = list(map(lambda x: f'{x}_lowadj', traded))
    closeadj_cols = list(map(lambda x: f'{x}_closeadj', traded))
    volume_cols = list(map(lambda x: f'{x}_volume', traded))
    historical_data = df.copy()
    historical_data = historical_data[open_cols + high_cols + low_cols + close_cols + openadj_cols +
                                      highadj_cols + lowadj_cols + closeadj_cols + volume_cols]
    historical_data.fillna(method="ffill", inplace=True)
    for inst in traded:
        # lets get return statistics using closing prices
        # and volatility statistics using rolling standard deviations of 25 day window
        # lets also see if a stock is being actively traded, by seeing if closing price today != yesterday
        historical_data[f'{inst}_ret'] = historical_data[f'{inst}_closeadj'] \
                                         / historical_data[f'{inst}_closeadj'].shift(1) - 1
        historical_data[f'{inst}_retvol'] = historical_data[f'{inst}_ret'].rolling(25).std()
        historical_data[f'{inst}_active'] = historical_data[f'{inst}_closeadj'] \
                                            != historical_data[f'{inst}_closeadj'].shift(1)
    historical_data.fillna(method="bfill", inplace=True)
    return historical_data
