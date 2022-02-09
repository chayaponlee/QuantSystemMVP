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
    By default, data will be save as a tuple format via pickle I/O functions
    Tuple Format: (data, ticker universe, actual available tickers)
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

    logger.info(f"Total tickers: {len(focused_stocks_ticker)}")

    # Retrieve data via for loop (might be a more efficient way to do this)
    stacked_data = []

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
                    logger.info(f'Error (after trying multiple times): {ex}')
                    logger.info(f"Failed to fetch {ticker}")
                    break
                else:
                    try_cnt += 1
                    time.sleep(5)
                    logger.info("Failed: Retrying")

        stacked_data.append(data)
        if i % 100 == 0 and i != 0:
            logger.info(f'Tickers iterated: {i}')
    e_time_chunk = time.time()

    logger.info("Done")
    logger.info(f"Total download time: {e_time_chunk - s_time_chunk} sec")

    # Concat into one dataframe
    stacked_hist = pd.concat(stacked_data)
    stacked_hist.sort_values(['ticker', 'date'], inplace=True)

    # actual downloaded tickers, as some of them are probably not available before 2012 due to subscription package
    # limits
    available_tickers = stacked_hist.ticker.unique()

    return stacked_hist, focused_stocks_ticker, available_tickers

