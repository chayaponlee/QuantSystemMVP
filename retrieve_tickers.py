import pandas as pd
import nasdaqdatalink as ndl
import time
import logging as log
from requests.exceptions import ChunkedEncodingError

log.basicConfig(level='INFO')

"""
Retrieve American stock data

Configuring nasdaqdatalink API: depending on your directory, you have to enter the API key in to the config file
For MacAir: you can configure the file at 
"/opt/homebrew/Caskroom/miniforge/base/envs/quant/lib/python3.9/site-packages/nasdaqdatalink/api_config.py"
"""

# Get ticket information: list of tickers
log.info("Fetching ticker metadata")
tickers_metadata = ndl.get_table('SHARADAR/TICKERS', paginate=True)
log.info("Done")

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
    (tickers_metadata.exchange.isin(filter_exchange)) & (tickers_metadata.category.isin(filter_cat))].ticker.unique()

log.info(f"Total tickers: {len(list(focused_stocks_ticker))}")

# Retrieve data via for loop (might be a more efficient way to do this)
stacked_data = []

s_time_chunk = time.time()
log.info("Fetching historical data")
for i, ticker in enumerate(focused_stocks_ticker):
    # log.info(i)
    try_cnt = 0
    while try_cnt <= 20:
        try:
            data = ndl.get_table('SHARADAR/SEP', ticker=ticker, paginate=True, date={'gte': '2012-01-01'})
            break
        except ChunkedEncodingError as ex:
            if try_cnt > 20:
                log.info(f'Error (after trying multiple times): {ex}')
                log.info(f"Failed to fetch {ticker}")
                break
            else:
                try_cnt += 1
                time.sleep(5)
                log.info("Failed: Retrying")

    stacked_data.append(data)
    if i % 100 == 0 and i != 0:
        log.info(f'Tickers iterated: {i}')
e_time_chunk = time.time()

log.info("Done")
log.info("Total download time: ", (e_time_chunk - s_time_chunk), "sec")

# Concat into one dataframe
stacked_hist = pd.concat(stacked_data)
stacked_hist.sort_values(['ticker', 'date'], inplace=True)

stacked_hist.to_csv('Data/stacked_hist.csv', index=False)
