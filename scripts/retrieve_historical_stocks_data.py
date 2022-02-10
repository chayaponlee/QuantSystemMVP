import sys
import os
import logging
from quantlib import data_utils as du
from quantlib import general_utils as gu
from quantlib import qlogger

logger = qlogger.init(__file__, logging.INFO)
PROJECT_PATH = os.getenv('QuantSystemMVP')

SAVE_PATH = f'{PROJECT_PATH}/Data/stock_hist.obj'

if __name__ == '__main__':
    stocks_df, stocks_wide_df, available_tickers = du.retrieve_historical_stocks()
    stocks_extended_df = du.extend_dataframe(available_tickers, stocks_wide_df)

    gu.save_file(SAVE_PATH, (stocks_df, stocks_wide_df, stocks_extended_df, available_tickers))

    logger.info(f"Historical Stock data written to {SAVE_PATH}")
