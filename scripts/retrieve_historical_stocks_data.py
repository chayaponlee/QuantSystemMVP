import os
import logging
from realgam.quantlib import data_utils as du, general_utils as gu, qlogger
import warnings

warnings.filterwarnings("ignore")

logger = qlogger.init(__file__, logging.INFO)
PROJECT_PATH = os.getenv('QuantSystemMVP')

SAVE_PATH = f'{PROJECT_PATH}/Data/stock_hist.obj'

if __name__ == '__main__':
    stocks_df = du.retrieve_historical_stocks()

    stocks_df, stocks_extended_df, available_tickers = du.extend_dataframe(stocks_df)

    gu.save_file(SAVE_PATH, (stocks_df, stocks_extended_df, available_tickers))

    logger.info(f"Historical Stock data written to {SAVE_PATH}")
