import os
import logging
from realgam.quantlib import data_utils_ver2 as du, general_utils as gu, qlogger
import warnings

warnings.filterwarnings("ignore")

logger = qlogger.init(__file__, logging.INFO)
PROJECT_PATH = os.getenv('QuantSystemMVP')

DATA_PATH = f'{PROJECT_PATH}/Data/stock_hist_perma.obj'
SAVE_PATH = f'{PROJECT_PATH}/Data/stock_hist_perma.obj'

if __name__ == '__main__':

    (old_stocks_df, old_stocks_extended_df, old_available_tickers) = gu.load_file(DATA_PATH)
    stocks_df = du.update_historical_data(old_stocks_df)

    stocks_extended_df, available_tickers = du.extend_dataframe(stocks_df)

    gu.save_file(SAVE_PATH, (stocks_df, stocks_extended_df, available_tickers))

    logger.info(f"Historical Stock data written to {SAVE_PATH}")
