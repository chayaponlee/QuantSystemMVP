import sys
import os
import logging
from quantlib.data_utils import retrieve_historical_stocks, extend_dataframe
from quantlib import general_utils as gu
from quantlib import qlogger

logger = qlogger.init(__file__, logging.INFO)
project_path = os.getenv('QuantSystemMVP')


if __name__ == '__main__':
    stocks_df, stocks_wide_df, available_tickers = retrieve_historical_stocks()
    stocks_extended_df = extend_dataframe( available_tickers, stocks_wide_df)

    save_path = f'{project_path}/Data/stock_hist.obj'
    gu.save_file(save_path, (stocks_df, stocks_wide_df, stocks_extended_df, available_tickers))

    logger.info(f"Historical Stock data written to {save_path}")
