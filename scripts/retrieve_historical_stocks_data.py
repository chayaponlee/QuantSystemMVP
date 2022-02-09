import sys
import os
import logging
from quantlib.data_utils import retrieve_historical_stocks
from quantlib import general_utils as gu
from quantlib import qlogger

logger = qlogger.init(__file__, logging.INFO)
project_path = os.getenv('QuantSystemMVP')


if __name__ == '__main__':
    stocks_data = retrieve_historical_stocks()

    save_path = f'{project_path}/Data/stock_hist.obj'
    gu.save_file(save_path, stocks_data)

    logger.info(f"Historical Stock data written to {save_path}")
