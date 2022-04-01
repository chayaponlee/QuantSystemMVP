import os
import logging
from realgam.quantlib import data_utils_ver2 as du, general_utils as gu, qlogger
import warnings

import pandas as pd
import numpy as np
from realgam.quantlib.engineer.alpha_engineer_vect import AlphaEngineerV
import time

warnings.filterwarnings("ignore")

logger = qlogger.init(__file__, logging.INFO)
PROJECT_PATH = os.getenv('QuantSystemMVP')

DATA_PATH = f'{PROJECT_PATH}/Data/historical/stock_hist_perma.obj'
SAVE_PATH = f'{PROJECT_PATH}/Data/test/test_alpha_2022-03-28.obj'

if __name__ == '__main__':
    logger.info('Loading data')
    stocks_df, _, _ = gu.load_file(DATA_PATH)
    stacked_hist = stocks_df.copy()

    tickers = stacked_hist.index.get_level_values('permaticker').unique()[:200]
    stacked_hist = stacked_hist[stacked_hist.index.isin(tickers, level='permaticker')]

    ae = AlphaEngineerV(stacked_hist)

    stot_time_chunk = time.time()

    logger.info('Begin')
    s_time_chunk = time.time()
    try:
        stacked_hist['alpha1'] = ae.alpha1()
    except Exception:
        logger.info('Error')

    e_time_chunk = time.time()
    logger.info(f"Total time: {e_time_chunk - s_time_chunk} sec")

    logger.info('Begin')
    s_time_chunk = time.time()
    try:
        stacked_hist['alpha2'] = ae.alpha2()
    except Exception:
        logger.info('Error')

    e_time_chunk = time.time()
    logger.info(f"Total time: {e_time_chunk - s_time_chunk} sec")

    logger.info('Begin')
    s_time_chunk = time.time()
    try:
        stacked_hist['alpha3'] = ae.alpha3()
    except Exception:
        logger.info('Error')

    e_time_chunk = time.time()
    logger.info(f"Total time: {e_time_chunk - s_time_chunk} sec")

    logger.info('Begin')
    s_time_chunk = time.time()
    try:
        stacked_hist['alpha4'] = ae.alpha4()
    except Exception:
        logger.info('Error')

    e_time_chunk = time.time()
    logger.info(f"Total time: {e_time_chunk - s_time_chunk} sec")

    logger.info('Begin')
    s_time_chunk = time.time()
    try:
        stacked_hist['alpha5'] = ae.alpha5()
    except Exception:
        logger.info('Error')

    e_time_chunk = time.time()
    logger.info(f"Total time: {e_time_chunk - s_time_chunk} sec")

    logger.info('Begin')
    s_time_chunk = time.time()
    try:
        stacked_hist['alpha6'] = ae.alpha6()
    except Exception:
        logger.info('Error')

    e_time_chunk = time.time()
    logger.info(f"Total time: {e_time_chunk - s_time_chunk} sec")

    logger.info('Begin')
    s_time_chunk = time.time()
    try:
        stacked_hist['alpha7'] = ae.alpha7()
    except Exception:
        logger.info('Error')

    e_time_chunk = time.time()
    logger.info(f"Total time: {e_time_chunk - s_time_chunk} sec")

    etot_time_chunk = time.time()
    logger.info(f"Total time: {etot_time_chunk - stot_time_chunk} sec")

    # gu.save_file(SAVE_PATH, stacked_hist)

    logger.info(f"Alpha engineered data written to {SAVE_PATH}")
