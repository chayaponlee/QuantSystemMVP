import os
import logging
from realgam.quantlib import data_utils_ver2 as du, general_utils as gu, qlogger
from joblib import Parallel, delayed
import warnings

import pandas as pd
import numpy as np
from realgam.quantlib.engineer.alpha_engineer_vect import AlphaEngineerV
import time

warnings.filterwarnings("ignore")

logger = qlogger.init(__file__, logging.INFO)
PROJECT_PATH = os.getenv('QuantSystemMVP')

DATA_PATH = f'{PROJECT_PATH}/Data/historical/stock_hist_perma.obj'
SAVE_PATH = f'{PROJECT_PATH}/Data/engineered/e_alpha_20220401.obj'

alpha_eng_index = [1, 2, 3, 4, 5, 6, 7]

if __name__ == '__main__':
    logger.info('Loading data')
    stocks_df, _, _ = gu.load_file(DATA_PATH)
    stacked_hist = stocks_df.copy()

    # tickers = stacked_hist.index.get_level_values('permaticker').unique()[:200]
    # stacked_hist = stacked_hist[stacked_hist.index.isin(tickers, level='permaticker')]

    ae = AlphaEngineerV(stacked_hist)

    def run_eng(method_name):

        return getattr(ae, method_name)()

    alpha_eng_names = [f'alpha{index}' for index in alpha_eng_index]

    logger.info('Begin')
    s_time_chunk = time.time()
    try:
        eng_values = Parallel(n_jobs=-1)(delayed(run_eng)(eng_name) for eng_name in alpha_eng_names)
    except Exception as e:
        logger.info(f'Error: {e}')

    e_time_chunk = time.time()
    logger.info(f"Total time: {e_time_chunk - s_time_chunk} sec")

    s_time_chunk = time.time()
    for eng_name, eng_value in zip(alpha_eng_names, eng_values):
        stacked_hist[eng_name] = eng_value
    e_time_chunk = time.time()
    logger.info(f"Total time: {e_time_chunk - s_time_chunk} sec")

    print(stacked_hist.columns)
    # gu.save_file(SAVE_PATH, stacked_hist)

    logger.info(f"Alpha engineered data written to {SAVE_PATH}")
