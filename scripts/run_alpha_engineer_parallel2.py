import os
import logging
from realgam.quantlib import data_utils_ver2 as du, general_utils as gu, qlogger
from joblib import Parallel, delayed
import warnings

import pandas as pd
import numpy as np
from realgam.quantlib.engineer.alpha_engineer import AlphaEngineer
from realgam.quantlib.engineer.op_functions import *
from realgam.quantlib.engineer.op_engineer_vect import OpEngineerV
import time

warnings.filterwarnings("ignore")

logger = qlogger.init(__file__, logging.INFO)
PROJECT_PATH = os.getenv('QuantSystemMVP')

DATA_PATH = f'{PROJECT_PATH}/Data/historical/stock_hist_perma.obj'
SAVE_PATH = f'{PROJECT_PATH}/Data/engineered/e_alpha_20220401.obj'

alpha_eng_index = [i for i in range(21,31,1)]

if __name__ == '__main__':
    logger.info('Loading data')
    stocks_df, _, _ = gu.load_file(DATA_PATH)
    stacked_hist = stocks_df.copy()

    openg = OpEngineerV(stacked_hist, 'permaticker', 'date')
    openg.ts_ret(inplace=True)
    uni_df = openg.df.copy()
    uni_df.rename(columns={'ts_ret_closeadj': 'returns'}, inplace=True)

    candle_name_dict = {'o': 'openadj', 'h': 'highadj', 'l': 'lowadj',
                        'c': 'closeadj', 'v': 'volume', 'r': 'returns'}

    ae = AlphaEngineer(uni_df, 'permaticker', 'date', candle_name_dict)


    def run_eng(alpha_engineer, method_name):

        return getattr(alpha_engineer, method_name)()


    # alpha_eng_index = [i for i in range(1, 31, 1)]
    # alpha_eng_index = [1,2]
    alpha_eng_names = [f'alpha{index}' for index in alpha_eng_index]

    logger.info('Begin')
    s_time_chunk = time.time()
    try:
        eng_values = Parallel(n_jobs=-1)(delayed(run_eng)(ae, eng_name) for eng_name in alpha_eng_names)
    except Exception as e:
        logger.info(f'Error: {e}')

    e_time_chunk = time.time()
    logger.info(f"Total time: {e_time_chunk - s_time_chunk} sec")

    s_time_chunk = time.time()
    for eng_name, eng_value in zip(alpha_eng_names, eng_values):
        uni_df[eng_name] = eng_value
    e_time_chunk = time.time()
    logger.info(f"Total time: {e_time_chunk - s_time_chunk} sec")

    print(uni_df.columns)
    # gu.save_file(SAVE_PATH, stacked_hist)

    logger.info(f"Alpha engineered data written to {SAVE_PATH}")
