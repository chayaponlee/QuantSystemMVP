import os
import logging
from realgam.quantlib import data_utils_ver2 as du, general_utils as gu, qlogger
from realgam.quantlib.engineer.op_engineer_vect import OpEngineerV
from realgam.quantlib.engineer.ta_engineer_vect import TalibEngineerV
import time
import warnings

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier


"""
Test1
With f_alpha_pre_20220330.obj, it takes 45 minutes to run 
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', criterion='entropy', n_jobs=-1)
scores = cross_validate(clf, X, y, cv=5, scoring=('accuracy', 'f1', 'precision', 'recall'), return_train_score=True)
"""

warnings.filterwarnings("ignore")

logger = qlogger.init(__file__, logging.INFO)
PROJECT_PATH = os.getenv('QuantSystemMVP')

FEAT = [f'alpha{i + 1}' for i in range(6)]
TARGET_COL = 'ret6_fwd'

FEAT_PATH = f'{PROJECT_PATH}/Data/feature/f_alpha_pre_20220330.obj'
SAVE_PATH = f'{PROJECT_PATH}/Data/model_results/RF_score_test_20220330.obj'

if __name__ == '__main__':

    eng_pre = gu.load_file(FEAT_PATH)
    # eng_pre = eng_pre.head(500000)
    X = eng_pre[FEAT]
    y = eng_pre[TARGET_COL]
    logger.info(f'Rows: {X.shape[0]}')
    # logger.info("Begin CV without multicores")
    # s_time_chunk = time.time()
    # clf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', criterion='entropy')
    # scores = cross_validate(clf, X, y, cv=5, scoring=('accuracy', 'f1', 'precision', 'recall'), return_train_score=True)
    # e_time_chunk = time.time()
    # logger.info(f"Total time: {e_time_chunk - s_time_chunk} sec")

    logger.info("Begin CV with max cores")
    s_time_chunk = time.time()
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', criterion='entropy', n_jobs=-1)
    scores = cross_validate(clf, X, y, cv=5, scoring=('accuracy', 'f1', 'precision', 'recall'), return_train_score=True)
    e_time_chunk = time.time()
    logger.info(f"Total time: {(e_time_chunk - s_time_chunk)/60} min")

    time_taken = (e_time_chunk - s_time_chunk)/60
    gu.save_file(SAVE_PATH, (scores, time_taken))

