import quantlib.data_utils as du
import quantlib.general_utils as gu

from dateutil.relativedelta import relativedelta

from subsystems.lbmom.subsys import Lbmom

import os
import logging
from quantlib import qlogger

logger = qlogger.init(__file__, logging.INFO)
PROJECT_PATH = os.getenv('QuantSystemMVP')

DATA_PATH = f'{PROJECT_PATH}/Data/stock_hist.obj'
CONFIG_PATH = f"{DATA_PATH}/subsystems/lbmom/config.json"

# we might want to later put VOL_TARGET in the config file
VOL_TARGET = 0.20

if __name__ == '__main__':

    # df, instruments = du.get_sp500_df()
    # df = du.extend_dataframe(traded=instruments, df=df)

    stocks_df, stocks_wide_df, stocks_extended_df, available_tickers = gu.load_file(f"{DATA_PATH}/Data/stock_hist.obj")
    print(available_tickers)

    # lets run the lbmom strategy through the driver.

      # we are targetting 20% annualized vol

    # let's perform the simulation for the past 5 years

    print(stocks_extended_df.index[-1])  # is today's date. (as I film) 2022-01-19
    # I want to start testing from 5 years back

    sim_start = stocks_extended_df.index[-1] - relativedelta(years=5)
    print(sim_start)

    strat = Lbmom(instruments_config=CONFIG_PATH, historical_df=stocks_extended_df,
                  simulation_start=sim_start, vol_target=VOL_TARGET)

    strat.get_subsys_pos()
