import json
import quantlib.indicators_cal as indicators_cal
import quantlib.backtest_utils as backtest_utils
import numpy as np
import pandas as pd
from datetime import datetime

import logging
import quantlib.qlogger as qlogger
logger = qlogger.init(__file__, logging.INFO)

class Lbmom():
    """
    Long biased momentum strategy
    """

    def __init__(self, instruments_config, historical_df, simulation_start, vol_target, backtest_dir_path):
        self.ma_pairs = [(23, 82), (44, 244), (124, 294), (37, 229), (70, 269), (158, 209), (81, 169), (184, 203),
                         (23, 265), (244, 268), (105, 106), (193, 250), (127, 294), (217, 274), (45, 178),
                         (103, 288), (204, 248), (142, 299), (71, 216), (129, 148), (149, 218)]
        self.n_adx = 14
        self.historical_df = historical_df
        self.simulation_start = simulation_start
        self.vol_target = vol_target
        with open(instruments_config) as f:
            self.instruments_config = json.load(f)
        self.sysname = 'LBMOM'
        self.backtest_dir_path = backtest_dir_path

    # we implement a few functions
    # 1. A function to get extra indicators specific to this strategy
    # 2. A function to run a backtest/get positions from this strategy

    def extend_historicals(self, instruments, historical_data):
        # we need indicators of `momentum`
        # let this be the moving average crossover, such that if the fastMA crossover the slowMA, then it is a buy
        # a long-biased momentum strategy is biased in the long direction. let this be a 100/0 L/S strategy.
        # let's also use a filter, to identify false positive signals. We use the average directional index, or the adx

        for inst in instruments:
            historical_data[f'{inst}_adx{self.n_adx}'] = indicators_cal.adx_series(
                high=historical_data[f'{inst}_highadj'],
                low=historical_data[f'{inst}_lowadj'],
                close=historical_data[f'{inst}_closeadj'],
                n=self.n_adx
            )
            for pair in self.ma_pairs:
                historical_data[f'{inst}_ema{pair}'] = indicators_cal.ema_series(
                    series=historical_data[f'{inst}_closeadj'],
                    n=pair[0]
                ) - indicators_cal.ema_series(
                    series=historical_data[f'{inst}_closeadj'],
                    n=pair[1]
                )  # fastMA - slowMA
        # now historical data has ohlcvs, and whether the fastMA - slowMA for each pair,
        # and the adx of the closing prices to see if there is a `trending` regime
        return historical_data

    def run_backtest(self, historical_data):
        # init parameters
        instruments = self.instruments_config["instruments"]

        # calculate/pre-process indicators
        historical_data = self.extend_historicals(instruments=instruments, historical_data=historical_data)

        # perform simulation
        portfolio_df = pd.DataFrame(index=historical_data[self.simulation_start:].index).reset_index()
        # wonderful technique in creating a column with assigned first value only
        portfolio_df.loc[0, 'capital'] = 10000

        # date_less1 is needed to ensure that the second condition returns dataframe 5 days before 'date'
        # if we use 'date', then it returns the dataframe including 'date' as well, which is weird
        # condition, if the instrument has been inactive for the past 5 days before current date then consider
        # halting trading for the instrument
        is_halted = lambda inst, arg_date, arg_date_prev: not np.isnan(
            historical_data.loc[arg_date, f'{inst}_active']) and \
                                                          (~historical_data[:arg_date_prev].tail(5)[
                                                              f'{inst}_active']).all()
        """
        Position-sizing with 3 different techniques combined
        1. Strategy Level Scalar for strategy level risk exposure
        2. Volatility targeting scalar for different assets
        3. Voting systems to account for degree of 'momentum'
        """
        logger.info("======= Begin Backtesting and Performing Position value filling =======")
        total_days = len(portfolio_df.index)

        for i in portfolio_df.index:
            date = portfolio_df.loc[i, 'date']
            strat_scalar = 2

            """
            Get PnL and Scalar for Portfolio
            """
            if i > 0:
                date_prev = portfolio_df.loc[i - 1, 'date']
                pnl = backtest_utils.get_backtest_day_stats(
                    portfolio_df, instruments, date, date_prev, i, historical_data)
                strat_scalar = backtest_utils.get_strat_scaler(
                    portfolio_df, 100, self.vol_target, i, strat_scalar)
            else:
                date_prev = date

            portfolio_df.loc[i, "strat scalar"] = strat_scalar

            tradable = [inst for inst in instruments if not is_halted(inst, date, date_prev)]
            non_tradable = [inst for inst in instruments if inst not in tradable]


            """
            Get Positions for Traded Instruments, Assign 0 to Non-Traded
            """
            for inst in non_tradable:
                portfolio_df.loc[i, f'{inst}_units'] = 0
                portfolio_df.loc[i, f'{inst}_w'] = 0

            nominal_total = 0
            for inst in tradable:
                # calculate trades

                # these votes are to collect the signals from each ma_pair, we have 21 ma_pairs so
                # intuitively, the more votes, stronger the signals, this is votes per day per instrument
                votes = np.sum(
                    [1 for pair in self.ma_pairs if historical_data.loc[date, f'{inst}_ema{pair}'] > 0])
                forecast = votes / len(self.ma_pairs)
                # apply trending filter using adx, if lower than 25 then overwrite with no signal
                forecast = 0 if historical_data.loc[date, f'{inst}_adx{self.n_adx}'] < 25 else forecast

                # volatility targetting
                position_vol_target = (1 / len(tradable)) + portfolio_df.loc[i, 'capital'] + \
                                      self.vol_target / np.sqrt(253)
                inst_price = historical_data.loc[date, f'{inst}_closeadj']

                # notice here we use 'date' when we check for last 25 active days, we want to include current date
                # to check as well, but be careful this might run into look forward bias depending on the strat
                # Here, we are checking if the ticker is active for last 25 days or not, if it is use ret vol,
                # but if not, we use 0.025 because if the stock is not active, the calculation may run into weird
                # std values, so we want to put it as an arbritary number
                percent_ret_vol = historical_data.loc[date, f'{inst}_retvol'] \
                    if historical_data[:date].tail(25)[f'{inst}_active'].all() else 0.025

                dollar_volatility = inst_price * percent_ret_vol # vol in dollar terms
                position = strat_scalar * forecast * self.vol_target / dollar_volatility
                portfolio_df.loc[i, f'{inst}_units'] = position

                nominal_total += abs(position * inst_price) # assuming all denominated in same currency

            for inst in tradable:
                units = portfolio_df.loc[i, f"{inst}_units"]
                nominal_inst = abs(units * historical_data.loc[date, f"{inst}_closeadj"])
                inst_w = nominal_inst / nominal_total
                portfolio_df.loc[i, f"{inst}_w"] = inst_w

            """
            Perform Logging and Calculations
            """
            portfolio_df.loc[i, "nominal"] = nominal_total
            portfolio_df.loc[i, "leverage"] = portfolio_df.loc[i, "nominal"] / portfolio_df.loc[i, "capital"]
            # positions have been taken, and from 2nd day onwards, we need to perform pnl calculation
            # also, leverage is small currently, but that is because of default scalar is only 2. We will equilibrate
            # once enough data is captured by scaling up, through the variable 'strat scalar'

            # actually, printing is also an i/o operation that takes alot of time
            # print(portfolio_df.loc[i])

            if (i + 1) % 100 == 0:
                logger.info(f'Days iterated: {i + 1}, Progress: {round((i + 1) / total_days * 100, 2)}%')

        logger.info("======= Yay, backtest done!!!! =======")

        # run diagnostics

        # return dataframe
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        backtest_csv_path = f'{self.backtest_dir_path}/{self.sysname}_{current_datetime}.csv'
        portfolio_df.to_csv(f"{backtest_csv_path}", index=False)
        logger.info(f'Backtest results imported to {backtest_csv_path}')
        return portfolio_df

    def get_subsys_pos(self):
        logger.info(f"Testing Strat: {self.sysname}")
        return self.run_backtest(historical_data=self.historical_df)

    # now, from our main driver, we pass the dataframe into the LBMOM strategy, than let the LBMOM perform some
    # calculations using the quantlib indicators calculator.
    # after the calculations, we pass into the simulator, where we can run some simulations and backtesting!

    # we covered some sound principles to make sure the logic is passed around, in flexible fashion.
    # we don't perform unnecssary calculations - we do general calc in the driver, such as returns, volatility etc
    # needed for all strats. then, indicators specific to strat is done inside the strategy to save time.

    # each strategy has a config file, so that we can control some parameters. later, we shall see how this might be useful
    #:)))


