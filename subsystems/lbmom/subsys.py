import json
from quantlib import indicators_calc


class Lbmom():

    """
    Long biased momentum strategy
    """

    def __init__(self, instruments_config, historical_df, simulation_start, vol_target):
        self.ma_pairs = [(23, 82), (44, 244), (124, 294), (37, 229), (70, 269), (158, 209), (81, 169), (184, 203),
                      (23, 265), (244, 268), (105, 106), (193, 250), (127, 294), (217, 274)]
        self.historical_df = historical_df
        self.simulation_start = simulation_start
        self.vol_target = vol_target
        with open(instruments_config) as f:
            self.instruments_config = json.load(f)
        self.sysname = 'LBMOM'

    # we implement a few functions
    # 1. A function to get extra indicators specific to this strategy
    # 2. A function to run a backtest/get positions from this strategy

    def extend_historicals(self, instruments, historical_data):
        # we need indicators of momentum
        # let this be the MA crossover, such that if the fastMA crosses over the slowMA, then buy
        # long-biased momentum only focuses on the long direction
        # let's also use a filter to identify false positive signals. We use the average directional index (adx)

        for inst in instruments:
            historical_data[f'{inst} adx'] = indicators_calc.adx_series(historical_data)

        return historical_data


    def run_backtest(self, historical_data):
        pass

    def get_subsys_pos(selfs):
        pass
