"""
helper functions for use in engineering
"""
import numpy as np


def rank_last_value(x, shift=2, pct=False):
    """
    One step of the procedure to get the rolling rank of the last value
    according to the previous window of values.

    Use the bottlneck routine below instead. It is much faster. Keep this here for testing.
    Use with .rolling(window=100).apply(rank_last_value, raw=True) for example
    """
    args = np.argsort(x)
    rank = np.argwhere(args == (x.shape[0] - 1))[0][0]
    if pct:
        return (rank + 1) / (x.shape[0] + shift)
    return rank
