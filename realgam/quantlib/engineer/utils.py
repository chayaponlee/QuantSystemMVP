"""
helper functions for use in engineering
"""
import numpy as np
import bottleneck as bk


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


def rolling_rank(x, window, pct=False, min_prob=None):
    """
    Get the rolling rank of the last value according to the previous window of values.
    """
    norm_rank = bk.move_rank(x, window, axis=0)  # [-1, 1]
    u = (norm_rank + 1) / 2  # [0, 1]
    if pct:
        if min_prob is None:
            min_prob = 1 / (window + 1)
            return u * (1 - 2 * min_prob) + min_prob  # [min_prob, 1 - min_prob]
    rank = u * (window - 1)
    return np.round(rank)
