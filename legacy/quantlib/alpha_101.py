"""
From 101 alpha formulas, we define the technical factors here

"""

# 1


# def alpha1(stacked_hist: FinancialDF, n_ts_arg_max: int = 5, n_std: int = 20) -> np.array:
#     """
#     Runtime: 58 sec
#     Formula: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, n_std) : close), 2.), n_ts_arg_max)) - 0.5)
#     :param n_std: lookback range for calculating return vol
#     :param n_ts_arg_max: lookback range for calculating ts argmax
#     :param stacked_hist: historical stock dataframe (columns: datetimeIndex, ticker, and closeadj)
#     :return: alpha1
#     """
#
#     stacked_hist.sort_values(['ticker', 'date'], inplace=True)
#     stacked_hist_copy = stacked_hist.copy()
#
#     stacked_hist_copy['ret'] = stacked_hist_copy.groupby('ticker').closeadj.apply(ic.calc_ret)
#     stacked_hist_copy[f'retvol{n_std}'] = stacked_hist_copy.groupby('ticker').closeadj.apply(ic.roll_std, n=n_std)
#
#     stacked_hist_copy['closeadj_sqr'] = stacked_hist_copy.closeadj ** 2
#     stacked_hist_copy['closeadj_argmax'] = (stacked_hist_copy.groupby('ticker').closeadj_sqr.rolling(n_ts_arg_max)
#                                             .apply(np.argmax, engine='cython', raw=True).values)
#
#     stacked_hist_copy[f'retvol{n_std}_sqr'] = stacked_hist_copy[f'retvol{n_std}'] ** 2
#     stacked_hist_copy[f'retvol{n_std}_argmax'] = (
#         stacked_hist_copy.groupby('ticker')[f'retvol{n_std}_sqr'].rolling(n_ts_arg_max)
#             .apply(np.argmax, engine='cython', raw=True).values)
#
#     stacked_hist_copy['tsargmax_withcond'] = np.where(stacked_hist_copy.ret < 0,
#                                                       stacked_hist_copy[f'retvol{n_std}_argmax'],
#                                                       stacked_hist_copy.closeadj_argmax) + 1  # + 1 so no 0 indices
#
#     stacked_hist_copy.loc[
#         stacked_hist_copy.closeadj_argmax.isnull() | stacked_hist_copy[f'retvol{n_std}_argmax'].isnull(),
#         'tsargmax_withcond'] = np.nan
#
#     # since tsargmax will be 1 to n_ts_arg_max, so the way we rank it will be based on ratios
#     # which is the same as ranking the whole universe (values are still integers in range of 1 to n_ts_arg_max)
#     stacked_hist_copy['alpha'] = stacked_hist_copy.tsargmax_withcond / n_ts_arg_max
#
#     return stacked_hist_copy.alpha.values


