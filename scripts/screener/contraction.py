import os
import logging
import nasdaqdatalink as ndl
import pandas as pd
import talib as ta

from realgam.quantlib import data_utils_ver2 as du, general_utils as gu, qlogger
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

logger = qlogger.init(__file__, logging.INFO)
PROJECT_PATH = os.getenv('QuantSystemMVP')

DATA_PATH = f'{PROJECT_PATH}/Data/historical/stock_hist_perma.obj'
SAVE_PATH = f'{PROJECT_PATH}/Screen_Data/'

N = 365 * 2
PRIMARY_KEY = 'permaticker'


def BBANDSG(df, period, multBB):
    upper, middle, low = ta.BBANDS(df.closeadj, timeperiod=period, nbdevup=multBB, nbdevdn=multBB, matype=0)

    return pd.DataFrame({'bb_up': upper, 'bb_mid': middle, 'bb_dn': low}, index=df.index)


def ATRG(df, period):
    atr = ta.ATR(df.highadj, df.lowadj, df.closeadj, timeperiod=period)
    return pd.DataFrame({'atr': atr}, index=df.index)


def KCG(df, multKC):
    upper = df.bb_mid.add(df.atr).mul(multKC)
    lower = df.bb_mid.subtract(df.atr).mul(multKC)
    return pd.DataFrame({'upper_kc': upper, 'lower_kc': lower}, index=df.index)


def get_sepa_stocks(sf1, sepa_stocks, ticker_info):
    ticker_info = ticker_info[['ticker', 'exchange']].drop_duplicates(keep='first').copy()

    sf1 = sf1.copy()
    sf1_arq = sf1.sort_values(['ticker', 'datekey'])
    sf1_arq['eps_lag1'] = sf1_arq.groupby('ticker').eps.shift(1)
    sf1_arq['rev_lag1'] = sf1_arq.groupby('ticker').revenue.shift(1)
    sf1_arq['npm_lag1'] = sf1_arq.groupby('ticker').netmargin.shift(1)

    sf1_arq['eps_change'] = (sf1_arq.eps - sf1_arq.eps_lag1) / sf1_arq.eps_lag1.abs()
    sf1_arq['rev_change'] = (sf1_arq.revenue - sf1_arq.rev_lag1) / sf1_arq.rev_lag1.abs()
    sf1_arq['npm_change'] = (sf1_arq.netmargin - sf1_arq.npm_lag1) / sf1_arq.npm_lag1.abs()

    sf1_arq.drop(columns=['eps_lag1', 'rev_lag1', 'npm_lag1'], inplace=True)

    sf1_arq['eps_change_lag1'] = sf1_arq.groupby('ticker').eps_change.shift(1)
    sf1_arq['rev_change_lag1'] = sf1_arq.groupby('ticker').rev_change.shift(1)
    sf1_arq['npm_change_lag1'] = sf1_arq.groupby('ticker').npm_change.shift(1)

    sf1_arq['eps_acc'] = (sf1_arq.eps_change - sf1_arq.eps_change_lag1)
    sf1_arq['rev_acc'] = (sf1_arq.rev_change - sf1_arq.rev_change_lag1)
    sf1_arq['npm_acc'] = (sf1_arq.npm_change - sf1_arq.npm_change_lag1)

    sf1_arq.drop(columns=['eps_change_lag1', 'rev_change_lag1', 'npm_change_lag1'], inplace=True)

    code33 = pd.DataFrame()
    code33['ticker'] = sf1_arq.ticker.unique()

    code33['eps_acc_count'] = sf1_arq.groupby('ticker').tail(3).groupby('ticker')['eps_acc'].apply(
        lambda x: (x > 0).sum()).reset_index(drop=True)
    code33['rev_acc_count'] = sf1_arq.groupby('ticker').tail(3).groupby('ticker')['rev_acc'].apply(
        lambda x: (x > 0).sum()).reset_index(drop=True)
    code33['npm_acc_count'] = sf1_arq.groupby('ticker').tail(3).groupby('ticker')['npm_acc'].apply(
        lambda x: (x > 0).sum()).reset_index(drop=True)

    class_a = code33[code33.eps_acc_count == 3].copy()
    class_a.sort_values(['eps_acc_count', 'rev_acc_count', 'npm_acc_count'], ascending=False, inplace=True)
    class_a = class_a.merge(ticker_info[['ticker', 'exchange']], how='left', on='ticker')
    class_a['exchange'].replace({'NYSEMKT': 'ARCA'}, inplace=True)
    class_a['tview_name'] = class_a.exchange + ':' + class_a.ticker

    code33_b = pd.DataFrame()
    code33_b['ticker'] = sf1_arq.ticker.unique()
    code33_b['eps_change_count'] = sf1_arq.groupby('ticker').tail(3).groupby('ticker')['eps_change'].apply(
        lambda x: (x > 0).sum()).reset_index(drop=True)
    code33_b['rev_change_count'] = sf1_arq.groupby('ticker').tail(3).groupby('ticker')['rev_change'].apply(
        lambda x: (x > 0).sum()).reset_index(drop=True)
    code33_b['npm_change_count'] = sf1_arq.groupby('ticker').tail(3).groupby('ticker')['npm_change'].apply(
        lambda x: (x > 0).sum()).reset_index(drop=True)
    code33_b = code33_b[~code33_b.ticker.isin(class_a.ticker)]

    class_b = code33_b[code33_b.eps_change_count == 3].copy()
    class_b.sort_values(['eps_change_count', 'rev_change_count', 'npm_change_count'], ascending=False, inplace=True)
    class_b = class_b.merge(ticker_info[['ticker', 'exchange']], how='left', on='ticker')
    class_b['exchange'].replace({'NYSEMKT': 'ARCA'}, inplace=True)
    class_b['tview_name'] = class_b.exchange + ':' + class_b.ticker

    return class_a, class_b


if __name__ == '__main__':

    logger.info("Fetching ticker metadata")
    ticker_info = ndl.get_table('SHARADAR/TICKERS', paginate=True)

    logger.info("Loading historical data")
    stocks_df, _, _ = gu.load_file(DATA_PATH)
    stocks_df = stocks_df.reset_index()

    date_N_ago = (datetime.now() - timedelta(days=N))

    eod_focus = stocks_df[stocks_df.date > date_N_ago]
    available_tickers = eod_focus[['permaticker', 'ticker']].drop_duplicates(keep='last')
    latest_ticker_universe_pair = available_tickers.drop_duplicates('permaticker', keep='last')
    latest_permatickers = list(latest_ticker_universe_pair['permaticker'])
    latest_tickers = list(latest_ticker_universe_pair['ticker'])

    symbol_name_converter = {}
    for permaticker, ticker in zip(latest_permatickers, latest_tickers):
        symbol_name_converter[permaticker] = ticker
        symbol_name_converter[ticker] = permaticker

    eod_focus.sort_values(['permaticker', 'date'], inplace=True)

    logger.info('Creating indicators')
    # Calculate ma’s, 52w high and low
    eod_focus['ma_50'] = eod_focus.groupby('ticker').rolling(50).closeadj.mean().values
    eod_focus['ma_150'] = eod_focus.groupby('ticker').rolling(150).closeadj.mean().values
    eod_focus['ma_200'] = eod_focus.groupby('ticker').rolling(200).closeadj.mean().values
    eod_focus['w52_high'] = eod_focus.groupby('ticker').rolling(200).closeadj.max().values
    eod_focus['w52_low'] = eod_focus.groupby('ticker').rolling(200).closeadj.min().values

    # Calculate pct change for various metrics necessary for trend template
    eod_focus['ma_200_lag1m'] = eod_focus.groupby('ticker').ma_200.shift(25)
    eod_focus['ma_200_lag5m'] = eod_focus.groupby('ticker').ma_200.shift(110)
    eod_focus['w52_low_pct_diff'] = eod_focus.closeadj / eod_focus.w52_low - 1
    eod_focus['w52_high_pct_diff'] = (eod_focus.closeadj / eod_focus.w52_high - 1).abs()
    eod_focus['close_lag1y'] = eod_focus.groupby('ticker').closeadj.shift(252)
    eod_focus['annual_ret'] = eod_focus.closeadj / eod_focus.close_lag1y - 1

    # params for bbands and keltner channels
    n_window = 20
    multBB = 2
    multKC = 1.5
    eod_focus[['bb_up', 'bb_mid', 'bb_low']] = eod_focus.groupby(PRIMARY_KEY).apply(BBANDSG, period=n_window,
                                                                                    multBB=multBB)
    eod_focus['atr'] = eod_focus.groupby(PRIMARY_KEY).apply(ATRG, period=n_window).values
    eod_focus[['kc_up', 'kc_low']] = eod_focus.groupby(PRIMARY_KEY).apply(KCG, multKC=multKC)

    # Take only latest data for all tickers and drop all stocks that doesn’t have enough data (ma200 is null)
    filter_time = eod_focus.groupby('ticker').tail(1).dropna()

    # IBD ranking
    filter_time['ibd_rank'] = filter_time.annual_ret.rank(pct=True)

    # create sepa candidates for 1m, 5m, and 5m wide
    logger.info("Creating SEPA candidates")
    sepa_1m = filter_time[((filter_time.closeadj > filter_time.ma_150)  # current price above 150 day moving average
                           # & (filter_time.closeadj > filter_time.ma_50) # current price above 50 day moving average
                           & (filter_time.closeadj > filter_time.ma_200)  # current price above 200 day moving average
                           & (filter_time.ma_200 > filter_time.ma_200_lag1m)  # current ma_200 is trending up
                           & (
                                       filter_time.ma_150 > filter_time.ma_200)  # 150 day moving average > 200 day moving average
                           & (filter_time.ma_50 > filter_time.ma_150)  # 50 day moving average > 150 day moving average
                           & (filter_time.ma_50 > filter_time.ma_200)  # 50 day moving average > 200 day moving average
                           & (filter_time.w52_low_pct_diff >= 0.3)  # current price > 52w_low atleast 30%
                           & (filter_time.w52_high_pct_diff <= 0.25)  # current price within 25% of 52w_high
                           & (filter_time.ibd_rank >= 0.7)  # ibd ranking > 0.7
                           )]

    sepa_5m = filter_time[((filter_time.closeadj > filter_time.ma_150)  # current price above 150 day moving average
                           # & (filter_time.closeadj > filter_time.ma_50) # current price above 50 day moving average
                           & (filter_time.closeadj > filter_time.ma_200)  # current price above 200 day moving average
                           & (filter_time.ma_200 > filter_time.ma_200_lag5m)  # current ma_200 is trending up
                           & (
                                       filter_time.ma_150 > filter_time.ma_200)  # 150 day moving average > 200 day moving average
                           & (filter_time.ma_50 > filter_time.ma_150)  # 50 day moving average > 150 day moving average
                           & (filter_time.ma_50 > filter_time.ma_200)  # 50 day moving average > 200 day moving average
                           & (filter_time.w52_low_pct_diff >= 0.3)  # current price > 52w_low atleast 30%
                           & (filter_time.w52_high_pct_diff <= 0.25)  # current price within 25% of 52w_high
                           & (filter_time.ibd_rank >= 0.7)  # ibd ranking > 0.7
                           )]

    sepa_5m_wide = filter_time[
        ((filter_time.closeadj > filter_time.ma_150)  # current price above 150 day moving average
         # & (filter_time.closeadj > filter_time.ma_50) # current price above 50 day moving average
         & (filter_time.closeadj > filter_time.ma_200)  # current price above 200 day moving average
         & (filter_time.ma_200 > filter_time.ma_200_lag5m)  # current ma_200 is trending up
         & (filter_time.ma_150 > filter_time.ma_200)  # 150 day moving average > 200 day moving average
         & (filter_time.ma_50 > filter_time.ma_150)  # 50 day moving average > 150 day moving average
         & (filter_time.ma_50 > filter_time.ma_200)  # 50 day moving average > 200 day moving average
         & (filter_time.w52_low_pct_diff >= 0.3)  # current price > 52w_low atleast 30%
         & (filter_time.w52_high_pct_diff <= 0.25)  # current price within 25% of 52w_high
         )]

    # print(sepa_1m.ticker.unique())
    # logger.info('Creating squeeze candidates')
    # # filter for squeeze candidates
    # sepa_1m = sepa_1m[(sepa_1m.bb_low > sepa_1m.kc_low) & (sepa_1m.bb_up < sepa_1m.kc_up)]
    # sepa_5m = sepa_5m[(sepa_5m.bb_low > sepa_5m.kc_low) & (sepa_5m.bb_up < sepa_5m.kc_up)]
    # sepa_5m_wide = sepa_5m_wide[(sepa_5m_wide.bb_low > sepa_5m_wide.kc_low) & (sepa_5m_wide.bb_up < sepa_5m_wide.kc_up)]
    # print(sepa_1m.ticker.unique())
    # filter symbols
    sepa_1m_sym = list(sepa_1m[PRIMARY_KEY])
    sepa_5m_sym = list(sepa_5m[PRIMARY_KEY])
    sepa_5m_wide_sym = list(sepa_5m_wide[(~sepa_5m_wide[PRIMARY_KEY].isin(sepa_5m_sym)) &
                                         (~sepa_5m_wide[PRIMARY_KEY].isin(sepa_1m_sym))][PRIMARY_KEY])


    # covert permaticker to ticker
    sepa_1m_sym = [symbol_name_converter[symbol] for symbol in sepa_1m_sym]
    sepa_5m_sym = [symbol_name_converter[symbol] for symbol in sepa_5m_sym]
    sepa_5m_wide_sym = [symbol_name_converter[symbol] for symbol in sepa_5m_wide_sym]



    # get unique symbols
    union_syms_temp = sepa_1m_sym + sepa_5m_sym + sepa_5m_wide_sym
    union_syms = list(dict.fromkeys(union_syms_temp))

    print(len(union_syms))
    logger.info('Loading Financials')
    financial_cols = ['ticker', 'dimension', 'calendardate', 'datekey', 'eps', 'revenue', 'netmargin']
    sf1 = ndl.get_table('SHARADAR/SF1', ticker=union_syms, qopts={"columns": financial_cols}, dimension='ARQ',
                        paginate=True)
    logger.info("Filtering Good Fundamentals Candidates")
    class_a_1m, class_b_1m = get_sepa_stocks(sf1, sepa_1m_sym, ticker_info)
    class_a_5m, class_b_5m = get_sepa_stocks(sf1, sepa_5m_sym, ticker_info)
    class_a_5mw, class_b_5mw = get_sepa_stocks(sf1, sepa_5m_wide_sym, ticker_info)

    # Outputing screened candidates
    # date_today = datetime.now()
    # class_a_1m[['tview_name']].to_csv(
    #     f'{SAVE_PATH}class_a_1m_{date_today.year}_{date_today.month}_{date_today.day}.txt',
    #     header=None, index=None, sep=',')
    # class_b_1m[['tview_name']].to_csv(
    #     f'{SAVE_PATH}class_b_1m_{date_today.year}_{date_today.month}_{date_today.day}.txt',
    #     header=None, index=None, sep=',')
    #
    # # class_a_5m[['tview_name']].to_csv(f'{SAVE_PATH}class_a_5m_{date_today.year}_{date_today.month}_{date_today.day}.txt',
    # #                                   header=None, index=None, sep = ',')
    # # class_b_5m[['tview_name']].to_csv(f'{SAVE_PATH}class_b_5m_{date_today.year}_{date_today.month}_{date_today.day}.txt',
    # #                                   header=None, index=None, sep = ',')
    #
    # class_b_5mw[['tview_name']].to_csv(
    #     f'{SAVE_PATH}class_a_5mw_{date_today.year}_{date_today.month}_{date_today.day}.txt',
    #     header=None, index=None, sep=',')
    # class_b_5mw[['tview_name']].to_csv(
    #     f'{SAVE_PATH}class_b_5mw_{date_today.year}_{date_today.month}_{date_today.day}.txt',
    #     header=None, index=None, sep=',')
    #
    # logger.info(f"Screened candidates saved to {SAVE_PATH}")
