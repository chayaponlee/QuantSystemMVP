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
SAVE_PATH = f'{PROJECT_PATH}/Screen_Data/contraction_bb_keltner/'

N = 365 * 2
PRIMARY_KEY = 'permaticker'


def BBANDSG(df, period, multBB):
    upper, middle, low = ta.BBANDS(df.closeadj, timeperiod=period, nbdevup=multBB, nbdevdn=multBB, matype=0)

    return pd.DataFrame({'bb_up': upper, 'bb_mid': middle, 'bb_dn': low}, index=df.index)


def ATRG(df, period):
    atr = ta.ATR(df.highadj, df.lowadj, df.closeadj, timeperiod=period)
    return pd.DataFrame({'atr': atr}, index=df.index)


def KCG(df, multKC):
    upper = df.bb_mid.add(df.atr.mul(multKC))
    lower = df.bb_mid.subtract(df.atr.mul(multKC))
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
    # available_tickers = eod_focus[['permaticker', 'ticker']].drop_duplicates(keep='last')
    # latest_ticker_universe_pair = available_tickers.drop_duplicates('permaticker', keep='last')
    # latest_permatickers = list(latest_ticker_universe_pair['permaticker'])
    # latest_tickers = list(latest_ticker_universe_pair['ticker'])

    # symbol_name_converter = {}
    # for permaticker, ticker in zip(latest_permatickers, latest_tickers):
    #     symbol_name_converter[permaticker] = ticker
    #     symbol_name_converter[ticker] = permaticker

    eod_focus.sort_values([PRIMARY_KEY, 'date'], inplace=True)

    eod_focus['ma_150'] = eod_focus.groupby('ticker').rolling(150).closeadj.mean().values
    eod_focus['ma_200'] = eod_focus.groupby('ticker').rolling(200).closeadj.mean().values

    # params for bbands and keltner channels
    n_window = 20
    multBB = multKC = 1.5

    eod_focus[['bb_up', 'bb_mid', 'bb_low']] = eod_focus.groupby(PRIMARY_KEY).apply(BBANDSG, period=n_window,
                                                                                    multBB=multBB)
    eod_focus['atr'] = eod_focus.groupby(PRIMARY_KEY).apply(ATRG, period=n_window).values
    eod_focus[['kc_up', 'kc_low']] = eod_focus.groupby(PRIMARY_KEY).apply(KCG, multKC=multKC)

    # Take only latest data for all tickers and drop all stocks that doesn’t have enough data (ma200 is null)
    filter_time = eod_focus.groupby(PRIMARY_KEY).tail(1).dropna()

    filter_time = filter_time[filter_time.date == filter_time.date.max()]

    logger.info('Creating squeeze candidates')

    # filter for trending candidates
    filter_time = filter_time[((filter_time.closeadj > filter_time.ma_150)  # current price above 150 day moving average
                           & (filter_time.closeadj > filter_time.ma_200)  # current price above 200 day moving average
                           & (filter_time.ma_150 > filter_time.ma_200))]
    # filter for squeeze candidates
    filter_time = filter_time[(filter_time.bb_low > filter_time.kc_low) & (filter_time.bb_up < filter_time.kc_up)]

    sym_list = list(filter_time.ticker.unique())

    logger.info(f'Number of contraction candidates: {len(sym_list)}')

    logger.info('Loading Financials')
    financial_cols = ['ticker', 'dimension', 'calendardate', 'datekey', 'eps', 'revenue', 'netmargin']
    sf1 = ndl.get_table('SHARADAR/SF1', ticker=sym_list, qopts={"columns": financial_cols}, dimension='ARQ',
                        paginate=True)
    logger.info("Filtering Good Fundamentals Candidates")
    class_a, class_b = get_sepa_stocks(sf1, sym_list, ticker_info)

    logger.info(f'Number of contraction A candidates: {class_a.shape[0]}')
    logger.info(list(class_a['tview_name']))

    logger.info(f'Number of contraction B candidates: {class_b.shape[0]}')
    logger.info(list(class_b['tview_name']))
    # Outputing screened candidates
    date_today = datetime.now()
    class_a[['tview_name']].to_csv(
        f'{SAVE_PATH}contraction_a2_{date_today.year}_{date_today.month}_{date_today.day}.txt',
        header=None, index=None, sep=',')
    class_b[['tview_name']].to_csv(
        f'{SAVE_PATH}contraction_b2_{date_today.year}_{date_today.month}_{date_today.day}.txt',
        header=None, index=None, sep=',')

    logger.info(f"Screened candidates saved to {SAVE_PATH}")
