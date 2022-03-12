import os
import logging
from realgam.quantlib import data_utils as du, general_utils as gu, qlogger
import warnings

import pandas as pd
import numpy as np
import yfinance as yf
from bokeh.plotting import figure, ColumnDataSource, show
from bokeh.models.widgets import Dropdown
from bokeh.models import HoverTool, Span
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot

from bokeh.models import BooleanFilter, CDSView, Select, Range1d, HoverTool, ColumnDataSource
from bokeh.palettes import Category20
from bokeh.models.formatters import NumeralTickFormatter

warnings.filterwarnings("ignore")

logger = qlogger.init(__file__, logging.INFO)
PROJECT_PATH = os.getenv('QuantSystemMVP')

DATA_PATH = f'{PROJECT_PATH}/Data/stock_hist.obj'


def compute_indicators(df):
    df = df.copy()

    df['ma_50'] = df.groupby('ticker').rolling(50).closeadj.mean().values
    df['ma_150'] = df.groupby('ticker').rolling(150).closeadj.mean().values
    df['ma_200'] = df.groupby('ticker').rolling(200).closeadj.mean().values
    df['w52_high'] = df.groupby('ticker').rolling(200).closeadj.max().values
    df['w52_low'] = df.groupby('ticker').rolling(200).closeadj.min().values

    df['ma_200_lag1m'] = df.groupby('ticker').ma_200.shift(25)
    df['ma_200_lag5m'] = df.groupby('ticker').ma_200.shift(110)
    df['w52_low_pct_diff'] = df.closeadj / df.w52_low - 1
    df['w52_high_pct_diff'] = (df.closeadj / df.w52_high - 1).abs()
    df['close_lag1y'] = df.groupby('ticker').closeadj.shift(252)
    df['annual_ret'] = df.closeadj / df.close_lag1y - 1

    return df.dropna()


def compute_breadth_indicators(df):
    df = df.copy()
    df['above_200'] = np.where(df.closeadj > df.ma_200, 1, 0).copy()
    df['above_150'] = np.where(df.closeadj > df.ma_150, 1, 0).copy()
    df['above_50'] = np.where(df.closeadj > df.ma_50, 1, 0).copy()

    df['pos_ret_0'] = np.where(df.ret > 0, 1, 0).copy()
    df['pos_ret_3'] = np.where(df.ret > 0.03, 1, 0).copy()
    df['pos_ret_5'] = np.where(df.ret > 0.05, 1, 0).copy()
    df['pos_ret_8'] = np.where(df.ret > 0.08, 1, 0).copy()

    m_breadth = df.groupby('date').agg(n_200=('above_200', 'sum'), n_150=('above_150', 'sum'),
                                       n_50=('above_50', 'sum'),
                                       n_pos_ret0=('pos_ret_0', 'sum'),
                                       n_pos_ret3=('pos_ret_3', 'sum'), n_pos_ret5=('pos_ret_5', 'sum'),
                                       n_pos_ret8=('pos_ret_8', 'sum'), n_stocks=('ticker', 'count'))

    m_breadth['p_200'] = m_breadth.n_200 / m_breadth.n_stocks * 100
    m_breadth['p_150'] = m_breadth.n_150 / m_breadth.n_stocks * 100
    m_breadth['p_50'] = m_breadth.n_50 / m_breadth.n_stocks * 100
    m_breadth['p_pos_ret0'] = m_breadth.n_pos_ret0 / m_breadth.n_stocks * 100
    m_breadth['p_pos_ret3'] = m_breadth.n_pos_ret3 / m_breadth.n_stocks * 100
    m_breadth['p_pos_ret5'] = m_breadth.n_pos_ret5 / m_breadth.n_stocks * 100
    m_breadth['p_pos_ret8'] = m_breadth.n_pos_ret8 / m_breadth.n_stocks * 100

    return m_breadth


def create_index_df():
    spy_temp = yf.Ticker('SPY')
    spy = spy_temp.history(period='360d')
    spy['Symbol'] = 'SPY'
    spy.reset_index(inplace=True)

    qqq_temp = yf.Ticker('QQQ')
    qqq = qqq_temp.history(period='360d')
    qqq['Symbol'] = 'QQQ'
    qqq.reset_index(inplace=True)

    index = pd.concat([spy, qqq])
    index.sort_values(['Symbol', 'Date'], inplace=True)

    index['ma_50'] = index.groupby('Symbol').rolling(50).Close.mean().values
    index['ma_150'] = index.groupby('Symbol').rolling(150).Close.mean().values
    index['ma_200'] = index.groupby('Symbol').rolling(200).Close.mean().values

    index = index[index.Date.isin(m_breadth.index)]
    index.set_index('Date', inplace=True)

    index.columns = [x.lower() for x in index.columns]

    return index


def plot_price_breadth(df, breadth):
    sym = df.symbol.iloc[0]
    inc = df.close > df.open
    dec = df.open > df.close
    w = 12 * 60 * 60 * 1000  # half day in ms
    pi = 3.14
    date_labels = {i: date.strftime('%b %d') for i, date in enumerate(df.index)}
    hover = HoverTool(
        tooltips=[
            ('d', '@date_copy'),
            ('o', '@open{0}'),
            ('h', '@high{0}'),
            ('l', '@low{0}'),
            ('c', '@close{0}'),
            ('v', '@volume{0}'),
        ],

        formatters={
            '@date_copy': 'datetime'
        },
        mode='mouse')
    TOOLS = "hover,pan,wheel_zoom,box_zoom,reset,save"

    p1 = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1400, plot_height=400,
                title=f"{sym} Candlestick with Volume")
    p1.xaxis.major_label_overrides = date_labels

    #     p1.xaxis.visible = False
    p1.xaxis.major_label_orientation = pi / 4
    p1.grid.grid_line_alpha = 0.3

    p1.segment(df.index, df.high, df.index, df.low, color="black")
    p1.vbar(df.index[inc], w, df.open[inc], df.close[inc], fill_color="green", line_color="black")
    p1.vbar(df.index[dec], w, df.open[dec], df.close[dec], fill_color="#F2583E", line_color="black")
    p1.line(df.index, df.ma_200, legend='MA_200', line_width=2, line_color='orange')
    p1.line(df.index, df.ma_150, legend='MA_150', line_width=2, line_color='purple')
    p1.line(df.index, df.ma_50, legend='MA_50', line_width=2, line_color='blue')
    p1.add_tools(hover)
    #     hover = p1.select(dict(type=HoverTool))
    #     hover.tooltips = [("da", "@account")]
    p1.legend.location = "top_left"

    p2 = figure(x_axis_type="datetime", tools="", toolbar_location=None, plot_width=1400, plot_height=130,
                x_range=p1.x_range)

    p2.xaxis.major_label_overrides = date_labels
    p2.xaxis.major_label_orientation = pi / 4
    p2.grid.grid_line_alpha = 0.3
    p2.vbar(df.index, w, df.volume, [0] * df.shape[0])

    p3 = figure(x_axis_type="datetime", tools="", toolbar_location=None, plot_width=1400, plot_height=200,
                x_range=p1.x_range)

    p3.xaxis.major_label_overrides = date_labels
    p3.xaxis.major_label_orientation = pi / 4
    p3.grid.grid_line_alpha = 0.3
    p3.line(breadth.index, breadth.p_200, legend='Breadth_200', line_width=2, line_color='purple')
    p3.line(breadth.index, breadth.p_50, legend='Breadth_50', line_width=2, line_color='orange')
    p3.line(breadth.index, breadth.p_150, legend='Breadth_150', line_width=2, line_color='blue')
    hline = Span(location=50, dimension='width', line_color='green', line_width=3, line_alpha=0.3)
    p3.renderers.extend([hline])
    p3.legend.location = "top_left"

    show(gridplot([[p1], [p2], [p3]]))


if __name__ == '__main__':
    (stocks_df, stocks_extended_df, available_tickers) = gu.load_file(DATA_PATH)

    stocks_df.sort_values(['ticker', 'date'], inplace=True)

    stocks_df = compute_indicators(stocks_df)
    m_breadth = compute_breadth_indicators(stocks_df)

    index = create_index_df()
    index_start_date = str(index.index.min()).split(' ')[0]

    m_breadth = m_breadth[index_start_date:]

    spy_df = index[index['symbol'] == 'SPY']
    qqq_df = index[index['symbol'] == 'QQQ']
    plot_price_breadth(spy_df, m_breadth)
    plot_price_breadth(qqq_df, m_breadth)


