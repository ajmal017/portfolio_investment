import pandas_datareader as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def get(tickers, startdate, enddate):

    def data(ticker):

        return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))

    datas = map(data, tickers)

    return pd.concat(datas, keys=tickers, names=['Ticker', 'Date'])


tickers = ['GE', 'IBM', 'GOOG']

all_data = get(tickers, datetime.datetime(2006, 10, 1), datetime.datetime(2019, 1, 1))

daily_close_px = all_data['Adj Close'].reset_index().pivot('Date','Ticker', 'Adj Close')

daily_pct_change = daily_close_px.pct_change()
daily_pct_change.hist(bins=50, sharex=True, figsize=(12, 8))
plt.show()

pd.plotting.scatter_matrix(daily_pct_change, diagonal='kde', alpha=0.1, figsize=(12, 12))
plt.show()
