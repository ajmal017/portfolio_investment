import pandas as pd
import pandas_datareader as pdr
import datetime


class YahooData:
    """
    Gets historical data from Yahoo

    Parameters
    ----------
    tickers: list of tickers (can use nyse_tickers.xlsx file in root)
    start_date: datetime.date (ex 2006, 10, 1)
    end_date : datetime.date (ex 2019, 1, 1)
    series : series to retrieve the dataframe on. Choose from ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']

    Returns
    -------
    get_data() : whole dataset for the input tickers
    get_series() : Pandas DataFrame containing only the series of interest, with Date as index and Tickers as columns

    """
    def __init__(self, tickers, start_date, end_date, series=None):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.get_data()
        self.series = series

    def get_data(self):

        return pdr.get_data_yahoo(self.tickers, start = self.start_date, end = self.end_date)

    def get_series(self):

        return self.data[self.series]



if __name__ == '__main__':
    ticker = ['GE', 'IBM', 'GOOG']
    start = datetime.datetime(2006, 10, 1)
    end = datetime.datetime(2019, 1, 1)
    y = YahooData(ticker, start, end, series = 'Adj Close')
    dataset = y.get_data()
    adj_close = y.get_series()
