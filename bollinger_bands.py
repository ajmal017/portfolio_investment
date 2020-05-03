from config import basedir
from utils.get_data import YahooData
import pandas as pd
import datetime
import matplotlib.pyplot as plt


stocks = pd.read_excel(f'{basedir}/nyse_tickers.xlsx')
stock_list = stocks['Symbol'].tolist()


class BollingerBand:
    """
    Parameters
    ----------
    data: price time series as DataFrame
        -> only one security at a time
    window: window size to compute rolling stats on

    Returns
    --------
    It adds to the price DataFrame the following columns:
    - rolling MA (window)
    - rolling std (window)
    - upper band (MA + 2 * std)
    - lower band (MA - 2 * std)
    """

    def __init__(self, data, window=30):
        self.data = data
        self.window = window

    def compute_bollinger_bands(self):
        if isinstance(self.data, pd.DataFrame):
            self.data[f'{self.window}-day MA'] = self.data.iloc[:, 0].rolling(window = self.window).mean()
            self.data[f'{self.window}-day STD'] = self.data.iloc[:, 0].rolling(window = self.window).std()
            self.data['upper_b'] = self.data[f'{self.window}-day MA'] + (self.data[f'{self.window}-day STD'] * 2)
            self.data['lower_b'] = self.data[f'{self.window}-day MA'] - (self.data[f'{self.window}-day STD'] * 2)

            return self.data


if __name__ == '__main__':
    ticker = ['GE', 'IBM', 'GOOG']
    start = datetime.datetime(2006, 1, 1)
    end = datetime.datetime(2020, 1, 1)
    series = 'Adj Close'
    # ge = YahooData(['GE'], start, end, series).get_series()
    # ibm = YahooData(['IBM'], start, end, series).get_series()
    google = YahooData(['GOOG'], start, end, series).get_series()

    bb = BollingerBand(google)
    bollinger_b = bb.compute_bollinger_bands()

    # 30-day Bollinger Band Google
    plt.plot(google.drop(columns = ['30-day STD']))
    plt.title('30-Day Bollinger Band : Google')
    plt.ylabel('Price $')
    plt.xticks(rotation = 45)
    plt.show()

    # let's use a better plot style
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize = (12, 6))
    ax = fig.add_subplot(111)

    x_axis = google.index.get_level_values(0)

    ax.fill_between(x_axis, google['upper_b'], google['lower_b'], color = 'grey')

    ax.plot(x_axis, google.iloc[:, 0], color = 'blue', lw = 1, label = 'Adj Close')
    ax.plot(x_axis, google['30-day MA'], color = 'black', lw = 1, label = '30-day MA')

    ax.set_title('30-Day Bollinger Band - Google')
    ax.set_xlabel('Date (Year/Month)')
    ax.set_ylabel('Price $')
    ax.legend()
    plt.show()
