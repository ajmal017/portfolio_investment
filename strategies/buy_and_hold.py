# import numpy as np
from utils.get_data import YahooData
from config import basedir
import pandas as pd
import statistics as stat
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import datetime


stocks = pd.read_excel(f'{basedir}/nyse_tickers.xlsx')
stock_list = stocks['Symbol'].tolist()

#
# dataframe = pd.DataFrame(columns = stock_list[: 5])
# for stock in stock_list[: 10]:
#     try:
#         dataframe[stock] = pdr.get_data_yahoo(stock, '2016-01-01', '2020-01-01')['Adj Close']
#     except:
#         pass

#dataframe.dropna(axis='columns', inplace = True)


class BuyAndHold:
    """
    Defines a Buy&Hold Strategy considering an equally weighted allocation within the portfolio's assets

    Parameters
    ----------
    prices: pandas Dataframe of historical prices
    benchmark: benchmark to compute information ratio
    capital: invested capital
    risk_free = risk free rate

    Returns
    -------
    buy_and_hold: returns, volatility and equity line as lists
    cumulative_ret: list of cumulative returns (being a buy&hold, each return corresponds to the last % change in price
                    since inception)
    sharpe_ratio: returns a list with a TS of Sharpe Ratios at each point in time (eg sharpe ratio so far).
                  [-1] to get the overall SR
    information_ratio: returns a list with a TS of Information Ratios at each point in time
                        (eg information ratio so far).
                      [-1] to get the overall IR

    """

    def __init__(self, prices, benchmark, capital=100, risk_free=0.02):
        self.prices = prices
        self.benchmark = benchmark
        self.capital = capital
        self.risk_free = risk_free
        self.weights = [1 / self.prices.shape[1]] * len(self.prices.columns)
        self.changes = []
        self.equity_line = []
        self.cumulative = []
        self.std_dev = []
        self.sharpe_roll = []

    def buy_and_hold(self):
        if isinstance(self.prices, pd.DataFrame):
            initial_price = self.prices.iloc[0, :]
            for idx, row in self.prices.iloc[1:, :].iterrows():
                current_price = row
                price_diff = (current_price - initial_price) / initial_price
                self.changes.append(sum(price_diff * self.weights))
                self.equity_line.append((self.changes[-1] + 1) * self.capital)
                if len(self.changes) > 1:
                    self.std_dev.append(stat.stdev(self.changes))

        return self.changes, self.std_dev, self.equity_line

    def cumulative_ret(self):

        for i in range(len(self.changes)):
            if i == 0:
                self.cumulative.append(self.changes[i])
            elif i > 0:
                self.cumulative.append(self.changes[i] + self.changes[i - 1])

        return self.cumulative

    def sharpe_ratio(self):

        for i in range(1, len(self.changes)):
            sharpe = (self.changes[i] - self.risk_free) / self.std_dev[i - 1]
            self.sharpe_roll.append(sharpe)
        return self.sharpe_roll

    def information_ratio(self):
        initial_benchmark_price = benchmark.iloc[0, :]
        extra_ret = []
        std_extra_ret = []
        for idx, row in self.benchmark.iloc[1:, :].iterrows():
            current_price = row
            price_diff = (current_price - initial_benchmark_price) / initial_benchmark_price
            extra_ret.append(price_diff[0])
            if len(extra_ret) > 1:
                std_extra_ret.append(stat.stdev(extra_ret))

        information_ratio = []
        for i in range(1, len(extra_ret)):
            information_ratio.append(extra_ret[i] / std_extra_ret[i - 1])

        return information_ratio

    def run(self):
        ret, vol, equity_line = self.buy_and_hold()
        cumulative = self.cumulative_ret()
        sharpe = self.sharpe_ratio()
        information_ratio = self.information_ratio()

        return ret, vol, equity_line, cumulative, sharpe, information_ratio


if __name__ == '__main__':
    # ticker = stock_list[: 10]
    ticker = ['GE', 'IBM', 'GOOG']
    start = datetime.datetime(2006, 1, 1)
    end = datetime.datetime(2020, 1, 1)
    series = 'Adj Close'
    dataframe = YahooData(ticker, start, end, series).get_series()
    benchmark = YahooData(['SPY'], start, end, series).get_series()

    strat = BuyAndHold(dataframe, benchmark)
    strat_ret, strat_vol, equity_line, strat_cumret, strat_sharpe,\
        strat_info_ratio = strat.run()

    plt.plot(dataframe.index[1:], equity_line, label = 'equity_line')
    plt.grid()
    plt.legend()
    plt.xticks(rotation = 45)
    plt.title('Buy & Hold Equity_Line')
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = 'col')
    ax1.plot(dataframe.index[2:], strat_vol, label= 'Std over time')
    plt.title('Metrics')
    ax1.grid()
    ax1.legend()
    ax2.plot(dataframe.index[2:], strat_sharpe, label = 'Sharpe over time')
    ax2.grid()
    ax2.legend()
    ax3.plot(dataframe.index[2:], strat_info_ratio, label = 'Information_ratio over time')
    ax3.grid()
    ax3.legend()
    plt.xticks(rotation = 45)
    plt.show()

    plt.hist(strat_ret)
    plt.show()
