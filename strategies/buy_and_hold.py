# import numpy as np
from utils.get_data import YahooData
from config import basedir
import pandas as pd
import numpy as np
import statistics as stat
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import datetime


stocks = pd.read_excel(f'{basedir}/nyse_tickers.xlsx')
stock_list = stocks['Symbol'].tolist()


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
        self.returns, self.stdev = self.buy_and_hold()
        self.cum_wealth = self.cumulative_wealth()

    def buy_and_hold(self):
        if isinstance(self.prices, pd.DataFrame):
            initial_prices = self.prices.iloc[0, :]
            returns = ((self.prices - initial_prices) / initial_prices) * self.weights
            returns = returns.sum(axis = 1).to_frame()
            returns.columns = ['Portfolio Returns']
            stdev = returns.std()

            return returns, stdev
        else:
            raise TypeError

    def cumulative_wealth(self):
        cum_wealth = (self.returns + 1) * self.capital

        return cum_wealth

    # def sharpe_ratio(self):
    #     ann_sharpe = (self.returns.mean() - self.risk_free) / self.stdev * np.sqrt(252)
    #
    #     return ann_sharpe.to_numpy()

    def get_benchmark_ret(self):
        benchmark_ret = self.benchmark.pct_change ( ).dropna ( )

        return benchmark_ret

    # def information_ratio(self):
    #     inf_ratio = (self.returns.mean().values -
    #                  self.get_benchmark_ret().mean().values) / self.stdev * np.sqrt(252)
    #     return inf_ratio.to_numpy()

    def run(self):
        ret, vol = self.buy_and_hold()
        cum_wealth = self.cumulative_wealth()
        # sharpe = self.sharpe_ratio()
        # info_ratio = self.information_ratio()

        return ret, vol, cum_wealth, sharpe, info_ratio


if __name__ == '__main__':
    # ticker = stock_list[: 10]
    ticker = ['GE', 'IBM', 'GOOG']
    start = datetime.datetime(2006, 1, 1)
    end = datetime.datetime(2020, 1, 1)
    series = 'Adj Close'
    dataframe = YahooData(ticker, start, end, series).get_series()
    benchmark = YahooData(['SPY'], start, end, series).get_series()

    strat = BuyAndHold(dataframe, benchmark)
    strat_ret, strat_vol, cum_wealth, sharpe, info_ratio = strat.run()


    plt.plot(cum_wealth, label = 'backtest')
    plt.grid()
    plt.legend()
    plt.xticks(rotation = 45)
    plt.title('Buy & Hold Cumulative Wealth')
    plt.show()


    plt.hist(strat_ret)
    plt.show()
