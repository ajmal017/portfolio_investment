from utils.get_data import YahooData
import pandas as pd
import numpy as np
import datetime
from config import basedir
import matplotlib.pyplot as plt

plt.style.use('ggplot')

stocks = pd.read_excel(f'{basedir}/nyse_tickers.xlsx')
stock_list = stocks['Symbol'].tolist()


class CrossSectionalMomentum:
    """
    Defines a Cross Sectional Momentum strategy.
    The strategy uses weekly returns, automatically computed by feeding prices in.

    Weight = (rank - average of ranks) / n° of securities


    Parameters
    ----------
    prices: Pandas Time Series dataframe of prices
    benchmark. Pandas Time Series dataframe of prices. Benchmark used to compute Information Ratio

    Returns
    -------
    time_series_mom: returns a Pd Series of portfolio returns
    cumulative_ret: returns the cumulative returns and the cumulative wealth (eg capital * cumulative returns)
    sharpe_ratio: returns the strategy Sharpe Ratio over the whole back-testing period
    information_ratio: returns the strategy Information Ratio over the whole back-testing period

    """

    def __init__(self, prices, benchmark, capital=100, risk_free=0):
        self.prices = prices
        self.benchmark_ret = benchmark.pct_change().dropna()
        self.capital = capital
        self.risk_free = risk_free
        self.returns = self.prices.pct_change().resample('W').last()
        self.port_ret, self.std, _ = self.backtest()
        self.port_cum, _ = self.cumulative_returns()

    def backtest(self):
        if isinstance(self.returns, pd.DataFrame):
            rank = pd.DataFrame(columns = self.returns.columns)
            weights = pd.DataFrame(columns = self.returns.columns)
            mean_ret = self.returns.rolling(window = 48).sum()[: -4]
            for idx, row in mean_ret.iterrows():
                tmp_rank = row.rank(ascending = False)
                rank.loc[len(rank)] = tmp_rank
                weights.loc[len(weights)] = (tmp_rank - tmp_rank.mean()) / len(tmp_rank)
            rank.index = self.returns.index[4:]
            weights.index = self.returns.index[4:]

            port_ret = (weights.shift(1) * self.returns).dropna().sum(axis = 1).to_frame()
            port_ret.colums = ['Portfolio Returns']
            stdev = port_ret.std()

            return port_ret, stdev, weights

    def cumulative_returns(self):
        cum_ret = self.port_ret.cumsum()
        cum_wealth = (cum_ret + 1) * self.capital

        return cum_ret, cum_wealth

    def sharpe_ratio(self):
        ann_sharpe = (self.port_ret.mean() - self.risk_free) / self.std * np.sqrt(52)

        return ann_sharpe[0]

    def information_ratio(self):
        inf_ratio = (self.port_ret.mean() -
                     self.benchmark_ret.mean()[0]) / self.std * np.sqrt(52)
        return inf_ratio[0]

    def cagr(self):
        end_val = self.port_cum.iloc[-1] + 1
        start_date = self.port_cum.index[0]
        end_date = self.port_cum.index[-1]
        days = (end_date - start_date).days
        cagr = round((float(end_val)) ** (252.0 / days) - 1, 4)

        return cagr


if __name__ == "__main__":
    ticker = ['GE', 'IBM', 'GOOG']
    # ticker = stock_list[80: 100]
    start = datetime.datetime(2006, 1, 1)
    end = datetime.datetime(2020, 1, 1)
    series = 'Adj Close'
    dataframe = YahooData(ticker, start, end, series).get_series()
    # dataframe.dropna(axis = 'columns', inplace = True)
    benchmark = YahooData(['SPY'], start, end, series).get_series()
    trade = CrossSectionalMomentum(dataframe, benchmark)
    port_ret, std, weights = trade.backtest()
    cum_ret, cum_wealth = trade.cumulative_returns()
    sharpe = trade.sharpe_ratio()
    info_ratio = trade.information_ratio()
    cagr = trade.cagr()

    plt.plot(cum_wealth, label = 'strategy_backtest')
    plt.title('Cross Sectional Momentum: Cumulative Wealth')
    plt.legend()
    plt.grid()
    plt.xticks(rotation = 45)
    plt.show()

    print(f'Strategy Return from {cum_ret.index [ 0 ].date ( )} to {cum_ret.index [ -1 ].date ( )} : '
            f'{(round ( float ( cum_ret.iloc [ -1 ] ) * 100 , 2 ))} %')
    print(f'CAGR : {round ( cagr , 2 )}%')
    print(f'Sharpe Ratio : {round ( sharpe , 2 )}')
    print(f'Information Ratio : {round ( info_ratio , 2 )}')
