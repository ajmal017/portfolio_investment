from utils.get_data import YahooData
from config import basedir
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')

stocks = pd.read_excel(f'{basedir}/nyse_tickers.xlsx')
stock_list = stocks['Symbol'].tolist()

class Macd:
    """
    Defines a trading strategy using a Moving Average Convergence-Divergence

    Parameters
    ----------
    dataframe: pandas Dataframe of prices


    """
    def __init__(self, prices, benchmark, capital=100, risk_free=0.01):
        self.prices = prices
        self.capital = capital
        self.benchmark_ret = benchmark.pct_change().dropna()
        self.risk_free = risk_free
        self.port_ret = self.backtest()
        self.cumret, self.cumwealth = self.cumulative_returns()

    def backtest(self):
        self.prices['12d_EMA'] = self.prices.iloc[:, 0].ewm(span = 12).mean()
        self.prices['26d_EMA'] = self.prices.iloc[:, 0].ewm(span = 26).mean()

        # Calculate MACD
        self.prices['MACD'] = self.prices['26d_EMA'] - self.prices['12d_EMA']

        # Calculate Signal
        self.prices['Signal'] = self.prices['MACD'].ewm(span = 9).mean()

        # Define Signal
        self.prices['trading_signal'] = np.where(self.prices['MACD'] > self.prices['Signal'], 1, -1)

        # Calculate Returns
        self.prices['returns'] = self.prices.iloc[:, 0].pct_change()

        # Calculate Strategy Returns
        returns = self.prices['strategy_returns'] = self.prices['returns'] * self.prices['trading_signal'].shift(1)

        return returns.dropna().to_frame()

    def cumulative_returns(self):
        cum_returns = self.port_ret.cumsum()
        cum_wealth = (cum_returns + 1) * self.capital

        return cum_returns, cum_wealth

    def sharpe_ratio(self):
        stdev = self.port_ret.std()
        ann_sharpe = (self.port_ret.mean() - self.risk_free) / stdev * np.sqrt(252)

        return ann_sharpe[0]

    def information_ratio(self):
        inf_ratio = ((self.port_ret.mean().values -
                     self.benchmark_ret.mean().values) / self.port_ret.std()) * np.sqrt(252)
        return inf_ratio[0]

    def cagr(self):
        end_val = self.cumret.iloc[-1] + 1
        start_date = self.cumret.index[0]
        end_date = self.cumret.index[-1]
        days = (end_date - start_date).days
        cagr = round(((float(end_val)) ** (252.0 / days)) - 1, 4)

        return cagr


if __name__ == '__main__':
    # ticker = ['GE', 'IBM', 'GOOG']
    start = datetime.datetime(2006, 1, 1)
    end = datetime.datetime(2020, 1, 1)
    series = 'Adj Close'
    dataframe = YahooData(['GOOG'], start, end, series).get_series()
    dataframe.dropna(axis = 'columns', inplace = True)
    benchmark = YahooData(['SPY'], start, end, series).get_series()

    trade = Macd(dataframe, benchmark)
    port_ret = trade.backtest()
    cum_ret, cum_wealth = trade.cumulative_returns()
    sharpe = trade.sharpe_ratio()
    info_ratio = trade.information_ratio()
    cagr = trade.cagr()

    plt.plot(cum_wealth , label = 'strategy_backtest')
    plt.title('MACD: Cumulative Wealth')
    plt.legend()
    plt.grid()
    plt.xticks(rotation = 45)
    plt.show()

    print(f'Strategy Return from {cum_ret.index [ 0 ].date ( )} to {cum_ret.index [ -1 ].date ( )} : '
            f'{(round ( float ( cum_ret.iloc [ -1 ] ) * 100 , 2 ))} %')
    print(f'CAGR : {round ( cagr , 2 )}%')
    print(f'Sharpe Ratio : {round ( sharpe , 2 )}')
    print(f'Information Ratio : {round ( info_ratio , 2 )}')
