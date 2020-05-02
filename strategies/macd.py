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
    Defines a trading strategy using a Moving Average Convergence-Divergence.
    The 12-day and 26-day Exponentially Weighted Moving Averages are computed.
    The MACD is defined as the difference. The signal is defined as the ewma of the MACD
    on a 9-day span. If Signal > MACD, enter the trade.

    Parameters
    ----------
    prices: Pandas Time Series dataframe of prices
    benchmark. Pandas Time Series dataframe of prices. Benchmark used to compute Information Ratio

    Returns
    -------
    backtest: returns portfolio returns as a pandas Dataframe, volatility and portfolio weights
    cumulative_ret: returns the cumulative returns and the cumulative wealth (eg capital * cumulative returns)
    sharpe_ratio: returns the strategy Sharpe Ratio over the whole back-testing period
    information_ratio: returns the strategy Information Ratio over the whole back-testing period
    cagr = returns the Compound Annual Growth Rate
    """
    def __init__(self, prices, benchmark, capital=100, risk_free=0.01):
        self.prices = prices
        self.capital = capital
        self.benchmark_ret = benchmark.pct_change().iloc[1:]
        self.risk_free = risk_free
        self.portfolio_ret, self.std = self.backtest()
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

        std = returns.std()

        return returns.iloc[1:].to_frame(), std

    def cumulative_returns(self):
        cum_returns = self.portfolio_ret.cumsum()
        cum_wealth = (cum_returns + 1) * self.capital

        return cum_returns, cum_wealth

    def sharpe_ratio(self):
        ann_sharpe = (self.portfolio_ret.mean() - self.risk_free) / self.std * np.sqrt(252)

        return ann_sharpe[0]

    def information_ratio(self):
        ret_diff = (self.portfolio_ret.values - self.benchmark_ret [ self.portfolio_ret.index [ 0 ]
                                                                     : self.portfolio_ret.index [ -1 ] ].values)
        std_diff = ret_diff.std()
        inf_ratio = (self.portfolio_ret.mean().values -
                     self.benchmark_ret.mean().values) / std_diff * np.sqrt(252)

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
    port_ret, port_vol = trade.backtest()
    cum_ret, cum_wealth = trade.cumulative_returns()
    sharpe = trade.sharpe_ratio()
    info_ratio = trade.information_ratio()
    cagr = trade.cagr()

    plt.plot(cum_wealth, label = 'strategy_backtest')
    plt.title('MACD: Cumulative Wealth')
    plt.legend()
    plt.grid()
    plt.xticks(rotation = 45)
    plt.show()

    print ( f'Strategy Return from {cum_ret.index [ 0 ].date ( )} to {cum_ret.index [ -1 ].date ( )} : '
            f'{(round ( float ( cum_ret.iloc [ -1 ] ) * 100 , 2 ))} % and Annualised Volatility '
            f'is: {round ( port_vol * np.sqrt ( 252 ) * 100 , 2 )}%' )
    print ( f'CAGR : {round ( cagr * 100 , 2 )}%' )
    print ( f'Sharpe Ratio : {round ( sharpe , 2 )}' )
    print ( f'Information Ratio : {round ( info_ratio , 2 )}' )
