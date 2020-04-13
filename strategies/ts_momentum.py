import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from config import basedir


stocks = pd.read_excel(f'{basedir}/nyse_tickers.xlsx')
stock_list = stocks['Symbol'].tolist()

dataframe = pd.DataFrame(columns = stock_list[: 5])
for stock in stock_list[: 10]:
    try:
        dataframe[stock] = pdr.get_data_yahoo(stock, '2016-01-01', '2020-01-01')['Adj Close']
    except:
        pass

dataframe.dropna(axis='columns', inplace = True)


class TimeSeriesMomentum:
    """
    Defines a TS Momentum Strategy based on Moskowitz, Pedersen 2012.
    The strategy uses weekly returns, automatically computed by feeding prices in.

    Weight = average return of (last 52 weeks - last 4 weeks) / last 52-week stdev


    Parameters
    ----------


    Returns
    -------


    """

    def __init__(self, prices, capital=100, risk_free=0.02):
        self.prices = prices
        # self.benchmark = benchmark
        self.capital = capital
        self.returns = self.prices.pct_change().resample('W').last()
        self.portfolio_ret = self.time_series_mom()

    def time_series_mom(self):
        if isinstance(self.returns, pd.DataFrame):
            roll_std = self.returns.rolling(window = 52).std()[52:]
            mean_ret = self.returns[: -4].rolling(window = 48).mean()[roll_std.first_valid_index():]
            weights = (mean_ret / roll_std).dropna()

            port_ret = weights * self.returns.shift(-1)
            port_ret = port_ret.shift(1).dropna().sum(axis=1)

            return port_ret

    def cumulative_ret(self):
        port_cumulative = self.portfolio_ret.cumsum()
        equity_line = (port_cumulative + 1) * self.capital

        return equity_line


if __name__ == "__main__":
    ts_mom = TimeSeriesMomentum(dataframe)
    portfolio_backtest = ts_mom.time_series_mom()
    equity_line = ts_mom.cumulative_ret()

    plt.plot(equity_line, label = 'strategy_backtest')
    plt.grid()
    plt.xticks(rotation = 45)
    plt.legend()
    plt.title('TS Momentum : Equity Line')
    plt.show()

