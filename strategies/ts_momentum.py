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

benchmark = pdr.get_data_yahoo(['^GSPC'], '2016-01-01', '2020-01-01')['Adj Close']


class TimeSeriesMomentum:
    """
    Defines a TS Momentum Strategy based on Moskowitz, Pedersen 2012.
    The strategy uses weekly returns, automatically computed by feeding prices in.

    Weight = average return of (last 52 weeks - last 4 weeks) / last 52-week stdev


    Parameters
    ----------
    prices: Pandas Time Series dataframe of prices


    Returns
    -------
    time_series_mom: returns a Pd Series of portfolio returns
    cumulative_ret: returns the Equity Line (eg capital * cumulative returns)
    sharpe_ratio: returns the strategy Sharpe Ratio over the whole back-testing period
    information_ratio: returns the strategy Information Ratio over the whole back-testing period


    """

    def __init__(self, prices, benchmark, capital=100, risk_free=0.02):
        self.prices = prices
        self.benchmark = benchmark
        self.capital = capital
        self.risk_free = risk_free
        self.returns = self.prices.pct_change().resample('W').last()
        self.portfolio_ret = self.time_series_mom()
        self.port_cum, _ = self.cumulative_ret()

    def time_series_mom(self):
        if isinstance(self.returns, pd.DataFrame):
            roll_std = self.returns.rolling(window = 52).std()[52:]
            mean_ret = self.returns[: -4].rolling(window = 48).mean()[roll_std.first_valid_index():]
            weights = (mean_ret / roll_std).dropna()

            port_ret = weights * self.returns.shift(-1)
            port_ret = port_ret.shift(1).dropna().sum(axis=1).to_frame()
            port_ret.columns = ['Portfolio Returns']

            return port_ret

    def cumulative_ret(self):
        port_cumulative = self.portfolio_ret.cumsum()
        equity_line = (port_cumulative + 1) * self.capital

        return port_cumulative, equity_line

    def sharpe_ratio(self):
        stdev = self.portfolio_ret.std()
        sharpe = (self.portfolio_ret.cumsum()[-1] - self.risk_free) / stdev

        return sharpe

    def get_benchmark_ret(self):
        benchmark_ret = self.benchmark.pct_change().dropna()

        return benchmark_ret

    def information_ratio(self):
        excess_ret = self.portfolio_ret - self.get_benchmark_ret()
        tracking_error = excess_ret.std()
        info_ratio = (excess_ret[-1]) / tracking_error

        return info_ratio


if __name__ == "__main__":
    ts_mom = TimeSeriesMomentum(dataframe, benchmark)
    portfolio_backtest = ts_mom.time_series_mom()
    cumulative, equity_line = ts_mom.cumulative_ret()
    sharpe_ratio = ts_mom.sharpe_ratio()
    information_ratio = ts_mom.information_ratio()

    plt.plot(equity_line, label = 'strategy_backtest')
    plt.grid()
    plt.xticks(rotation = 45)
    plt.legend()
    plt.title('TS Momentum : Equity Line')
    plt.show()

    print(f'Strategy Return from {equity_line.index[0].date()} to {equity_line.index[-1].date()} : '
          f'{round(cumulative[-1] * 100, 2)}%')
    print(f'Sharpe Ratio : {sharpe_ratio}')
    print(f'Information Ratio : {information_ratio}')


