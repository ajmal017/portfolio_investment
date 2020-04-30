import datetime
import matplotlib.pyplot as plt
from config import basedir
from utils.metrics import *
from utils.get_data import YahooData

plt.style.use('ggplot')

stocks = pd.read_excel(f'{basedir}/nyse_tickers.xlsx')
stock_list = stocks['Symbol'].tolist()


class TimeSeriesMomentum:
    """
    Defines a TS Momentum Strategy based on Moskowitz, Pedersen 2012.
    The strategy uses weekly returns, automatically computed by feeding prices in.

    Weight = average return of (last 52 weeks - last 4 weeks) / last 52-week stdev


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
        self.portfolio_ret, self.std, self.weights = self.backtest()
        self.port_cum, _ = self.cumulative_returns()

    def backtest(self):
        if isinstance(self.returns, pd.DataFrame):
            roll_std = self.returns.rolling(window = 52).std().dropna()
            mean_ret = self.returns.rolling(window = 48).mean()[: -4].dropna()
            mean_ret.index = roll_std.index
            weights = (mean_ret / roll_std).dropna()
            # weights.index = self.returns.index[4:]
            port_ret = weights.shift(1) * self.returns
            port_ret = port_ret.dropna().sum(axis=1).to_frame()
            port_ret.columns = ['Portfolio Returns']
            stdev = port_ret.std()

            return port_ret, stdev, weights

    def cumulative_returns(self):
        port_cumulative = self.portfolio_ret.cumsum()
        cum_wealth = (port_cumulative + 1) * self.capital

        return port_cumulative, cum_wealth

    def sharpe_ratio(self):
        ann_sharpe = (self.portfolio_ret.mean() - self.risk_free) / self.std * np.sqrt(52)

        return ann_sharpe[0]

    def information_ratio(self):
        inf_ratio = (self.portfolio_ret.mean().values -
                     self.benchmark_ret.mean().values) / self.std * np.sqrt(52)

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
    dataframe.dropna(axis = 'columns', inplace = True)
    benchmark = YahooData(['SPY'], start, end, series).get_series()

    trade = TimeSeriesMomentum(dataframe, benchmark)
    port_ret, stdev, port_weights = trade.backtest()
    cumulative, wealth = trade.cumulative_returns()
    sharpe = trade.sharpe_ratio()
    info_ratio = trade.information_ratio()

    plt.plot(wealth, label = 'strategy_backtest')
    plt.grid()
    plt.xticks(rotation = 45)
    plt.legend()
    plt.title('TS Momentum : Cumulative Wealth')
    plt.show()

    print(f'Strategy Return from {cumulative.index[0].date()} to {cumulative.index[-1].date()} : '
          f'{(round(float(cumulative.iloc[-1])* 100, 2))} %')
    print(f'CAGR : {round(cagr(), 2)}%')
    print(f'Sharpe Ratio : {round(sharpe, 2)}')
    print(f'Information Ratio : {round(info_ratio, 2)}')


