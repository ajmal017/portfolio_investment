from utils.get_data import YahooData
from config import basedir
import pandas as pd
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

    def __init__(self, prices, capital=100):
        self.prices = prices
        self.benchmark = benchmark
        self.capital = capital
        # self.weights = [1 / self.prices.shape[1]] * len(self.prices.columns)
        self.weights = 1 / len(self.prices.columns)
        self.portfolio_ret, self.stdev = self.backtest()
        self.cum_wealth = self.cumulative_wealth()

    def backtest(self):
        if isinstance(self.prices, pd.DataFrame):
            initial_prices = self.prices.iloc[0, :]
            returns = ((self.prices - initial_prices) / initial_prices) * self.weights
            returns = returns.sum(axis = 1).to_frame()
            returns.columns = ['Portfolio Returns']
            stdev = returns.std()

            return returns, stdev
        else:
            print('Input not as pd DataFrame')
            raise TypeError

    def cumulative_wealth(self):
        cum_wealth = (self.portfolio_ret + 1) * self.capital

        return cum_wealth

    def cagr(self):
        start_val = 1
        end_val = self.portfolio_ret.iloc[-1]
        start_date = self.portfolio_ret.index[0]
        end_date = self.portfolio_ret.index[-1]
        days = (end_date - start_date).days
        CAGR_final = round(((float(end_val) / float(start_val)) ** (252.0 / days)) - 1, 4)

        return CAGR_final

    def run(self):
        ret, vol = self.backtest()
        cum_wealth = self.cumulative_wealth()
        cagr = self.cagr()

        return ret, vol, cum_wealth, cagr


if __name__ == '__main__':
    # ticker = stock_list[: 10]
    ticker = ['GE', 'IBM', 'GOOG']
    start = datetime.datetime(2006, 1, 1)
    end = datetime.datetime(2020, 1, 1)
    series = 'Adj Close'
    dataframe = YahooData(ticker, start, end, series).get_series()
    benchmark = YahooData(['SPY'], start, end, series).get_series()

    strat = BuyAndHold(dataframe, benchmark)
    strat_ret, strat_vol, cum_wealth, cagr = strat.run()

    plt.plot(cum_wealth, label = 'backtest')
    plt.grid()
    plt.legend()
    plt.xticks(rotation = 45)
    plt.title('Buy & Hold Cumulative Wealth')
    plt.show()
