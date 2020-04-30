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
        self.capital = capital
        self.weights = 1 / len(self.prices.columns)
        self.portfolio_ret, self.stdev = self.backtest()
        self.eq_line = self.equity_line()

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

    def equity_line(self):
        eq_line = (self.portfolio_ret + 1) * self.capital

        return eq_line

    def cagr(self):
        end_val = self.portfolio_ret.iloc[-1] + 1
        start_date = self.portfolio_ret.index[0]
        end_date = self.portfolio_ret.index[-1]
        days = (end_date - start_date).days
        cagr = round((float(end_val)) ** (252.0 / days) - 1, 4)

        return cagr

    def run(self):
        ret, vol = self.backtest()
        equity_line = self.equity_line()
        cagr = self.cagr()

        return ret, vol, equity_line, cagr


if __name__ == '__main__':
    # ticker = stock_list[: 10]
    ticker = ['GE', 'IBM', 'GOOG']
    start = datetime.datetime(2006, 1, 1)
    end = datetime.datetime(2020, 1, 1)
    series = 'Adj Close'
    dataframe = YahooData(ticker, start, end, series).get_series()
    benchmark = YahooData(['SPY'], start, end, series).get_series()

    strat = BuyAndHold(dataframe)
    strat_ret, strat_vol, eq_line, cagr = strat.run()

    print("Final Return from{} to {} is {}% and CAGR is {}%".format(
        strat_ret.index[0], strat_ret.index[-1], round(float(strat_ret.iloc[-1]) * 100, 2), round(cagr * 100, 4)))

