from utils.get_data import YahooData
from config import basedir
import pandas as pd
import numpy as np
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
    backtest: returns portfolio returns, annualised volatility and equity line
    cumulative_ret: list of cumulative returns (being a buy&hold, each return corresponds to the last % change in price
                    since inception)
    cagr: returns the Compound Annual Growth Rate
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
            returns = ((self.prices[1:] - initial_prices) / initial_prices) * self.weights
            returns = returns.sum(axis = 1).to_frame()
            returns.columns = ['Portfolio Returns']
            ret = self.prices.pct_change().iloc[1:]
            ret = ret * self.weights
            ret = ret.sum(axis=1).to_frame()
            std = ret.std() * np.sqrt(252)

            return returns, std[0]
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

    trade = BuyAndHold(dataframe)
    port_ret, port_vol, eq_line, cagr = trade.run()

    plt.plot(port_ret * 100, label = 'strategy_backtest')
    plt.title('Buy&Hold : Cumulative Wealth')
    plt.legend()
    plt.grid()
    plt.xticks(rotation = 45)
    plt.show()

    print(f'Strategy Return from {port_ret.index[0].date()} to {port_ret.index[-1].date()} : '
            f'{(round(float(port_ret.iloc[-1]) * 100, 2))} % and Annualised Volatility '
            f'is: {round(port_vol * 100, 2)}%')
    print(f'CAGR : {round(cagr * 100, 2 )}%')
