import numpy as np
import pandas as pd
import statistics as stat
import pandas_datareader as pdr
import matplotlib.pyplot as plt


data = pdr.get_data_yahoo(['^GSPC'], '2016-01-01', '2020-01-01')['Adj Close']

class SimpleStrategy :
    """
    Defines a Buy&Hold Strategy considering an equally weighted allocation within the portfolio

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, prices, capital=100, risk_free=0.02):
        self.prices = prices
        self.capital = capital
        self.risk_free = risk_free
        self.weights = [1 / self.prices.shape[1]] * len(self.prices.columns)
        self.changes = []
        self.equity_line = []
        self.cumulative = []
        self.std_dev = []
        self.sharpe_roll = []

    def buy_and_hold(self):
        if isinstance(self.prices, pd.DataFrame):
            initial_price = self.prices.iloc[0, :]
            for idx, row in self.prices.iloc[1:, :].iterrows():
                current_price = row
                price_diff = (current_price - initial_price) / initial_price
                self.changes.append(sum(price_diff * self.weights))
                self.equity_line.append((self.changes[-1] + 1) * self.capital)
                if len(self.changes) > 1:
                    self.std_dev.append ( stat.stdev ( self.changes ) )

        return self.changes, self.std_dev, self.equity_line

    def cumulative_ret(self):

        for i in range(len(self.changes)):
            if i == 0:
                self.cumulative.append(self.changes[i])
            elif i > 0:
                self.cumulative.append(self.changes[i] + self.changes[i - 1])

        return self.cumulative

    def sharpe_ratio(self):

        for i in range(len(self.changes)):
            if i == 0:
                continue
            elif i > 0:
                sharpe = (self.changes [ i ] - self.risk_free) / self.std_dev [ i - 1]
                self.sharpe_roll.append ( sharpe )
        return self.sharpe_roll

    def run(self):
        ret, vol, equity_line = self.buy_and_hold()
        cumulative = self.cumulative_ret()
        sharpe = self.sharpe_ratio()

        return ret, vol, equity_line, cumulative, sharpe


if __name__ == '__main__':
    strat = SimpleStrategy(data)
    strat_ret, strat_vol, equity_line, strat_cumret, strat_sharpe = strat.run()
    # strat_ret, strat_vol, equity_line = strat.buy_and_hold()
    # strat_cumret = strat.cumulative_ret()
    # strat_sharpe = strat.sharpe_ratio()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = 'col')
    ax1.plot(data.index[1:], equity_line, label = 'equity_line')
    ax1.grid()
    ax1.legend()
    ax2.plot(data.index[2:], strat_vol, label= 'rolling_vol')
    ax2.grid()
    ax2.legend()
    ax3.plot(data.index[2:], strat_sharpe, label = 'rolling_sharpe')
    ax3.grid()
    ax3.legend()
    plt.title('Buy & Hold GSPC')
    plt.xticks(rotation = 45)
    plt.show()
