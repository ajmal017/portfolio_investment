import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from utils.get_data import YahooData
import scipy.optimize as sco

plt.style.use('fivethirtyeight')
np.random.seed(777)


def portfolio_annualised_perf(weights, mean_ret, cov_matrix):
    returns = np.sum(mean_ret * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


def random_portfolios(num_port, mean_ret, cov_matrix, risk_free):
    results = np.zeros((3, num_port))
    weights_record = []

    for i in range(num_port):
        weights = np.random.random(len(mean_ret))
        weights /= np.sum(weights)
        weights_record.append(weights)
        port_std, port_ret = portfolio_annualised_perf(weights, mean_ret, cov_matrix)
        results[0, i] = port_std
        results[1, i] = port_ret
        results[2, i] = (port_ret - risk_free) / port_std

    return results, weights_record


def display_simulated_ef(mean_returns, cov_matrix, num_portf, risk_free):
    results, weights = random_portfolios(num_portf, mean_returns, cov_matrix, risk_free)

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index = dataframe.columns, columns = ['allocation'])
    max_sharpe_allocation.allocation = [round(i*100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index = dataframe.columns, columns = ['allocation'])
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    print("-" * 80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp, 2))
    print("Annualised Volatility:", round(sdp, 2))
    print("\n")
    print(max_sharpe_allocation)
    print("-" * 80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min, 2))
    print("Annualised Volatility:", round(sdp_min, 2))
    print("\n")
    print(min_vol_allocation)

    plt.figure(figsize = (10, 7))
    plt.scatter(results[0, :], results[1, :], c = results[2, :], cmap = 'YlGnBu', marker = 'o',
                s = 10, alpha = 0.3)
    plt.colorbar()
    plt.scatter(sdp, rp, marker = '*', color = 'r', s = 500, label = 'Maximum Sharpe ratio')
    plt.scatter(sdp_min, rp_min, marker = '*', color = 'g', s = 500, label = 'Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing = 0.8)
    plt.show()

if __name__ == '__main__':

    ticker = ['APPL', 'AMZN', 'GOOG', 'FB']
    start = datetime.datetime(2006, 1, 1)
    end = datetime.datetime(2017, 12, 31)
    series = 'Adj Close'
    dataframe = YahooData(ticker, start, end, series).get_series()
    dataframe.dropna(axis = 'columns', inplace = True)

    plt.figure(figsize = (14, 7))
    for c in dataframe.columns.values:
        plt.plot(dataframe.index, dataframe[c], lw = 3, label = c)
    plt.legend(loc = 'upper left', fontsize = 12)
    plt.ylabel('Price in $')
    plt.xticks(rotation = 45)
    plt.show()

    returns = dataframe.pct_change()
    plt.figure(figsize = (14, 7))
    for c in returns.columns.values:
        plt.plot(returns.index, returns[c], lw = 1, label = c)
    plt.legend(loc = 'upper left', fontsize = 12)
    plt.ylabel('Daily Returns')
    plt.xticks(rotation = 45)
    plt.show()

    mean_ret = returns.mean()
    cov_matrix = returns.cov()
    num_portf = 25000
    risk_free = 0.01
    display_simulated_ef(mean_ret, cov_matrix, num_portf, risk_free)
