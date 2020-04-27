from utils.get_data import YahooData
from utils.kalman_filter import kalman_filter_average, kalman_filter_regression
from config import basedir
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from math import sqrt
from utils.stat_test import find_cointegrated_pairs, half_life
plt.style.use('bmh')

stocks = pd.read_excel(f'{basedir}/nyse_tickers.xlsx')
stock_list = stocks['Symbol'].tolist()


class PairsKalmanTrading:
    """
    Parameters
    ----------
    dataframe: pandas DataFrame of prices

    Returns
    -------
    final_res: cumulative returns as pandas DataFrame
    sharpe_final: Sharpe Ratio
    CAGR_final: Compund Annual Growth Rate
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.training_p, self.backtest_p = self.split_sample()
        self.pairs = self.coint_pairs()

    def split_sample(self):
        split = int(len(self.dataframe) * 0.4)
        train_p = self.dataframe[: split]
        backt_p = self.dataframe[split:]
        return train_p, backt_p

    def coint_pairs(self):
        _, pval_m, pairs = find_cointegrated_pairs(self.training_p)
        fig, ax = plt.subplots(figsize = (15, 10))
        sns.heatmap(pval_m, xticklabels = tickers, yticklabels = tickers, cmap = 'RdYlGn_r',
                    mask = (pval_m >= 0.05), ax = ax)
        plt.show()

        return pairs

    def backtest(self):
        results = []
        for pair in self.pairs:
            x = self.backtest_p[pair[0]]
            y = self.backtest_p[pair[1]]
            df1 = pd.DataFrame({'y': y, 'x': x})
            df1.index = pd.to_datetime(df1.index)
            state_means = kalman_filter_regression(kalman_filter_average(x), kalman_filter_average(y))
            df1['hr'] = - state_means[:, 0]
            df1['spread'] = df1.y + (df1.x * df1.hr)

            # calculate half life
            halflife = half_life(df1['spread'])

            # calculate z-score with window = half life period
            mean_spread = df1.spread.rolling(window = halflife).mean()
            std_spread = df1.spread.rolling(window = halflife).std()
            df1['zScore'] = (df1.spread - mean_spread) / std_spread

            # trading logic
            entry_zscore = 2
            exit_zscore = 0

            # set up num units long
            df1['long entry'] = ((df1.zScore < - entry_zscore) & (df1.zScore.shift(1) > - entry_zscore))
            df1['long exit'] = ((df1.zScore > - exit_zscore) & (df1.zScore.shift(1) < - exit_zscore))
            df1['num units long'] = np.nan
            df1.loc[df1['long entry'], 'num units long'] = 1
            df1.loc[df1['long exit'], 'num units long'] = 0
            df1['num units long'][0] = 0
            df1['num units long'] = df1['num units long'].fillna(method = 'pad')
            # set up num units short
            df1['short entry'] = ((df1.zScore > entry_zscore) & (df1.zScore.shift(1) < entry_zscore))
            df1['short exit'] = ((df1.zScore < exit_zscore) & (df1.zScore.shift(1) > exit_zscore))
            df1.loc[df1['short entry'], 'num units short'] = -1
            df1.loc[df1['short exit'], 'num units short'] = 0
            df1['num units short'][0] = 0
            df1['num units short'] = df1['num units short'].fillna(method = 'pad')
            df1['numUnits'] = df1['num units long'] + df1['num units short']
            df1['spread pct ch'] = (df1['spread'] - df1['spread'].shift(1)) / (
                        (df1['x'] * abs(df1['hr'])) + df1['y'])
            df1['port rets'] = df1['spread pct ch'] * df1['numUnits'].shift(1)
            df1['cum rets'] = df1['port rets'].cumsum()
            df1['cum rets'] = df1['cum rets'] + 1

            try:
                sharpe = ((df1['port rets'].mean() / df1['port rets'].std()) * sqrt(252))
            except ZeroDivisionError:
                sharpe = 0.0

            start_val = 1
            end_val = df1['cum rets'].iat[-1]
            start_date = df1.iloc[0].name
            end_date = df1.iloc[-1].name
            days = (end_date - start_date).days
            CAGR = round(((float(end_val) / float(start_val)) ** (252.0 / days)) - 1, 4)
            trad_res = df1['cum rets']
            results.append(df1['cum rets'])

            # plot pair - results
            print("The pair {} and {} produced a Sharpe Ratio of {} and a CAGR of {}".format(
                pair[0], pair[1], round(sharpe, 2), round(CAGR, 4)))
            trad_res.plot(figsize = (20, 15), legend = True)
            plt.show()

        results_df = pd.concat(results, axis = 1).dropna()

        results_df /= len(results_df.columns)  # to get equally weighted  curves
        final_res = results_df.sum(axis = 1).to_frame()  # final equity line
        final_res.plot(figsize = (20, 15))
        plt.show()

        # calculate and print our some final stats for our combined equity curve
        sharpe_final = (final_res.pct_change().mean() / final_res.pct_change().std()) * (sqrt(252))
        start_val = 1
        end_val = final_res.iloc[-1]
        start_date = final_res.index[0]
        end_date = final_res.index[-1]
        days = (end_date - start_date).days
        CAGR_final = round(((float(end_val) / float(start_val)) ** (252.0 / days)) - 1, 4)
        print("Sharpe Ratio is {} and CAGR is {}".format(round(sharpe_final, 2), round(CAGR_final, 4)))

        return final_res, sharpe_final, CAGR_final


if __name__ == '__main__':
    # Download the data from Yahoo #
    tickers = ['AAPL', 'ADBE', 'EBAY', 'GE', 'GOOG', 'IBM', 'MSFT', 'QCOM', 'HPQ', 'JNPR', 'AMD', 'IBM', 'SPY']
    start = datetime(2006, 1, 1)
    end = datetime(2020, 1, 1)
    series = 'Adj Close'
    dataframe = YahooData(tickers, start, end, series).get_series()

    trade = PairsKalmanTrading(dataframe)
    rets, sharpe, cagr = trade.backtest()
