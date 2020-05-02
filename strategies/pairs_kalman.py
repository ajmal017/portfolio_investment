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
plt.style.use('ggplot')

stocks = pd.read_excel(f'{basedir}/nyse_tickers.xlsx')
stock_list = stocks['Symbol'].tolist()


class PairsKalmanTrading:
    """
    Parameters
    ----------
    dataframe: pandas DataFrame of prices

    Returns
    -------
    final_res: cumulative returns as pandas Series
    sharpe_final: Sharpe Ratio
    CAGR_final: Compund Annual Growth Rate
    """

    def __init__(self, dataframe, benchmark):
        self.dataframe = dataframe
        self.benchmark = benchmark
        # self.benchmark_ret = benchmark.pct_change().iloc[1:]
        self.training_p, self.backtest_p, self.benchmark_ret = self.split_sample()
        self.pairs = self.coint_pairs()

    def split_sample(self):
        split = int(len(self.dataframe) * 0.4)
        train_p = self.dataframe[: split]
        backt_p = self.dataframe[split:]
        benchmark_ret = self.benchmark[split:].pct_change().iloc[1:]
        return train_p, backt_p, benchmark_ret

    def coint_pairs(self):
        _, pval_m, pairs = find_cointegrated_pairs(self.training_p)
        fig, ax = plt.subplots(figsize = (15, 10))
        sns.heatmap(pval_m, xticklabels = tickers, yticklabels = tickers, cmap = 'RdYlGn_r',
                    mask = (pval_m >= 0.05), ax = ax)
        plt.show()

        return pairs

    def backtest(self):
        results = []
        result_cum = []
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

            # define portfolio returns
            df1['port rets'] = df1['spread pct ch'] * df1['numUnits'].shift(1)
            df1['cum rets'] = df1['port rets'].cumsum()
            df1['equity_line'] = df1['cum rets'] + 1

            try:
                sharpe = ((df1['port rets'].mean() / df1['port rets'].std()) * sqrt(252))
            except ZeroDivisionError:
                sharpe = 0.0

            start_val = 1
            end_val = df1['equity_line'].iat[-1]
            start_date = df1.iloc[0].name
            end_date = df1.iloc[-1].name
            days = (end_date - start_date).days
            CAGR = round(((float(end_val) / float(start_val)) ** (252.0 / days)) - 1, 4)
            results.append(df1['port rets'])
            result_cum.append(df1['cum rets'])

            # plot pair - results
            print("The pair {} and {} produced a Sharpe Ratio of {} and a CAGR of {}".format(
                pair[0], pair[1], round(sharpe, 2), round(CAGR, 4)))

        results_df = pd.concat(results, axis = 1).dropna()
        cum_results_df = pd.concat(result_cum, axis = 1).dropna()

        results_df /= len(results_df.columns)  # to get equally weighted  curves
        final_res = results_df.sum(axis = 1)
        cum_results_df /= len(cum_results_df.columns)
        final_cum_results = cum_results_df.sum(axis = 1)
        (final_cum_results+1).plot(figsize = (20, 15))
        std = final_res.std()

        # calculate and print our some final stats for our combined equity curve
        sharpe_final = (final_res.mean() / std) * np.sqrt(252)
        start_val = 1
        end_val = final_cum_results.iloc[-1] + 1
        start_date = final_cum_results.index[0]
        end_date = final_cum_results.index[-1]
        days = (end_date - start_date).days
        CAGR_final = round(((float(end_val) / float(start_val)) ** (252.0 / days)) - 1, 4)


        diff_ret = final_res.values - self.benchmark_ret.values
        diff_std = diff_ret.std()
        ann_information_ratio = (final_res.mean() - self.benchmark_ret.mean()) / diff_std * np.sqrt(252)

        return final_res, std, final_cum_results, sharpe_final, ann_information_ratio[0], CAGR_final


if __name__ == '__main__':
    # Download the data from Yahoo #
    tickers = ['AAPL', 'ADBE', 'EBAY', 'GE', 'GOOG', 'IBM', 'MSFT', 'QCOM', 'HPQ', 'JNPR', 'AMD', 'IBM', 'SPY']
    start = datetime(2006, 1, 1)
    end = datetime(2020, 1, 1)
    series = 'Adj Close'
    dataframe = YahooData(tickers, start, end, series).get_series()
    benchmark = YahooData(['SPY'], start, end, series).get_series()

    trade = PairsKalmanTrading(dataframe, benchmark)
    port_ret, port_vol, cum_ret, sharpe, info_ratio, cagr = trade.backtest()

    print ( f'Strategy Return from {cum_ret.index[0].date()} to {cum_ret.index[-1].date()} : '
            f'{(round(float(cum_ret.iloc[-1]) * 100, 2))} % and Annualised Volatility '
            f'is: {round(port_vol * np.sqrt(252) * 100, 2)}%')
    print(f'CAGR : {round(cagr * 100 ,2 )}%')
    print(f'Annualised Sharpe Ratio : {round(sharpe, 2)}')
    print(f'Annualised Information Ratio : {round(info_ratio, 2)}')

    plt.plot(cum_ret + 1)
    plt.title('Kalman_PairsTrading EW Portfolio')
    plt.legend('Equity Line', loc = 'upper left')
    plt.show()
