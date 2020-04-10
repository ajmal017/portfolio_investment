import pandas as pd
import numpy as np
from math import sqrt
from pandas_datareader import data
from config import basedir
import matplotlib.pyplot as plt

stocks = pd.read_excel(f'{basedir}/nyse_tickers.xlsx')
stock_list = stocks['Symbol'].tolist()

frames = []

for stock in stock_list[: 10]:
    try:
        df = data.DataReader(stock, 'yahoo', start = '1/1/2000')

        df['std'] = df['Close'].rolling(window = 20).std()
        df['Moving Average'] = df['Close'].rolling(window = 20).mean()

        df['Criteria'] = (df['Open'] - df['Low'].shift(1)) < df['std']
        df['Criteria_2'] = df['Open'] > df['Moving Average']

        df['buy'] = df['Criteria'] & df['Criteria_2']

        df['pct_change'] = (df['Close'] - df['Open']) / df['Open']

        df['returns'] = df['pct_change'][df['buy'] == True]

        frames.append(df['returns'])

    except:
        pass

masterFrame = pd.concat(frames, axis = 1)
masterFrame['total'] = masterFrame.sum(axis = 1)
masterFrame['count'] = masterFrame.count(axis = 1) - 1
masterFrame['return'] = masterFrame['total'] / masterFrame['count']

if __name__ == "__main__":
    masterFrame [ 'return' ].dropna ( ).cumsum ( ).plot ( )
    plt.show()
    days = len(masterFrame [ 'return' ].dropna ( ).cumsum ( ))
    sharpe = (masterFrame['return'].mean() * 252) / (masterFrame['return'].std() * (sqrt(252)))
    annual_ret = (masterFrame['return'].dropna().cumsum()[-1] + 1) ** (365.0/days) - 1

