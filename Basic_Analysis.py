import pandas_datareader as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

ibm = pdr.get_data_yahoo('IBM', '2016-01-01', '2019-01-01')
                          #start=datetime.datetime(2006, 1, 1),
                          #end=datetime.datetime(2019, 1, 1))

ibm.head()  # first rows
ibm.tail()  # last rows

summary = ibm.describe()

# ibm.index  # inspect the index
# ibm.columns  # inspect columns
ts = ibm['Close'][-10:]  # extract last 10 obs

print(ibm.loc[pd.Timestamp('2006-11-01'):pd.Timestamp('2006-12-31')].head())  # inspect first rows Nov & Dec
print(ibm.loc['2007'].head())  # inspect first rows 2007
print(ibm.iloc[22:43])  # inspect Nov 2006
print(ibm.iloc[[22, 43], [0, 3]])  # inspect Open and Close at 2006-11-01 and 2006-12-01

# 20-row sample
sample = ibm.sample(20)
print(sample)

# Resample to monthly level
monthly_ibm = ibm.resample('M').mean()
print(monthly_ibm)

ibm['diff'] = ibm.Open-ibm.Close
del ibm['diff']

ibm['Close'].plot(grid=True)
plt.show()

daily_close = ibm['Adj Close']
daily_pct_change = daily_close.pct_change()
# or daily_pct_change = daily_close / daily_close.shift(1) - 1
daily_pct_change.fillna(0, inplace=True)
print(daily_pct_change)

daily_log_ret = np.log(daily_close.pct_change()+1)
# or daily_log_ret = np.log(daily_close / daily_close.shift(1))
print(daily_log_ret)

monthly = ibm.resample('BM').apply(lambda x: x[-1])

daily_pct_change.hist(bins=50)
plt.show()

print(daily_pct_change.describe())

cum_daily_return = (1 + daily_pct_change).cumprod()
print(cum_daily_return)

cum_daily_return.plot(grid=True, figsize=(12, 8))
plt.show()

cum_monthly_ret = cum_daily_return.resample('M').mean()
print(cum_monthly_ret)
