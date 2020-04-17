import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model
plt.style.use('ggplot')

index = pd.read_excel('/Users/Filippo/Desktop/Python/PerformanceGraphExport.xls')

del index['Dow Jones Sustainability World Index TR']
del index['Dow Jones Sustainability World Index NTR']
index.set_index('Effective date', inplace=True)

def stationarity_test(x, cutoff = None):
    result = adfuller(x)
    tstat = result[0]
    pval = result[1]
    if pval < cutoff:
        print(f'The series is likely to be stationary. Pval: {pval}')
        return 1
    else:
        print(f'The series is likely non stationary. Pval: {pval}')
        return 0


summary_prices = index.describe()

plt.plot(index)
plt.grid()
plt.show()

# sust_returns = np.log(index) - np.log(index).shift(1)
# sust_returns = sust_returns[1:]
# adfuller(sust_returns)

# sust_returns.hist(bins=50)
# plt.show()

# ret_info = sust_returns.describe()

# weekly_ret = sust_returns.resample('W').mean()

# Calculate exponential moving average
# Calculate MAvarages
index['12d_EMA'] = index['Dow Jones Sustainability World Index'].ewm(span=12).mean()
index['26d_EMA'] = index['Dow Jones Sustainability World Index'].ewm(span=26).mean()

index[['Dow Jones Sustainability World Index', '12d_EMA', '26d_EMA']].plot(figsize=(10, 5))
plt.show()

# Calculate Returns and analysis of volatility
index['returns'] = index['Dow Jones Sustainability World Index'].pct_change()

summary_ret = index['returns'].describe()
stationarity_test(index['returns'][1:], cutoff=0.05)
plot_acf(index['returns'][1:])  # autocorr returns
plt.show()
index['sq_returns'] = [x**2 for x in index['returns']]
plt.plot(index['sq_returns'])
plt.show()
plot_acf(index['sq_returns'][1:])
plt.show()

index['ret_volatility'] = index['returns'].rolling(window = 10).std()
plt.plot(index['ret_volatility'])
plt.grid()
plt.show()

end_t = int(len(index) * 0.7)
train, test = index['returns'][: end_t], index['returns'][end_t:]

model = arch_model(train, mean='Zero', vol='GARCH', p=3, q=3, dist='StudentsT')
fit = model.fit()

# Calculate MACD
index['MACD'] = index['26d_EMA'] - index['12d_EMA']

# Calculate Signal
index['Signal'] = index.MACD.ewm(span=9).mean()

index[['MACD', 'Signal']].plot(figsize=(10, 5))
plt.show()

# Define Signal
index['trading_signal'] = np.where(index['MACD'] > index['Signal'], 1, -1)

# Calculate Strategy Returns
index['strategy_returns'] = index.returns * index.trading_signal.shift(1)

# Calculate Cumulative Returns
cumulative_returns = (index.strategy_returns + 1).cumprod()-1

# Plot Strategy Returns
cumulative_returns.plot(figsize=(10, 5))
plt.legend()
plt.show()

# Total number of trading days in a year is 252
trading_days = 252

# Calculate CAGR by multiplying the average daily returns with number of trading days
annual_returns = ((1 + index.returns.mean())**(trading_days) - 1)*100

print('The CAGR is %.2f%%' % annual_returns)

# Calculate the annualised volatility
annual_volatility = index.returns.std() * np.sqrt(trading_days) * 100
print('The annualised volatility is %.2f%%' % annual_volatility)

# Assume the annual risk-free rate is 6%
risk_free_rate = 0.05
daily_risk_free_return = risk_free_rate/trading_days

# Calculate the excess returns by subtracting the daily returns by daily risk-free return
excess_daily_returns = index.returns - daily_risk_free_return

# Calculate the sharpe ratio using the given formula
sharpe_ratio = (excess_daily_returns.mean() /
                excess_daily_returns.std()) * np.sqrt(trading_days)
print('The Sharpe ratio is %.2f' % sharpe_ratio)
