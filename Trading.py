# Import yfinance
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get the data for stock Facebook from 2017-04-01 to 2019-04-30
data = yf.download('AAPL', start="2017-04-01", end="2019-04-30")

plt.style.use('ggplot')

plt.figure(figsize=(10, 5))
data['Close'].plot(figsize=(10, 5))
plt.legend()
plt.show()

# Calculate exponential moving average
data['12d_EMA'] = data.Close.ewm(span=12).mean()
data['26d_EMA'] = data.Close.ewm(span=26).mean()

data[['Close', '12d_EMA', '26d_EMA']].plot(figsize=(10, 5))
plt.show()

# Calculate MACD
data['MACD'] = data['26d_EMA'] - data['12d_EMA']

# Calculate Signal
data['Signal'] = data.MACD.ewm(span=9).mean()

data[['MACD', 'Signal']].plot(figsize=(10, 5))
plt.show()

# Define Signal
data['trading_signal'] = np.where(data['MACD'] > data['Signal'], 1, -1)

# Calculate Returns
data['returns'] = data.Close.pct_change()

# Calculate Strategy Returns
data['strategy_returns'] = data.returns * data.trading_signal.shift(1)

# Calculate Cumulative Returns
cumulative_returns = (data.strategy_returns + 1).cumprod()-1

# Plot Strategy Returns
cumulative_returns.plot(figsize=(10, 5))
plt.legend()
plt.show()

# Total number of trading days in a year is 252
trading_days = 252

# Calculate CAGR by multiplying the average daily returns with number of trading days
annual_returns = ((1 + data.returns.mean())**trading_days - 1)*100

print('The CAGR is %.2f%%' % annual_returns)

# Calculate the annualised volatility
annual_volatility = data.returns.std() * np.sqrt(trading_days) * 100
print('The annualised volatility is %.2f%%' % annual_volatility)

# Assume the annual risk-free rate is 6%
risk_free_rate = 0.06
daily_risk_free_return = risk_free_rate/trading_days

# Calculate the excess returns by subtracting the daily returns by daily risk-free return
excess_daily_returns = data.returns - daily_risk_free_return

# Calculate the sharpe ratio using the given formula
sharpe_ratio = (excess_daily_returns.mean() /
                excess_daily_returns.std()) * np.sqrt(trading_days)
print('The Sharpe ratio is %.2f' % sharpe_ratio)
