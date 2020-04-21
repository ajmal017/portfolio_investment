from config import basedir
from utils.get_data import YahooData
import pandas as pd
import datetime
import matplotlib.pyplot as plt


stocks = pd.read_excel(f'{basedir}/nyse_tickers.xlsx')
stock_list = stocks['Symbol'].tolist()

ticker = ['GE', 'IBM', 'GOOG']
start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2020, 1, 1)
series = 'Adj Close'
ge = YahooData(['GE'], start, end, series).get_series()
ibm = YahooData(['IBM'], start, end, series).get_series()
google = YahooData(['GOOG'], start, end, series).get_series()

for item in (ge, ibm, google):
    item['30-day MA'] = item[item.columns[0]].rolling(window = 30).mean()
    item['30-day std'] = item[item.columns[0]].rolling(window = 30).std()

    item['upper_b'] = item['30-day MA'] + (item['30-day std'] * 2)
    item['lower_b'] = item['30-day MA'] - (item['30-day std'] * 2)


# 30-day Bollinger Band Google
plt.plot(google.drop(columns = ['30-day std']))
plt.title('30-Day Bollinger Band - Google')
plt.ylabel('Price $')
plt.xticks(rotation = 45)
plt.show()

# let's use a better plot style
plt.style.use('fivethirtyeight')
fig = plt.figure(figsize = (12, 6))
ax = fig.add_subplot(111)

x_axis = google.index.get_level_values(0)

ax.fill_between(x_axis, google['upper_b'], google['lower_b'], color = 'grey')

ax.plot(x_axis, google.iloc[:, 0], color = 'blue', lw = 1, label = 'Adj Close')
ax.plot(x_axis, google['30-day MA'], color = 'black', lw = 1, label = '30-day MA')

ax.set_title('30-Day Bollinger Band - Google')
ax.set_xlabel('Date (Year/Month')
ax.set_ylabel('Price $')
ax.legend()
plt.show()
