import pandas_datareader as pdr
from pandas_datareader import data, wb
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

risk_free = 0.05

def capm(start_date, end_date, ticker_1, ticker_2):

    # getting data from yahoo finance
    stock_1 = pdr.get_data_yahoo(ticker_1, start_date, end_date)
    stock_2 = pdr.get_data_yahoo(ticker_2, start_date, end_date)

    # creating monthly returns
    return_stock_1 = stock_1.resample('M').last()
    return_stock_2 = stock_2.resample('M').last()

    # building up a dataframe
    data = pd.DataFrame({'s_adjclose': return_stock_1['Adj Close'], 'm_adjclose' : return_stock_2['Adj Close']},
                        index=return_stock_1.index)
    # natural logarithm of returns
    data[['s_returns', 'm_returns']] = np.log(data[['s_adjclose', 'm_adjclose']]
                                              / data[['m_adjclose', 's_adjclose']].shift(1))
    data = data.dropna() # getting rid of NA

    # var-cov matrix -> symmetric
    covmat = np.cov(data['s_returns'], data['m_returns'])
    print(covmat)

    #computing beta
    beta = covmat[0,1] / covmat[1,1]
    print('Beta from formula: ', beta)

    #fitting a line using linear regression [stock returns, market returns]. The slope is the beta
    beta, alpha = np.polyfit(data['m_returns'], data['s_returns'], deg=1)
    print('Beta from regression ', beta)

    #plot
    fig, axis = plt.subplots(1, figsize=(20, 10))
    axis.scatter(data['m_returns'], data['s_returns'], label='Data points')
    axis.plot(data['m_returns'], beta * data['m_returns'] + alpha, color='red', label='CAPM Line')
    plt.title('Capital Asset Pricing Model')
    plt.xlabel('Mkt Return $R_m$', fontsize=18)
    plt.ylabel('Stock Return $R_m$')
    plt.text(0.08, 0.05, r'$R_a = \beta * R_m + \alpha$', fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.show()

    #computing expected return
    exp_ret = risk_free + beta * (data['m_returns'].mean() * 12 - risk_free)
    print('Expected Return: ', exp_ret)

if name == '  main  ':
    # using historical data where mkt = S&P500
    capm('2010-01-01', '2019-01-01', 'IBM', '^GSPC')


