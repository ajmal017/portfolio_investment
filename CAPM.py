import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

risk_free = 0.05


def capm(start_date, end_date, ticker_1, mkt_ticker):

    # Retrieve data from yahoo finance
    stock_1 = pdr.get_data_yahoo(ticker_1, start_date, end_date)
    mkt = pdr.get_data_yahoo(mkt_ticker, start_date, end_date)

    # Create monthly returns
    return_stock_1 = stock_1.resample('M').last()
    return_mkt = mkt.resample('M').last()

    # Build up a dataframe
    data = pd.DataFrame({'s_adjclose': return_stock_1['Adj Close'], 'm_adjclose': return_mkt['Adj Close']},
                        index=return_stock_1.index)
    # Natural logarithm of returns
    data[['s_returns', 'm_returns']] = np.log(data[['s_adjclose', 'm_adjclose']]
                                              / data[['m_adjclose', 's_adjclose']].shift(1))
    data = data.dropna()  # getting rid of NA

    # Var-Cov matrix -> symmetric
    covmat = np.cov(data['s_returns'], data['m_returns'])
    print(covmat)

    # Compute beta
    beta_cov = covmat[0, 1] / covmat[1, 1]
    print('Beta from formula: ', round(beta_cov, 3))

    # Fit a line using linear regression [stock returns, market returns]. The slope is the beta
    beta_reg, alpha = np.polyfit(data['m_returns'], data['s_returns'], deg=1)
    print('Beta from regression ', round(beta_reg), 3)
    line = beta_reg * data['m_returns'] + alpha

    # Plot
    fig, axis = plt.subplots(1, figsize=(20, 10))
    axis.scatter(data['m_returns'], data['s_returns'], label='Data points')
    axis.plot(data['m_returns'], line, color='red', label='CAPM Line')
    plt.title('Capital Asset Pricing Model', fontsize = 25)
    plt.xlabel('Mkt Return $R_m$', fontsize = 15)
    plt.ylabel('Stock Return $R_m$', fontsize = 15)
    plt.text(0.08, 0.05, r'$R_a = \beta * R_m + \alpha$', fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Compute expected return
    exp_ret = risk_free + beta_reg * (data['m_returns'].mean() * 12 - risk_free)
    print('Expected Return: ', f'{round(exp_ret*100, 2)}%')



if __name__ == "__main__":
    # using historical data where mkt = S&P500
    capm('2010-01-01', '2019-01-01', 'IBM', '^GSPC')



