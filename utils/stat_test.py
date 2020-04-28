from statsmodels.tsa.stattools import coint, adfuller
import scipy.stats as stats
import numpy as np
import statsmodels.api as sm

r = np.random.normal(0, 1, 100)


def jarque_bera_test(data, cutoff=0.05):
    """
    Returns the result for a JB test. If the data come from a normal distribution, the JB stat
    has a chi-squared distribution with two degrees of freedom.
    JB = n(S^2 / 6 + (K - 3)^2 / 24)

    Parameters
    ----------
    data: univariate series
    cutoff: level of significance of the test

    """
    n = data.shape[0]

    if n < 500:
        print('Warning: JB Test wokrs better with large sample sizes (> 500)')
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)

    S = float(n) * (skew ** 2 / 6 + (kurt - 3) ** 2 / 24)

    t = stats.chi2(2).ppf(cutoff)
    if S < t:
        print(f'No evidence to reject as non-Normal according to the Jarque-Bera test.'
              f'\n t_stat {S} < critical_value {t}')
    else:
        print(f"Reject that is Normal according to the Jarque-Bera test."
              f"\n t_stat {S} > critical_value {t}")


def adfuller_test(series, signif=0.05, name='', verbose=False):
    r = adfuller(series, autolag = 'AIC')
    output = {'test_stat': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
    p_value = output['pvalue']

    def adjust(val, length=6): return str(val).ljust(length)

    # Summary
    print(f'ADFuller Test on "{name}"', "\n   ", '-'*47)
    print(f'Null Hypothesis: Data has unit root -> non-stationary')
    print(f'Significance level = {signif}')
    print(f'Test Stat = {output["test_stat"]}')
    print(f'N. of lag chosen = {output["n_lags"]}')

    for key, val in r[4].items():
        print(f'Critical value {adjust(key)} = {round(val , 3)}')
    if p_value <= signif:
        print(f' -> P-value = {p_value}. Reject Null Hypothesis')
        print(f' -> Series is likely Stationary')
        res = 1
        return res
    else:
        print(f' -> P-value = {p_value}. Cannot reject Null Hypothesis')
        print(f'Series is likely not stationary')
        res = 0
        return res


def find_cointegrated_pairs(data):
    """
    Defines a function too look for cointegrated assets within the dataset. Default level of confidence = 5%

    Parameters
    ----------
    data: pd:DataFrame of returns

    Returns
    -------
    score_matrix : matrix of results of cointegration tests
    pval_matrix: matrix of p-values
    pairs: list of cointegrated assets

    """
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []

    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue

            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs


def cointegration_test(x, y, cutoff=0.05):
    res_x = adfuller_test(x)
    res_y = adfuller_test(y)

    if all([(res_x == res_y, res_x == 0)]):
        score, pvalue, _ = coint(x, y)
        if pvalue < cutoff:
            print(f'Coint test pval = {pvalue}. The two series are likey cointegrated')
        else:
            print(f'Coint test pval = {pvalue}. The two series are likey not cointegrated')


def half_life(spread):
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    spread_ret = spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    spread_lag2 = sm.add_constant(spread_lag)
    model = sm.OLS(spread_ret, spread_lag2)
    res = model.fit()
    halflife = int(round(-np.log(2) / res.params[1], 0))
    if halflife <= 0:
        halflife = 1
    return halflife


