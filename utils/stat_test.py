from statsmodels.tsa.stattools import coint, adfuller
import numpy as np

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

    for key , val in r[4].items():
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

def cointegration_test(x, y, cutoff= 0.05):
    res_x = adfuller_test(x)
    res_y = adfuller_test(y)

    if all([(res_x == res_y, res_x == 0)]):
        score, pvalue, _ = coint(x, y)
        if pvalue < cutoff:
            print(f'Coint test pval = {pvalue}. The two series are likey cointegrated')
        else:
            print(f'Coint test pval = {pvalue}. The two series are likey not cointegrated')
