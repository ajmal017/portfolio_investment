import math
import numpy as np
import pandas as pd


bdays_per_week = 5
bdays_per_month = 21
bdays_per_year = 262

weeks_per_year = 52
months_per_year = 12
quarters_per_year = 4

DAILY = 'daily'
WEEKLY = 'weekly'
MONTHLY = 'monthly'
QUARTERLY = 'quarterly'
YEARLY = 'yearly'

ann_factors = {
    DAILY: bdays_per_year,
    WEEKLY: weeks_per_year,
    MONTHLY: months_per_year,
    QUARTERLY: quarters_per_year,
    YEARLY: 1
}


def roll(*args, **kwargs):
    """
    Calculate given stat across a rolling tim period.

    Parameters
    ----------
    returns: pd.Series or np.ndarray
           Daily returns of the portfolio, noncumulative.
    factor_returns (optional): float / series
           Benchmark return to compare returns against.
    function:
           the function to roll for each rolling window.
    window (keyword): int
           the number of period included in each calculation.
    (other keywaord): other keywords required to be passed to the function
           in the 'function' argument may also be passed in.

    Returns
    -------
    np.ndarray, pd.Series
           depends on input type
           ndarray(s) --> ndarray
           Series(s) --> pd.Series
           A series or ndarray of the results of the stat across the rolling window
    """

    func = kwargs.pop('function')
    window = kwargs.pop('window')
    if len(args) > 2:
        raise ValueError('Cannot pass more than 2 return series')

    if len(args) == 2:
        if not isinstance(args[0], type(args[1])):
            raise ValueError('The two returns argument are not the same')

    if isinstance(args[0], np.ndarray):
        return _roll_ndarray(func, window, *args, **kwargs)

    return _roll_pandas(func, window, *args, **kwargs)


def up(returns, factor_returns, **kwargs):
    """
    Calculate a given stat filtering only positive factor return periods.

    Parameters
    ----------
    returns: pd.Series or np.ndarray
          Daily returns of portfolio, noncumulative.
    factor_returns (optional): float / series
          Benchmark return to compare returns against.
    function:
          the function to run for each rolling window.
    (other keywords): other keywords that are required to be passed to
          the function in the 'function' argument may also be passed in.

    Returns
    -------
    Same as the return of the function
    """
    func = kwargs.pop('function')
    returns = returns[factor_returns > 0]
    factor_returns = factor_returns[factor_returns > 0]
    return func(returns, factor_returns, **kwargs)


def down(returns, factor_returns, **kwargs):
    """
    Calculate a given stat filtering only negative factor return periods.

    Parameters
    ----------
    returns: pd.Series or np.ndarray
          Daily returns of portfolio, noncumulative.
    factor_returns (optional): float / series
          Benchmark return to compare returns against.
    function:
          the function to run for each rolling window.
    (other keywords): other keywords that are required to be passed to
          the function in the 'function' argument may also be passed in.

    Returns
    -------
    Same as the return of the function
    """

    func = kwargs.pop('funciton')
    returns = returns[factor_returns < 0]
    factor_returns = factor_returns[factor_returns < 0]
    return func(returns, factor_returns, **kwargs)


def _roll_ndarray(func, window, *args, **kwargs):
    data = []
    for i in range(window, len(args[0]) + 1):
        rets = [s[i - window:i] for s in args]
        data.append(func(*rets, **kwargs))
    return np.array(data)


def _roll_pandas(func, window, *args, **kwargs):
    data = {}
    index_values = []
    for i in range(window, len(args[0]) + 1):
        rets = [s.iloc[i - window:i] for s in args]
        index_value = args[0].index[i - 1]
        index_values.append(index_value)
        data[index_value] = func(*rets, **kwargs)
    return pd.Series(data, index=type(args[0].index)(index_values))


# def rolling_window(array, length, mutable=False):
#     """
#     Restride an arrray of shape
#         (X_0, ..., X_N)
#     into an array of shape
#         (length, X_0 - length + 1, ..., X_N)
#     where each slice at index i along the first axis is equivalent to
#         result[i] = array[length * i:length * (i + 1)]
#
#     Parameters
#     ----------
#     array : np.ndarray
#         The base array
#     length : int
#         Length of the synthetic first axis to generate.
#     mutable: bool, optional
#         Return a mutable array? The returned array shares the same memory as
#         the input array. This means that writes into the returned array affect
#         'array'. The returned array also uses strides to map the same values
#         to multiple indices. Writes to a single index may appear to change many
#         values in the returned array
#
#     Returns
#     -------
#     out : np.ndarray
#     """
#     if not length:
#         raise ValueError("Can't have 0-length window")
#
#     orig_shape = array.shape
#     if not orig_shape:
#         raise IndexError("Cannot restride a scalar")
#     elif orig_shape[0] < length:
#         raise IndexError(
#             "Cannot restride array of shape {shape} with"
#             "a window length of {len}".format(
#                 shape=orig_shape,
#                 len=length
#             )
#         )
#
#     num_windows = (orig_shape[0] - length + 1)
#     new_shape = (num_windows, length) + orig_shape[1:]
#
#     new_strides = (array.strides[0],) + array.strides
#
#     out = as_strided(array, new_shape, new_strides)
#     out.setflags(write=mutable)
#
#     return out



def annualization_factor(period, annualization):
    """
    Return annualization factor from period entered or if a custom
    value is passed in.
    Parameters
    ----------
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    Returns
    -------
    annualization_factor : float
    """
    if annualization is None:
        try:
            factor = ann_factors[period]
        except KeyError:
            raise ValueError(
                "Period cannot be '{}'. "
                "Can be '{}'.".format(
                    period, "', '".join(ann_factors.keys())
                )
            )
    else:
        factor = annualization
    return factor


def simple_ret(prices):
    """
    Compute simple returns from price TS.

    Parameters
    ----------
    prices: pd.Series, pd.DataFrame or np.ndarray
            Assets as columns. Index coerced  to be tz-aware

    Returns
    -------
    return: array-like
            Returns of assets in wide-format. Assets as columns
    """
    if isinstance(prices, (pd.DataFrame, pd.Series)):
        out = prices.pct_change().iloc[1:]
    else:
        out = np.diff(prices, axis=0)
        np.divide(out, prices[:-1], out=out)

    return out


def cum_returns(returns, starting_value=0, out=None):
    """
    Compute cum returns from simple returns.

    Parameters
    ----------
    returns: pd.Series, np.ndarray, or pd.DataFrame
             Returns of the portfolio as a percentage, noncumulative.
             - TS with decimal returns
             - Accepts 2-dim data -> each column is cumulated.
    starting_value : float, optimal
             The starting returns.
    out: array-like, optional
         Array to use as output buffer.
         If not passed, a new array will be created.

    Returns
    -------
    cum_returns : array-like
         Series of cum_returns.
    """
    if len(returns) < 1:
        return returns.copy()

    nanmask = np.isnan(returns)
    if np.any(nanmask):
        returns = returns.copy()
        returns[nanmask] = 0

    allocated_output = out is None
    if allocated_output:
        out = np.empty_like(returns)

    np.add(returns, 1, out=out)
    out.cumprod(axis = 0, out = out)

    if starting_value == 0:
        np.subtract(out, 1, out=out)
    else:
        np.multiply(out, starting_value, out=out)

    if allocated_output:
        if returns.ndim == 1 and isinstance(returns, pd.Series):
            out = pd.Series(out, index = returns.index)
        elif isinstance(returns, pd.DataFrame):
            out = pd.DataFrame(
                out, index = returns.index, columns = returns.columns
            )

    return out


def cum_ret_final(returns, starting_value=0):
    """
    Compute tot returns from simple returns.

    Parameters
    ----------
    returns: pd.DataFrame, pd.Series, or np.ndarray
           Noncumulative simple returns of one or more TS.
    starting_value: float, optional
           The starting returns.

    Returns
    -------
    tot_returns: pd.Series, np.ndarray, or float.
    If input is 1-dim (a series or 1D numpy array), the result is a scalar.
    If input is 2-dim (a DataFrame or 2D numpy array), the result is a 1D array
    containing cum returns for each column of input
    """
    if len(returns) == 0:
        return np.nan

    if isinstance(returns, pd.DataFrame):
        result = (returns + 1). prod()
    else:
        result = np.nanprod(returns + 1, axis=0)

    if starting_value == 0:
        result -= 1
    else:
        result *= starting_value

    return result


def aggregate_returns(returns, convert_to):
    """
    Aggregate returns by week, month or year.

    Parameters
    ----------
    returns : pd.Series
           Daily returns of the portfolio, noncumulative.
    convert_to : str
           Can be 'weekly', 'monthly', or 'yearly'.

    Returns
    -------
    aggregated_returns : pd.Series
    """

    def cumulate_ret(x):
        return cum_returns(x).iloc[-1]

    if convert_to == WEEKLY:
        grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
    elif convert_to == MONTHLY:
        grouping = [lambda x: x.year, lambda x: x.month]
    elif convert_to == QUARTERLY:
        grouping = [lambda x: x.year, lambda x: int(math.ceil(x.month / 3.))]
    elif convert_to == YEARLY:
        grouping = [lambda x: x.year]
    else:
        raise ValueError(
            'convert_to must be {}, {}, or {}'.format(WEEKLY, MONTHLY, YEARLY)
        )

    return returns.groupby(grouping).apply(cumulate_ret)


def annual_return(returns, period=DAILY, annualization=None):
    """
        Determines the mean annual growth rate of returns. This is equivilent
        to the compound annual growth rate.
        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Periodic returns of the portfolio, noncumulative.
        period : str, optional
            Defines the periodicity of the 'returns' data for purposes of
            annualizing. Value ignored if `annualization` parameter is specified.
            Defaults are::
                'monthly':12
                'weekly': 52
                'daily': 252
        annualization : int, optional
            Used to suppress default values available in `period` to convert
            returns into annual returns. Value should be the annual frequency of
            `returns`.
        Returns
        -------
        annual_return : float
            Annual Return as CAGR (Compounded Annual Growth Rate).
        """

    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)
    num_years = len(returns) / ann_factor
    ending_value = cum_ret_final(returns, starting_value = 1)

    return ending_value ** (1 / num_years) - 1

#
# def annual_volatility(returns, period=DAILY, alpha=2.0,
#                       annualization = None, out=None):
#     """
#         Determines the annual volatility of a portfolio.
#         Parameters
#         ----------
#         returns : pd.Series or np.ndarray
#             Periodic returns of the portfolio, noncumulative.
#         period : str, optional
#             Defines the periodicity of the 'returns' data for purposes of
#             annualizing. Value ignored if `annualization` parameter is specified.
#             Defaults are::
#                 'monthly':12
#                 'weekly': 52
#                 'daily': 252
#         alpha : float, optional
#             Scaling relation (Levy stability exponent).
#         annualization : int, optional
#             Used to suppress default values available in `period` to convert
#             returns into annual returns. Value should be the annual frequency of
#             `returns`.
#         out : array-like, optional
#             Array to use as output buffer.
#             If not passed, a new array will be created.
#         Returns
#         -------
#         annual_volatility : float
#         """
#
#     allocated_output = out is None
#     if allocated_output:
#         out = np.empty(returns.shape[1:])
#
#     returns_1d = returns.ndim == 1
#
#     if len(returns) < 2:
#         out[()] = np.nan
#         if returns_1d:
#             out = out.item()
#         return out
#
#     ann_factor = annualization_factor(period, annualization)
#     nanstd(returns, ddof=1, axis=0, out=out)
#     out = np.multiply(out, ann_factor ** (1.0 / alpha), out=out)
#     if returns_1d:
#         out = out.item()
#     return out

