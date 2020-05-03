from distributions.standard_normal import standard_norm_pdf, standard_norm_cdf
from math import exp, log, sqrt


def d_j(j, S, K, r, v, T):
    """
    d_j = \frac{log(\frac{S}{K})+(r+(-1)^{j-1} \frac{1}{2}v^2)T}{v sqrt(T)}
    """
    return (log(S/K) + (r + ((-1)**(j-1))*0.5*v*v)*T)/(v*(T**0.5))


def vanilla_call_price(S, K, r, v, T):
    """
    Price of a European call option struck at K, with
    spot S, constant rate r, constant vol v (over the
    life of the option) and time to maturity T
    """
    return S * standard_norm_cdf(d_j(1, S, K, r, v, T)) - \
        K*exp(-r*T) * standard_norm_cdf(d_j(2, S, K, r, v, T))


def vanilla_put_price(S, K, r, v, T):
    """
    Price of a European put option struck at K, with
    spot S, constant rate r, constant vol v (over the
    life of the option) and time to maturity T
    """
    return -S * standard_norm_cdf(-d_j(1, S, K, r, v, T)) + \
        K*exp(-r*T) * standard_norm_cdf(-d_j(2, S, K, r, v, T))
