import numpy as np
import math
import time


class OptionPricing:

    def __init__(self, S0, E, T, rf, sigma, iterations):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_option_sim(self):

        option_data = np.zeros([self.iterations, 2])  # first column of zeros, second column stores the payoff
        # the column of zeros is needed as payoff = max(0, S-E) for a call option

        rand = np.random.normal(0, 1, [1, self.iterations])
        # dimensions: 1-dimensional array with as many items as iterations

        # stock price equation
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2) + self.sigma * np.sqrt(self.T) * rand)

        option_data[:, 1] = stock_price - self.E  # S-E in order to compute the max (0, S-E)

        # average for MC Method
        # np.amax returns the max according to formula
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)

        return np.exp(-1.0 * self.rf * self.T) * average  # exp(-rT) discount factor

    def put_option_sim(self):

        option_data = np.zeros([self.iterations, 2])  # first column of zeros, second column stores the payoff
        # the column of zeros is needed as payoff = max(=, S-E) for a call option

        rand = np.random.normal(0, 1, [1, self.iterations])
        # dimensions: 1-dimensional array with as many items as iterations

        # stock price equation
        stock_price = self.S0 * np.exp(
            self.T * (self.rf - 0.5 * self.sigma ** 2) + self.sigma * np.sqrt(self.T) * rand)

        option_data[:, 1] = self.E - stock_price  # S-E in order to compute the max (0, S-E)

        # average for MC Method
        # np.amax returns the max according to formula
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)

        return np.exp(-1.0 * self.rf * self.T) * average  # exp(-rT) discount factor


if __name__ == " __main__ ":

    S0 = 100  # Stock price at time zero
    E = 100  # Strike
    T = 1  # Expiry
    rf = 0.05  # Rf rate
    sigma = 0.2  # volatility of underlying
    iterations = 1000000  # number of iterations

    model = OptionPricing(S0, E, T, rf, sigma, iterations)
    print('Call option price with MC approach ', model.call_option_sim())
    print('Put option price with MC approach ', model.put_option_sim())

# simulation = OptionPricing(100, 110, 1, 0.05, 0.2, 100000)