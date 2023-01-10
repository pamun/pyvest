from finlib.investment_universe import InvestmentUniverse
from finlib.general import Portfolio
from finlib.general import standard_utility_function

import numpy as np

from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds


class Investor:

    def __init__(self, investment_universe, gamma, utility_function=None,
                 optimization_tolerance=1e-6):

        self.__assign_investment_universe(investment_universe)
        self.__assign_gamma(gamma)
        self.__assign_utility_function(utility_function)
        self.__assign_optimization_tolerance(optimization_tolerance)

        self.__optimal_portfolio = None

    ################################# ATTRIBUTES ##############################

    @property
    def investment_universe(self):
        return self.__investment_universe

    @investment_universe.setter
    def investment_universe(self, value):
        self.__assign_investment_universe(value)

    @property
    def utility_function(self):
        return self.__utility_function

    @utility_function.setter
    def utility_function(self, value):
        self.__assign_utility_function(value)

    @property
    def gamma(self):
        return self.__gamma

    @gamma.setter
    def gamma(self, value):
        self.__assign_gamma(value)

    @property
    def optimal_portfolio(self):
        return self.__optimal_portfolio

    ################################# PUBLIC ##################################

    def calculate_optimal_portfolio(self, x0=None):

        nb_assets = len(self.__investment_universe.mu)

        # Initial guess (seed value)
        if x0 is None:
            x0 = np.ones(nb_assets) / nb_assets

        # Sum portfolio weights equals 1 constraint
        sum_weights_assets_equals_one_constraint = LinearConstraint(
            np.ones(nb_assets), 1, 1)
        min_weight_bound = Bounds(self.__investment_universe.min_weight,
                                  np.inf)

        optimal_portfolio_result = minimize(
            lambda x: - self.__utility_function(
                x, self.__investment_universe.mu,
                self.__investment_universe.cov, self.__gamma), x0,
            bounds=min_weight_bound,
            constraints=[sum_weights_assets_equals_one_constraint],
            tol=self.__optimization_tolerance)

        # Assign results
        self.__optimal_portfolio = Portfolio(optimal_portfolio_result.x,
                                             self.__investment_universe.mu,
                                             self.__investment_universe.cov)

        return self.__optimal_portfolio

    ################################ PRIVATE ##################################

    def __assign_investment_universe(self, investment_universe):
        if isinstance(investment_universe, InvestmentUniverse):
            self.__investment_universe = investment_universe
        else:
            raise TypeError("The variable 'investment_universe' must be an "
                            "instance of InvestmentUniverse.")

    def __assign_utility_function(self, utility_function):
        if utility_function is None:
            self.__utility_function = standard_utility_function
        elif callable(utility_function):
            self.__utility_function = utility_function
        else:
            raise TypeError("The variable 'utility_function' must be "
                            "callable.")

    def __assign_gamma(self, gamma):
        if type(gamma) is float or type(gamma) is int:
            self.__gamma = gamma
        else:
            raise TypeError("The variable 'gamma' must be of type float or "
                            "int.")

    def __assign_optimization_tolerance(self, optimization_tolerance):
        if type(optimization_tolerance) is float:
            self.__optimization_tolerance = optimization_tolerance
        else:
            raise TypeError("The variable 'optimization_tolerance' must be of "
                            "type float.")