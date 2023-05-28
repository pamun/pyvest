# from pyvest.investment_universe.investment_universe import InvestmentUniverse
from pyvest.general.portfolio import Portfolio
from pyvest.general.general import standard_utility_function

import numpy as np

from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds


class Investor:

    def __init__(self, investment_universe=None, wealth=0, weights=None,
                 gamma=None, utility_function=None,
                 optimization_tolerance=1e-6):

        self.__assign_investment_universe(investment_universe)
        self.__assign_gamma(gamma)
        self.__assign_utility_function(utility_function)
        self.__assign_optimization_tolerance(optimization_tolerance)
        self.__assign_wealth(wealth)
        self.__assign_weights(weights)

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
    def wealth(self):
        return self.__wealth

    @wealth.setter
    def wealth(self, value):
        self.__assign_wealth(value)

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, value):
        self.__assign_weights(value)

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

    def plot_indifference_curves(self, compare_with=None):
        pass

    ################################ PRIVATE ##################################

    def __assign_investment_universe(self, investment_universe):
        self.__investment_universe = investment_universe
        # if isinstance(investment_universe, InvestmentUniverse):
        #     self.__investment_universe = investment_universe
        # else:
        #     raise TypeError("The parameter 'investment_universe' must be an "
        #                     "instance of InvestmentUniverse.")

    def __assign_utility_function(self, utility_function):
        if utility_function is None:
            self.__utility_function = standard_utility_function
        elif callable(utility_function):
            self.__utility_function = utility_function
        else:
            raise TypeError("The parameter 'utility_function' must be "
                            "callable.")

    def __assign_gamma(self, gamma):
        if gamma is None:
            self.__gamma = None
        elif type(gamma) is float or type(gamma) is int:
            self.__gamma = float(gamma)
        else:
            raise TypeError("The parameter 'gamma' must be None or of type "
                            "float or int.")

    def __assign_wealth(self, wealth):
        if type(wealth) is float or type(wealth) is int:
            self.__wealth = float(wealth)
        else:
            raise TypeError("The parameter 'wealth' must be of type float or "
                            "int.")

    def __assign_weights(self, weights):
        if weights is None or type(weights) is list:
            self.__weights = weights
        else:
            raise TypeError("The parameter 'weights' must be None or of type "
                            "list.")

    def __assign_optimization_tolerance(self, optimization_tolerance):
        if type(optimization_tolerance) is float:
            self.__optimization_tolerance = optimization_tolerance
        else:
            raise TypeError("The parameter 'optimization_tolerance' must be "
                            "of type float.")