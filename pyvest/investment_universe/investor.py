import pyvest.investment_universe.investment_universe as inv_univ
from pyvest.general.portfolio import Portfolio
from pyvest.general.general import standard_utility_function

import numpy as np

from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds


class Investor:

    def __init__(self, investment_universe, gamma, wealth=None, portfolio=None,
                 utility_function=None, name=None,
                 optimization_tolerance=1e-6):

        self.__assign_investment_universe(investment_universe)
        self.__assign_gamma(gamma)
        self.__assign_wealth(wealth)
        self.__assign_portfolio(portfolio)
        self.__assign_name(name)
        self.__assign_utility_function(utility_function)
        self.__assign_optimization_tolerance(optimization_tolerance)

        self.__portfolio = None
        self.__optimal_portfolio = None

    def __repr__(self):
        return self.__generate_output()

    def __str__(self):
        return self.__generate_output()

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
    def portfolio(self):
        return self.__portfolio

    @portfolio.setter
    def portfolio(self, value):
        self.__assign_portfolio(value)

    @property
    def optimal_portfolio(self):
        return self.__optimal_portfolio

    @property
    def portfolio_utility(self):
        return self.__utility_function(self.__portfolio.weights,
                                       self.__portfolio.augmented_mu,
                                       self.__portfolio.augmented_cov,
                                       self.gamma)

    @property
    def optimal_portfolio_utility(self):
        return self.__utility_function(self.__optimal_portfolio.weights,
                                       self.__optimal_portfolio.augmented_mu,
                                       self.__optimal_portfolio.augmented_cov,
                                       self.gamma)

    @property
    def name(self):
        return self.__name

    ################################# PUBLIC ##################################

    def calculate_optimal_portfolio(self, x0=None):

        nb_risky_assets = len(self.__investment_universe.mu)
        nb_assets = nb_risky_assets if self.__investment_universe.r_f is None \
            else nb_risky_assets + 1

        # Initial guess (seed value)
        if x0 is None:
            x0 = np.ones(nb_assets) / nb_assets

        # Sum portfolio weights equals 1 constraint
        sum_weights_assets_equals_one_constraint = LinearConstraint(
            np.ones(nb_assets), 1, 1)

        if self.__investment_universe.min_weight is None \
                and self.__investment_universe.min_weight_r_f is None:
            min_weights = None
        elif self.__investment_universe.r_f is None:
            min_weights = self.__investment_universe.min_weight
        else:
            min_weights = np.append(self.__investment_universe.min_weight,
                                    self.__investment_universe.min_weight_r_f)

        min_weights_bound = Bounds(min_weights, np.inf) \
            if min_weights is not None else None

        optimal_portfolio_result = minimize(
            lambda x: - self.__utility_function(
                x, self.__investment_universe.augmented_mu,
                self.__investment_universe.augmented_cov, self.__gamma), x0,
            bounds=min_weights_bound,
            constraints=[sum_weights_assets_equals_one_constraint],
            tol=self.__optimization_tolerance)

        # Assign results
        self.__optimal_portfolio = Portfolio(
            optimal_portfolio_result.x, self.__investment_universe.mu,
            self.__investment_universe.cov, r_f=self.__investment_universe.r_f)

        if self.__portfolio is None:
            self.__portfolio = self.__optimal_portfolio

        return self.__optimal_portfolio

    def calculate_indifference_curve(self, utility, min_std=0, max_std=10,
                                     nb_points=1000):
        # TODO: MAKE IT WORK FOR A GENERAL UTILITY FUNCTION

        delta = (max_std - min_std) / (nb_points - 1)

        std_array = np.arange(min_std, max_std + delta, delta)
        mu_array = utility + self.gamma * std_array ** 2

        return std_array, mu_array

    def calculate_portfolio_utility(self, portfolio):

        if isinstance(portfolio, Portfolio):
            portfolio_obj = portfolio
        elif (isinstance(portfolio, list) or isinstance(portfolio, np.ndarray)
              or isinstance(portfolio, tuple)):
            portfolio_obj = Portfolio(portfolio, self.investment_universe.mu,
                                      self.investment_universe.cov,
                                      r_f=self.investment_universe.r_f,
                                      assets=self.investment_universe.assets)
        else:
            raise TypeError("The variable 'portfolio' must be an object of "
                            "type Portfolio or a list of weights.")

        return self.__utility_function(portfolio_obj.weights,
                                       portfolio_obj.augmented_mu,
                                       portfolio_obj.augmented_cov, self.gamma)

    ################################ PRIVATE ##################################

    def __assign_investment_universe(self, investment_universe):
        self.__investment_universe = investment_universe
        if isinstance(investment_universe, inv_univ.InvestmentUniverse):
            self.__investment_universe = investment_universe
        else:
            raise TypeError("The parameter 'investment_universe' must be an "
                            "instance of InvestmentUniverse.")

    def __assign_utility_function(self, utility_function):
        if utility_function is None:
            self.__utility_function = standard_utility_function
        elif callable(utility_function):
            self.__utility_function = utility_function
        else:
            raise TypeError("The parameter 'utility_function' must be "
                            "callable.")

    def __assign_gamma(self, gamma):
        if type(gamma) is float or type(gamma) is int:
            self.__gamma = float(gamma)
        else:
            raise TypeError("The parameter 'gamma' must of type float or int.")

    def __assign_wealth(self, wealth):
        if wealth is None:
            self.__wealth = None
        elif type(wealth) is float or type(wealth) is int:
            self.__wealth = float(wealth)
        else:
            raise TypeError("The parameter 'wealth' must be None or of type "
                            "float or int.")

    def __assign_portfolio(self, portfolio):
        if portfolio is None or type(portfolio) is Portfolio:
            self.__portfolio = portfolio
        elif type(portfolio) is list \
                and self.__investment_universe is not None:
            self.__portfolio = Portfolio(
                portfolio, self.__investment_universe.mu,
                self.__investment_universe.cov,
                r_f=self.__investment_universe.r_f,
                assets=self.__investment_universe.assets)
        else:
            raise TypeError("The parameter 'portfolio' must be None or of "
                            "type Portfolio or list.")

        if self.__investment_universe is not None:
            self.__investment_universe.market_portfolio

    def __assign_name(self, name):
        if name is None or type(name) is str:
            self.__name = name
        else:
            raise TypeError("The parameter 'name' must be None or of type "
                            "str.")

    def __assign_optimization_tolerance(self, optimization_tolerance):
        if type(optimization_tolerance) is float:
            self.__optimization_tolerance = optimization_tolerance
        else:
            raise TypeError("The parameter 'optimization_tolerance' must be "
                            "of type float.")

    def __generate_output(self):

        if self.name is not None:
            output = "name: {}\n".format(self.name)
        else:
            output = ""

        output += "gamma: {}".format(str(self.gamma))

        if self.wealth is not None:
            output += "\nwealth: {}".format(str(self.wealth))

        if self.portfolio is not None:
            output += "\nportfolio:"
            for portfolio_line in str(self.portfolio).splitlines():
                output += "\n  -{}".format(portfolio_line)

            output += "\n  -{}: {}".format("utility", self.portfolio_utility)

        if self.optimal_portfolio is not None:
            output += "\noptimal portfolio:"
            for portfolio_line in str(self.optimal_portfolio).splitlines():
                output += "\n  -{}".format(portfolio_line)

            output += "\n  -{}: {}".format("utility",
                                           self.optimal_portfolio_utility)

        return output
