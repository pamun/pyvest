import numpy as np

import math

from pyvest.general.portfolio import Portfolio
from pyvest.general.general import calculate_portfolio_standard_deviation
from pyvest.general.general import calculate_portfolio_sharpe_ratio
from pyvest.investment_universe.investment_universe_visualizer import \
    InvestmentUniverseVisualizer

from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds

from pyvest.investment_universe.investor import Investor
from pyvest.investment_universe.portfolios_generator import \
    FeasiblePortfoliosGeneratorRandom, FeasiblePortfoliosGeneratorFrontier


class InvestmentUniverse:
    MAX_NB_ITERATIONS = 100

    def __init__(self, assets, mu, cov, r_f=None, min_weight=0,
                 min_weight_r_f=None, parameters=None):

        self.__assign_parameters(parameters)

        self.__assets = assets

        self.__nb_risky_assets = len(mu)
        self.__nb_assets = self.__nb_risky_assets if r_f is None \
            else self.__nb_risky_assets + 1

        self.__assign_r_f(r_f)
        self.__assign_mu(mu)
        self.__assign_cov(cov)
        self.__assign_min_weight(min_weight)
        self.__assign_min_weight_r_f(min_weight_r_f)

        self.__investors = {}
        self.__nb_unnamed_investors = 0

        self.__calculate_assets_std()

        self.__feasible_portfolios = None
        self.__feasible_portfolios_with_r_f = None
        self.__feasible_portfolios_surface = None
        self.__feasible_portfolios_surface_with_r_f = None
        self.__feasible_portfolios_equation = None
        self.__feasible_portfolios_equation_with_r_f = None
        self.__mvp = None
        self.__efficient_frontier = None
        self.__efficient_frontier_equation = None
        self.__tangency_portfolio = None
        self.__cal = None
        self.__cal_equation = None
        self.__other_portfolios = None
        self.__market_portfolio = None
        self.__total_wealth = None
        self.__investors = {}

        self.__min_weights_bound = None
        self.__sum_weights_assets_equals_one_constraint = None

        self.__efficient_mu_min = None
        self.__efficient_mu_max = None

        self.__visualizer = None

        self.__feasible_portfolios_generator_random = None
        self.__feasible_portfolios_generator_frontier = None
        self.__feasible_portfolios_generator_random_r_f = None
        self.__feasible_portfolios_generator_frontier_r_f = None

    ################################# ATTRIBUTES ##############################

    @property
    def assets(self):
        return self.__assets

    @assets.setter
    def assets(self, value):
        self.__assets = value

    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, value):
        self.__mu = np.array(value)

    @property
    def augmented_mu(self):
        return self.__augmented_mu

    @augmented_mu.setter
    def augmented_mu(self, value):
        self.__augmented_mu = np.array(value)

    @property
    def cov(self):
        return self.__cov

    @cov.setter
    def cov(self, value):
        self.__cov = np.array(value)

    @property
    def augmented_cov(self):
        return self.__augmented_cov

    @augmented_cov.setter
    def augmented_cov(self, value):
        self.__augmented_cov = np.array(value)

    @property
    def r_f(self):
        return self.__r_f

    @r_f.setter
    def r_f(self, value):
        self.__assign_r_f(value)

    @property
    def min_weight(self):
        return self.__min_weights

    @min_weight.setter
    def min_weight(self, value):
        self.__assign_min_weight(value)

    @property
    def min_weight_r_f(self):
        return self.__min_weight_r_f

    @min_weight_r_f.setter
    def min_weight_r_f(self, value):
        self.__assign_min_weight_r_f(value)

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, value):
        self.__assign_parameters(value)

    @property
    def std(self):
        return self.__std

    @property
    def feasible_portfolios(self):
        return self.__feasible_portfolios

    @property
    def feasible_portfolios_with_r_f(self):
        return self.__feasible_portfolios_with_r_f

    @property
    def feasible_portfolios_surface(self):
        return self.__feasible_portfolios_surface

    @property
    def feasible_portfolios_surface_with_r_f(self):
        return self.__feasible_portfolios_surface_with_r_f

    @property
    def feasible_portfolios_equation(self):
        return self.__feasible_portfolios_equation

    @property
    def feasible_portfolios_equation_with_r_f(self):
        return self.__feasible_portfolios_equation_with_r_f

    @property
    def mvp(self):
        return self.__mvp

    @property
    def efficient_frontier(self):
        return self.__efficient_frontier

    @property
    def efficient_frontier_equation(self):
        return self.__efficient_frontier_equation

    @property
    def tangency_portfolio(self):
        return self.__tangency_portfolio

    @property
    def cal(self):
        return self.__cal

    @property
    def cal_equation(self):
        return self.__cal_equation

    @property
    def other_portfolios(self):
        return self.__other_portfolios

    @property
    def visualizer(self):
        return self.__visualizer

    @property
    def investors(self):
        return self.__investors

    @property
    def market_portfolio(self):
        return self.__market_portfolio

    @property
    def total_wealth(self):
        return self.__total_wealth

    ################################# PUBLIC FUNCTIONS #################################

    def calculate_feasible_portfolios(self, nb_portfolios=20000,
                                      with_r_f=False,
                                      random=False):

        if self.__min_weights is None and not with_r_f:
            self.__feasible_portfolios_equation = \
                self.__calculate_feasible_portfolio_equation
        elif self.__min_weights is None and with_r_f:
            tangency_portfolio = self.__calculate_tangency_portfolio() \
                if self.tangency_portfolio is None else self.tangency_portfolio
            self.__feasible_portfolios_equation_with_r_f = \
                lambda x: self.__calculate_cal_equation(x, tangency_portfolio)
        else:
            self.__calculate_feasible_portfolios_with_min_weights(
                nb_portfolios, with_r_f, random)

    def __calculate_feasible_portfolios_with_min_weights(
            self, nb_portfolios=20000, with_r_f=False, random=False):

        if random and with_r_f:
            if self.__feasible_portfolios_generator_random_r_f is None:
                self.__feasible_portfolios_generator_random_r_f = \
                    FeasiblePortfoliosGeneratorRandom(self, with_r_f)
            portfolios_generator = \
                self.__feasible_portfolios_generator_random_r_f
        elif random and not with_r_f:
            if self.__feasible_portfolios_generator_random is None:
                self.__feasible_portfolios_generator_random = \
                    FeasiblePortfoliosGeneratorRandom(self, with_r_f)
            portfolios_generator = self.__feasible_portfolios_generator_random
        elif with_r_f:
            if self.__feasible_portfolios_generator_frontier_r_f is None:
                self.__feasible_portfolios_generator_frontier_r_f = \
                    FeasiblePortfoliosGeneratorFrontier(self, with_r_f)
            portfolios_generator = \
                self.__feasible_portfolios_generator_frontier_r_f
        else:
            if self.__feasible_portfolios_generator_frontier is None:
                self.__feasible_portfolios_generator_frontier = \
                    FeasiblePortfoliosGeneratorFrontier(self, with_r_f)
            portfolios_generator = \
                self.__feasible_portfolios_generator_frontier

        portfolios_generator.generate_portfolios(
            nb_portfolios=nb_portfolios, distance=0.05)
        if with_r_f:
            self.__feasible_portfolios_surface_with_r_f = \
                portfolios_generator.frontier
        else:
            self.__feasible_portfolios_surface = portfolios_generator.frontier

    def calculate_mvp(self, x0=None):

        # Initial guess (seed value)
        if x0 is None:
            x0 = np.ones(self.__nb_risky_assets) / self.__nb_risky_assets

        # Assign results
        self.__mvp = self.__calculate_mvp(x0)

        return self.__mvp

    def calculate_efficient_portfolio(self, expected_return=None,
                                      standard_deviation=None, name=None,
                                      x0=None, tolerance=None):

        tolerance = self.__parameters["optimization_tolerance"] \
            if tolerance is None else tolerance

        # Initial guess (seed value)
        if x0 is None:
            x0 = np.ones(self.__nb_risky_assets) / self.__nb_risky_assets

        mvp = self.__calculate_mvp(x0) if not self.__mvp else self.__mvp
        self.__calculate_efficient_mu_min_max(mvp)

        if expected_return is not None and standard_deviation is not None:
            raise ValueError("Only one of 'mu' and 'sigma' must be passed as "
                             "argument.")
        elif expected_return is not None:
            efficient_portfolio = \
                self.__calculate_efficient_portfolio_from_mu(expected_return,
                                                             x0, tolerance)
        elif standard_deviation is not None:
            efficient_portfolio = \
                self.__calculate_efficient_portfolio_from_sigma(
                    standard_deviation, x0, tolerance)
        else:
            raise ValueError("Either 'mu' or 'sigma' must be passed as "
                             "argument.")

        self.calculate_portfolio(efficient_portfolio, name)

        return efficient_portfolio

    def calculate_efficient_frontier(self, nb_portfolios=1000,
                                     x0=None,
                                     tolerance=None,
                                     return_portfolios=False):
        if self.__min_weights is None:
            mvp_mu_sigma = self.__calculate_mvp_mu_sigma_from_equation()
            self.__efficient_frontier_equation = {
                "equation": self.__calculate_feasible_portfolio_equation,
                "mvp": mvp_mu_sigma
            }
            efficient_frontier = self.__efficient_frontier_equation
        else:
            efficient_frontier = \
                self.__calculate_efficient_frontier_with_min_weights(
                    nb_portfolios, x0, tolerance, return_portfolios)

        if return_portfolios:
            return efficient_frontier

    def __calculate_efficient_frontier_with_min_weights(
            self, nb_portfolios=1000, x0=None, tolerance=None,
            return_portfolios=False):

        tolerance = self.__parameters["optimization_tolerance"] \
            if tolerance is None else tolerance

        # Initial guess (seed value)
        if x0 is None:
            x0 = np.ones(self.__nb_risky_assets) / self.__nb_risky_assets

        mvp = self.__calculate_mvp(x0) if not self.__mvp else self.__mvp

        self.__calculate_efficient_mu_min_max(mvp)
        efficient_mu_array = self.__calculate_efficient_mu_array(nb_portfolios)

        # Calculate the efficient portfolios
        self.__efficient_frontier = []
        for efficient_mu in efficient_mu_array:
            efficient_portfolio = \
                self.__calculate_efficient_portfolio_from_mu(efficient_mu, x0,
                                                             tolerance)
            self.__efficient_frontier.append(efficient_portfolio)

        if return_portfolios:
            return self.__efficient_frontier

    def calculate_tangency_portfolio(self, x0=None, tolerance=None):

        self.__tangency_portfolio = \
            self.__calculate_tangency_portfolio(x0, tolerance)

        return self.__tangency_portfolio

    def __calculate_tangency_portfolio(self, x0=None, tolerance=None):

        if self.__r_f is None:
            raise ValueError("You need to add a risk-free asset first!")

        tolerance = self.__parameters["optimization_tolerance"] \
            if tolerance is None else tolerance

        # Initial guess (seed value)
        if x0 is None:
            x0 = np.ones(self.__nb_risky_assets) / self.__nb_risky_assets

        # Sum portfolio weights equals 1 constraint
        self.__sum_weights_assets_equals_one_constraint = LinearConstraint(
            np.ones(self.__nb_risky_assets), 1, 1)

        tangency_portfolio_result = minimize(
            lambda x: -calculate_portfolio_sharpe_ratio(x, self.__mu,
                                                        self.__cov,
                                                        self.__r_f),
            x0,
            bounds=self.__min_weights_bound,
            constraints=[self.__sum_weights_assets_equals_one_constraint],
            tol=tolerance)

        tangency_portfolio_weights = np.append(tangency_portfolio_result.x,
                                               [0])

        tangency_portfolio = Portfolio(tangency_portfolio_weights, self.__mu,
                                       self.__cov, r_f=self.__r_f,
                                       assets=self.__assets)

        return tangency_portfolio

    def calculate_cal(self, return_portfolios=False):

        tangency_portfolio = self.__calculate_tangency_portfolio() \
            if self.tangency_portfolio is None else self.tangency_portfolio

        if self.__min_weights is None:
            self.__cal_equation = \
                lambda x: self.__calculate_cal_equation(x, tangency_portfolio)
        else:
            self.__calculate_cal_with_min_weights(tangency_portfolio)

        if return_portfolios:
            return self.__cal

    def __calculate_cal_with_min_weights(self, tangency_portfolio):

        min_fraction, max_fraction, step_fraction = self.__get_cal_parameters()

        self.__cal = []
        for tangency_weight in np.arange(min_fraction,
                                         max_fraction,
                                         step_fraction):
            cal_weights = tangency_weight * tangency_portfolio.weights
            cal_weights[-1] = 1 - tangency_weight

            cal_portfolio = Portfolio(cal_weights, self.__mu,
                                      self.__cov, self.__r_f,
                                      assets=self.__assets)
            self.__cal.append(cal_portfolio)

        return self.__cal

    def calculate_portfolio_on_cal(self, expected_return=None,
                                   standard_deviation=None,
                                   weight_r_f=None, weight_tangency=None,
                                   name=None):

        nb_not_none = sum([
            x is not None for x in [expected_return, standard_deviation,
                                    weight_r_f, weight_tangency]])
        if nb_not_none != 1:
            raise ValueError("One and only one of 'mu', 'sigma', 'weight_r_f' "
                             "and 'weight_tangency' must be passed as "
                             "argument.")

        tangency_portfolio = self.__calculate_tangency_portfolio() \
            if self.tangency_portfolio is None else self.tangency_portfolio

        if expected_return is not None:
            standard_deviation = self.__calculate_cal_equation(
                expected_return, tangency_portfolio)

        if standard_deviation is not None:
            weight_tangency = \
                standard_deviation / tangency_portfolio.standard_deviation

        if weight_r_f is not None:
            weight_tangency = 1 - weight_r_f

        if self.__min_weights is not None:
            _, max_fraction, _ = self.__get_cal_parameters()
            if weight_tangency > max_fraction:
                raise ValueError("The value of 'weight_tangency' must be less "
                                 "than {}.".format(max_fraction))

        portfolio_weights = weight_tangency * tangency_portfolio.weights
        portfolio_weights[-1] = 1 - weight_tangency

        portfolio = self.calculate_portfolio(portfolio_weights, name)

        return portfolio

    def calculate_portfolio(self, portfolio, name=None):

        if isinstance(portfolio, Portfolio):
            portfolio_obj = portfolio
        elif (isinstance(portfolio, list)
              or isinstance(portfolio, np.ndarray)
              or isinstance(portfolio, tuple)) \
                and ((len(portfolio) == self.__nb_risky_assets
                      and self.__r_f is None)
                     or (len(portfolio) == self.__nb_risky_assets + 1
                         and self.__r_f is not None)):
            portfolio_obj = Portfolio(portfolio, self.__mu, self.__cov,
                                      r_f=self.__r_f, assets=self.__assets)
        else:
            raise TypeError("The variable 'portfolio' must be an object of "
                            "type Portfolio or a list of weights of dimension "
                            "{}.".format(self.__nb_assets))

        if self.__other_portfolios is None:
            self.__other_portfolios = {}

        self.__other_portfolios[tuple(portfolio_obj.weights)] = \
            (portfolio_obj, name)

        return portfolio_obj

    def remove_portfolio(self, portfolio=None):
        if portfolio is None:
            self.__other_portfolios = None
        elif isinstance(portfolio, list) or isinstance(portfolio, tuple) \
                or isinstance(portfolio, np.ndarray):
            del self.__other_portfolios[tuple(portfolio)]
        elif isinstance(portfolio, Portfolio):
            del self.__other_portfolios[portfolio.weights]
        elif isinstance(portfolio, str):
            keys_to_delete = []
            for key, value in self.__other_portfolios.items():
                if value[1] == portfolio:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                del self.__other_portfolios[key]
        else:
            raise TypeError("The parameter 'portfolio' must be either a "
                            "Portfolio, a list of weights, or a string!")

    def get_portfolio_by_name(self, portfolio_name):
        portfolio = None
        if self.__other_portfolios is not None:
            for weights, (ptf_obj, name) in self.__other_portfolios.items():
                if name == portfolio_name:
                    portfolio = ptf_obj

        return portfolio

    def plot(self, compare_with=None, labels=None, weights_visible=True,
             zoom_individual=False, min_expected_return=None,
             max_expected_return=None, min_standard_deviation=None,
             max_standard_deviation=None, investors=None,
             indifference_curves=None, optimal_portfolios=True,
             legend='upper left'):
        investment_universes = [self]
        if isinstance(compare_with, InvestmentUniverse):
            investment_universes.append(compare_with)
        elif isinstance(compare_with, list):
            investment_universes.extend(compare_with)

        if self.__visualizer is None:
            self.__visualizer = InvestmentUniverseVisualizer(
                investment_universes, labels=labels,
                weights_visible=weights_visible)
        else:
            self.__visualizer.investment_universes = investment_universes
            self.__visualizer.labels = labels
            self.__visualizer.reset_colors()
        self.__visualizer.plot(zoom_individual=zoom_individual,
                               min_expected_return=min_expected_return,
                               max_expected_return=max_expected_return,
                               min_standard_deviation=min_standard_deviation,
                               max_standard_deviation=max_standard_deviation,
                               investors=investors,
                               indifference_curves=indifference_curves,
                               optimal_portfolios=optimal_portfolios,
                               legend=legend)

    def add_investor(self, gamma, wealth=None, portfolio=None,
                     utility_function=None, name=None):

        if name is None:
            self.__nb_unnamed_investors += 1
            name = "Investor {}".format(self.__nb_unnamed_investors)

        investor = Investor(self, gamma, wealth, portfolio, utility_function,
                            name)

        self.__investors[name] = investor

        return investor

    def calculate_market_portfolio(self):

        nb_weights = self.__nb_risky_assets + 1 if self.__r_f is not None \
            else self.__nb_risky_assets

        market_assets_value = np.zeros(nb_weights)
        self.__total_wealth = 0
        for investor in self.__investors.values():
            self.__total_wealth += investor.wealth
            weights = investor.portfolio.weights
            market_assets_value += np.array(weights) * investor.wealth

        market_weights = market_assets_value / self.__total_wealth

        if self.__r_f is not None:
            market_weights = \
                market_weights / sum(market_weights[:self.__nb_risky_assets])
            market_weights[-1] = 0.0

        self.__market_portfolio = Portfolio(market_weights, self.__mu,
                                            self.__cov, r_f=self.__r_f,
                                            assets=self.__assets)

        return self.__market_portfolio

    def calculate_optimal_portfolios(self, investors=None):

        if investors is None:
            investors = self.investors
        elif type(investors) is str:
            investors = [investors]

        optimal_portfolios = {}
        for investor_name in investors:
            investor = self.investors[investor_name]
            optimal_portfolio = investor.calculate_optimal_portfolio()
            optimal_portfolios[investor_name] \
                = {
                "portfolio": optimal_portfolio,
                "utility": investor.optimal_portfolio_utility
            }

        return optimal_portfolios

    ########################## PRIVATE ##########################

    def __assign_mu(self, mu):
        self.mu = mu
        if self.__r_f is not None:
            self.__augmented_mu = np.concatenate((self.__mu,
                                                  [self.__r_f]))
        else:
            self.__augmented_mu = self.mu

    def __assign_cov(self, cov):
        self.cov = cov
        if self.__r_f is not None:
            zeros_column = (np.zeros((len(self.__cov), 1)))
            zeros_row = (np.zeros((1, len(self.__cov) + 1)))
            self.__augmented_cov = \
                np.concatenate(
                    (np.concatenate((self.__cov, zeros_column), axis=1),
                     zeros_row))
        else:
            self.__augmented_cov = self.cov

    def __assign_r_f(self, r_f):
        if r_f is None or type(r_f) is float or type(r_f) is int:
            self.__r_f = r_f
        else:
            raise TypeError(
                "The parameter 'r_f' must be of type float or int.")

    def __assign_min_weight(self, min_weight):
        if min_weight is None:
            self.__min_weights = None
        elif type(min_weight) is float or type(min_weight) is int:
            self.__min_weights = min_weight * np.ones(self.__nb_risky_assets)
        elif type(min_weight) is list or type(min_weight) is np.array:
            self.__min_weights = min_weight
        else:
            raise TypeError(
                "The parameter 'min_weight' must be of type float ,int or "
                "list, or None.")

    def __assign_min_weight_r_f(self, min_weight_r_f):
        if min_weight_r_f is None and self.__min_weights is None:
            self.__min_weight_r_f = None
        elif min_weight_r_f is None:
            self.__min_weight_r_f = self.__parameters["min_weight_r_f"]
        else:
            self.__min_weight_r_f = min_weight_r_f

    def __assign_parameters(self, parameters):
        if parameters is None:
            self.__assign_default_parameters()
        else:
            self.__check_parameters(parameters)
            self.__parameters = parameters

    def __assign_default_parameters(self):
        self.__parameters = {
            "optimization_tolerance": 1e-8,
            "cal_min_fraction": 0,
            "cal_step_fraction": 0.001,
            "min_weight_r_f": -4,
            "cal_max_std": 100
        }

    def __check_parameters(self, parameters):
        if type(parameters) is not dict:
            raise TypeError(
                "The variable 'parameters' must be a dictionary.")

        if "optimization_tolerance" not in parameters:
            raise ValueError(
                "The dictionary parameters must contain the key "
                "'optimization_tolerance'.")
        if type(parameters["optimization_tolerance"]) is not float:
            raise TypeError(
                "The value of 'optimization_tolerance' must be of type float.")

    def __calculate_assets_std(self):
        std = []
        for i in range(0, len(self.__assets)):
            std.append(math.sqrt(self.__cov[i][i]))
        self.__std = np.array(std)
        return self.__std

    def __calculate_mvp(self, x0):

        # Sum portfolio weights equals 1 constraint
        self.__sum_weights_assets_equals_one_constraint = LinearConstraint(
            np.ones(self.__nb_risky_assets), 1, 1)

        self.__min_weights_bound = Bounds(self.__min_weights, np.inf) \
            if self.__min_weights is not None else None

        # Minimize
        mvp_result = minimize(
            lambda x: calculate_portfolio_standard_deviation(x, self.__cov),
            x0,
            bounds=self.__min_weights_bound,
            constraints=[self.__sum_weights_assets_equals_one_constraint],
            tol=self.__parameters["optimization_tolerance"])

        if self.__r_f is not None:
            mvp_weights = \
                np.append(mvp_result.x, [0])
        else:
            mvp_weights = mvp_result.x

        return Portfolio(mvp_weights, self.__mu, self.__cov, r_f=self.__r_f,
                         assets=self.__assets)

    def __calculate_efficient_mu_min_max(self, mvp):
        self.__efficient_mu_min = mvp.expected_return

        if self.__min_weights is not None:
            mu_argmax = np.argmax(self.__mu)
            mu_max = self.__mu[mu_argmax]
            others_mu = np.delete(self.__mu, mu_argmax)
            others_min_weight = np.delete(self.__min_weights,
                                          mu_argmax)
            self.__efficient_mu_max = (1 - sum(
                others_min_weight)) * mu_max + np.dot(
                others_mu, others_min_weight)
        else:
            LARGE_FACTOR = 2
            self.__efficient_mu_max = LARGE_FACTOR * max(self.mu)

    def __calculate_efficient_mu_array(self, nb_portfolios):
        # Define the range of expected return over which to calculate the
        # efficient frontier.
        delta_mu = self.__efficient_mu_max - self.__efficient_mu_min
        step = delta_mu / nb_portfolios
        efficient_mu_array = np.arange(self.__efficient_mu_min,
                                       self.__efficient_mu_max, step)

        return efficient_mu_array

    def __calculate_efficient_portfolio_from_mu(self, mu, x0, tolerance,
                                                maximize=False):

        factor = -1 if maximize else 1

        efficient_mu_constraint = LinearConstraint(self.__mu.T, mu, mu)
        efficient_portfolio_result = minimize(
            lambda x: factor * calculate_portfolio_standard_deviation(
                x, self.__cov),
            x0,
            bounds=self.__min_weights_bound,
            constraints=[self.__sum_weights_assets_equals_one_constraint,
                         efficient_mu_constraint],
            tol=tolerance)
        if not efficient_portfolio_result.success:
            raise ValueError(
                "minimize was not successful with bounds={} and constraints={}!"
                .format(self.__min_weights_bound, mu))

        if self.__r_f is not None:
            efficient_portfolio_weights = \
                np.append(efficient_portfolio_result.x, [0])
        else:
            efficient_portfolio_weights = efficient_portfolio_result.x

        efficient_portfolio = Portfolio(efficient_portfolio_weights, self.__mu,
                                        self.__cov, r_f=self.__r_f,
                                        assets=self.__assets)

        return efficient_portfolio

    def __calculate_efficient_portfolio_from_sigma(self, sigma, x0, tolerance):

        nb_iter = 0

        mu_min = self.__efficient_mu_min
        mu_max = self.__efficient_mu_max
        tentative_mu = (mu_min + mu_max) / 2
        tentative_portfolio = \
            self.__calculate_efficient_portfolio_from_mu(tentative_mu, x0,
                                                         tolerance)
        tentative_sigma = tentative_portfolio.standard_deviation
        while abs(tentative_sigma - sigma) >= tolerance:
            if sigma - tentative_sigma > 0:
                mu_min = tentative_mu
                tentative_mu = (tentative_mu + mu_max) / 2
            else:
                mu_max = tentative_mu
                tentative_mu = (mu_min + tentative_mu) / 2
            tentative_portfolio = \
                self.__calculate_efficient_portfolio_from_mu(tentative_mu, x0,
                                                             tolerance)
            tentative_sigma = tentative_portfolio.standard_deviation
            nb_iter += 1
            if nb_iter > self.MAX_NB_ITERATIONS:
                raise StopIteration("Number of iterations exceeded "
                                    "MAX_NB_ITERATIONS ({})"
                                    .format(self.MAX_NB_ITERATIONS))

        return tentative_portfolio

    def __get_cal_parameters(self):

        min_fraction = self.__parameters["cal_min_fraction"]
        step_fraction = self.__parameters["cal_step_fraction"]

        if self.__efficient_frontier is not None:
            max_std = max([ptf.standard_deviation
                           for ptf in self.__efficient_frontier])
        else:
            max_std = self.__parameters["cal_max_std"]

        if self.__tangency_portfolio is not None:
            max_fraction = min(
                1.0 - self.min_weight_r_f,
                max_std / self.__tangency_portfolio.standard_deviation)
        else:
            max_fraction = 1.0 - self.min_weight_r_f

        return min_fraction, max_fraction, step_fraction

    def __calculate_feasible_portfolio_equation(self, mu):

        one_vector = np.ones((len(self.mu), 1))

        s_1_1 = float(np.dot(one_vector.T,
                             np.dot(np.linalg.inv(self.cov), one_vector)))
        s_1_mu = float(np.dot(one_vector.T,
                              np.dot(np.linalg.inv(self.cov), self.mu)))
        s_mu_mu = float(
            np.dot(self.mu.T, np.dot(np.linalg.inv(self.cov), self.mu)))
        d = s_mu_mu * s_1_1 - s_1_mu ** 2

        sigma = math.sqrt(1 / d * (
                s_1_1 * mu ** 2 - 2 * s_1_mu * mu + s_mu_mu))

        return sigma

    def __calculate_cal_equation(self, mu, tangency_portfolio):

        tang_ptf_std = tangency_portfolio.standard_deviation
        tang_ptf_exp_ret = tangency_portfolio.expected_return
        slope = tang_ptf_std / (tang_ptf_exp_ret - self.r_f)

        sigma = slope * abs(mu - self.r_f)

        return sigma

    def __calculate_mvp_mu_sigma_from_equation(self):

        one_vector = np.ones((len(self.mu), 1))

        s_1_1 = float(np.dot(one_vector.T,
                             np.dot(np.linalg.inv(self.cov), one_vector)))
        s_1_mu = float(np.dot(one_vector.T,
                              np.dot(np.linalg.inv(self.cov), self.mu)))

        mvp_mu = s_1_mu / s_1_1
        mvp_sigma = self.__calculate_feasible_portfolio_equation(mvp_mu)

        return mvp_mu, mvp_sigma
