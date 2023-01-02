import numpy as np
import matplotlib.pyplot as plt
import math

from finlib.general.finance import Portfolio
from finlib.general.finance import calculate_portfolio_standard_deviation
from finlib.general.finance import calculate_portfolio_sharpe_ratio

from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds


################################################################################################
##################################### INVESTMENT UNIVERSE ######################################
################################################################################################

class InvestmentUniverse:

    def __init__(self, tickers_list, r_bar, sigma, r_f=None, min_weight=0, optimization_tolerance=1e-6):
        # The argument "nb_assets" denotes the number of assets in the portfolio

        self.__tickers_list = tickers_list
        self.__r_bar = np.array(r_bar)
        self.__sigma = np.array(sigma)
        self.__assign_r_f(r_f)
        self.__assign_min_weight(min_weight)
        self.__assign_optimization_tolerance(optimization_tolerance)

        self.__nb_assets = len(r_bar)
        self.__calculate_assets_std()

        self.__feasible_portfolios = None
        self.__mvp = None
        self.__efficient_frontier = None
        self.__tangency_portfolio = None
        self.__cal = self.__cal_r_bar_list = self.__cal_std_list = None
        self.__other_portfolios = None
        self.__min_weight_bound = None
        self.__sum_weights_assets_equals_one_constraint = None

    ################################# ATTRIBUTES #################################

    @property
    def tickers_list(self):
        return self.__tickers_list

    @tickers_list.setter
    def tickers_list(self, value):
        self.__tickers_list = value

    @property
    def r_bar(self):
        return self.__r_bar

    @r_bar.setter
    def r_bar(self, value):
        self.__r_bar = value

    @property
    def sigma(self):
        return self.__sigma

    @sigma.setter
    def sigma(self, value):
        self.__sigma = value

    @property
    def r_f(self):
        return self.__r_f

    @r_f.setter
    def r_f(self, value):
        self.__assign_r_f(value)

    @property
    def min_weight(self):
        return self.__min_weight

    @min_weight.setter
    def min_weight(self, value):
        self.__assign_min_weight(value)

    @property
    def optimization_tolerance(self):
        return self.__optimization_tolerance

    @optimization_tolerance.setter
    def optimization_tolerance(self, value):
        self.__assign_optimization_tolerance(value)

    @property
    def assets_std(self):
        return self.__assets_std

    @property
    def feasible_portfolios(self):
        return self.__feasible_portfolios

    @property
    def mvp(self):
        return self.__mvp

    @property
    def efficient_frontier(self):
        return self.__efficient_frontier

    @property
    def tangency_portfolio(self):
        return self.__tangency_portfolio

    @property
    def cal(self):
        return self.__cal

    @property
    def other_portfolios(self):
        return self.__other_portfolios

    ################################# PUBLIC FUNCTIONS #################################

    def add_feasible_portfolios(self, nb_portfolios=20000):
        self.__feasible_portfolios = []
        for i in range(1, nb_portfolios):
            portfolio_weights = self.__add_random_portfolio_weights(self.__min_weight)
            portfolio = Portfolio(portfolio_weights, self.__r_bar, self.__sigma)
            self.__feasible_portfolios.append(portfolio)

    def add_mvp(self, x0=None):

        # Initial guess (seed value)
        if x0 is None:
            x0 = np.ones(self.__nb_assets) / self.__nb_assets

        # Sum portfolio weights equals 1 constraint
        self.__sum_weights_assets_equals_one_constraint = LinearConstraint(np.ones(self.__nb_assets), 1, 1)
        self.__min_weight_bound = Bounds(self.__min_weight, np.inf)

        # Minimize
        mvp_result = minimize(lambda x: calculate_portfolio_standard_deviation(x, self.__sigma),
                              x0,
                              bounds=self.__min_weight_bound,
                              constraints=[self.__sum_weights_assets_equals_one_constraint],
                              tol=self.__optimization_tolerance)

        # Assign results
        self.__mvp = Portfolio(mvp_result.x, self.__r_bar, self.__sigma)

    def add_efficient_frontier(self, step=0.01, x0=None):

        # Initial guess (seed value)
        if x0 is None:
            x0 = np.ones(self.__nb_assets) / self.__nb_assets

        if not self.__mvp:
            self.add_mvp(x0)

        # Define the range of expected return over which to calculate the efficient frontier
        # Define the step between each calculation
        efficient_r_bar_min = self.__mvp.expected_return
        efficient_r_bar_max = (1 - self.__min_weight) * max(self.__r_bar) + self.__min_weight * min(self.__r_bar)
        efficient_r_bar_array = np.arange(efficient_r_bar_min,
                                          efficient_r_bar_max, step)

        # Calculate the efficient portfolios
        self.__efficient_frontier = []
        for efficient_r_bar in efficient_r_bar_array:
            efficient_r_bar_constraint = LinearConstraint(self.__r_bar.T,
                                                          efficient_r_bar,
                                                          efficient_r_bar)
            efficient_portfolio_result = minimize(lambda x: calculate_portfolio_standard_deviation(x, self.__sigma),
                                                  x0,
                                                  bounds=self.__min_weight_bound,
                                                  constraints=[self.__sum_weights_assets_equals_one_constraint,
                                                               efficient_r_bar_constraint],
                                                  tol=self.__optimization_tolerance)
            if not efficient_portfolio_result.success:
                raise ValueError("minimize was not successful with bounds={} and constraints={}!"
                                 .format(self.__min_weight_bound, efficient_r_bar))
            efficient_portfolio_weights = efficient_portfolio_result.x
            efficient_portfolio = Portfolio(efficient_portfolio_weights, self.__r_bar, self.__sigma)
            self.__efficient_frontier.append(efficient_portfolio)

    def add_tangency_portfolio(self, x0=None):

        if self.__r_f is None:
            raise ValueError("You need to add a risk-free asset first!")

        # Initial guess (seed value)
        if x0 is None:
            x0 = np.ones(self.__nb_assets) / self.__nb_assets

        tangency_portfolio_result = minimize(
            lambda x: -calculate_portfolio_sharpe_ratio(x, self.__r_bar, self.__sigma, self.__r_f),
            x0,
            bounds=self.__min_weight_bound,
            constraints=[self.__sum_weights_assets_equals_one_constraint],
            tol=self.__optimization_tolerance)

        tangency_portfolio_weights = tangency_portfolio_result.x
        self.__tangency_portfolio = Portfolio(tangency_portfolio_weights, self.__r_bar, self.__sigma)

    def add_cal(self, min_fraction_tangency=0, max_fraction_tangency=3,
                step_fraction_tangency=0.001):

        if not self.__tangency_portfolio:
            self.add_tangency_portfolio()

        self.__cal = []
        for tangency_weight in np.arange(min_fraction_tangency,
                                         max_fraction_tangency,
                                         step_fraction_tangency):
            cal_portfolio_weights = np.concatenate(
                (tangency_weight * self.__tangency_portfolio.weights, [1 - tangency_weight]))
            cal_portfolio = Portfolio(cal_portfolio_weights, self.__r_bar, self.__sigma, self.__r_f)
            self.__cal.append(cal_portfolio)

    def add_portfolio(self, portfolio, portfolio_name=None):
        if self.__other_portfolios is None:
            self.__other_portfolios = {}
        if portfolio_name is None:
            portfolio_name = str(len(self.__other_portfolios) + 1)
        self.__other_portfolios[portfolio_name] = portfolio

    def remove_other_portfolios(self):
        self.__other_portfolios = None

    ########################## PRIVATE ##########################

    def __assign_r_f(self, r_f):
        if type(r_f) is float or type(r_f) is int:
            self.__r_f = r_f
        else:
            raise TypeError("The variable 'r_f' must be of type float or int.")

    def __assign_min_weight(self, min_weight):
        if type(min_weight) is float or type(min_weight) is int:
            self.__min_weight = min_weight
        else:
            raise TypeError("The variable 'min_weight' must be of type float or int.")

    def __assign_optimization_tolerance(self, optimization_tolerance):
        if type(optimization_tolerance) is float:
            self.__optimization_tolerance = optimization_tolerance
        else:
            raise TypeError("The variable 'optimization_tolerance' must be of type float.")

    # Portfolio weights generator
    def __add_random_portfolio_weights(self, smallest_weight):
        # This function adds random portfolio weights
        # The argument "smallest_weight" denotest the smallest weight admissible for a given asset
        # For example, "smallest_weight=0" indicates that short sales are not allowed,
        # and "smallest_weight=-1" implies that the weight of each asset in the portfolio must be equal or greater to -1
        # The function returns an array of portfolio weights

        weights = np.random.dirichlet(np.ones(self.__nb_assets), size=1)[0]
        norm_weights = weights + smallest_weight - self.__nb_assets * weights * smallest_weight

        return norm_weights

    def __calculate_assets_std(self):
        self.__assets_std = []
        for i in range(0, len(self.__tickers_list)):
            self.__assets_std.append(math.sqrt(self.__sigma[i][i]))
        return self.__assets_std


class InvestmentUniverseVisualizer:

    def __init__(self, investment_universes, labels=None):

        if labels is None:
            labels = ["1", "2", "3", "4"]
        self.__assign_investment_universes(investment_universes)

        self.__labels = labels
        self.__alpha = 1.0 / len(self.__investment_universes)

        self.__ax = None
        self.__fig = None

        self.__min_std = 0
        self.__max_std = max([max(x.assets_std) for x in
                              self.__investment_universes]) + 0.5
        self.__min_r_bar = min(min([min(x.r_bar) for x in
                                    self.__investment_universes]),
                               min([x.r_f for x in
                                    self.__investment_universes])) - 0.5
        self.__max_r_bar = max(max([max(x.r_bar) for x in
                                    self.__investment_universes]),
                               max([x.r_f for x in
                                    self.__investment_universes])) + 0.5

        self.__feasible_portfolios_visible = False
        self.__mvp_visible = False
        self.__efficient_frontier_visible = False
        self.__tangency_portfolio_visible = False
        self.__cal_visible = False
        self.__r_f_visible = False
        self.__other_portfolios_visible = False

    ################################# ATTRIBUTES #################################

    @property
    def feasible_portfolios_visible(self):
        return self.__feasible_portfolios_visible

    @feasible_portfolios_visible.setter
    def feasible_portfolios_visible(self, value):
        self.__check_bool(value, "feasible_portfolios_visible")
        self.__feasible_portfolios_visible = value

    @property
    def mvp_visible(self):
        return self.__mvp_visible

    @mvp_visible.setter
    def mvp_visible(self, value):
        self.__check_bool(value, "mvp_visible")
        self.__mvp_visible = value

    @property
    def efficient_frontier_visible(self):
        return self.__efficient_frontier_visible

    @efficient_frontier_visible.setter
    def efficient_frontier_visible(self, value):
        self.__check_bool(value, "efficient_frontier_visible")
        self.__efficient_frontier_visible = value

    @property
    def tangency_portfolio_visible(self):
        return self.__tangency_portfolio_visible

    @tangency_portfolio_visible.setter
    def tangency_portfolio_visible(self, value):
        self.__check_bool(value, "tangency_portfolio_visible")
        self.__tangency_portfolio_visible = value

    @property
    def cal_visible(self):
        return self.__cal_visible

    @cal_visible.setter
    def cal_visible(self, value):
        self.__check_bool(value, "cal_visible")
        self.__cal_visible = value

    @property
    def r_f_visible(self):
        return self.__r_f_visible

    @r_f_visible.setter
    def r_f_visible(self, value):
        self.__check_bool(value, "r_f_visible")
        self.__r_f_visible = value

    @property
    def other_portfolios_visible(self):
        return self.__other_portfolios_visible

    @other_portfolios_visible.setter
    def other_portfolios_visible(self, value):
        self.__check_bool(value, "other_portfolios_visible")
        self.__other_portfolios_visible = value

    ##################### fig ###################
    @property
    def fig(self):
        return self.__fig

    ##################### ax ###################
    @property
    def ax(self):
        return self.__ax

    ################################# PUBLIC FUNCTIONS #################################

    def plot(self, feasible_portfolios_size=50):

        self.__fig, self.__ax = plt.subplots(figsize=(16, 10))

        self.__ax.set_xlim([self.__min_std, self.__max_std])
        self.__ax.set_ylim([self.__min_r_bar, self.__max_r_bar])
        self.__ax.grid()

        self.__ax.set_title("Risk-return tradeoff", fontsize=35)
        self.__ax.set_ylabel("Expected returns", fontsize=30)
        self.__ax.set_xlabel("Standard deviation", fontsize=30)
        self.__ax.tick_params(axis='both', labelsize=25)

        # all_tickers_list = list(set.union(*[set(x.tickers_list) for x in self.__investment_universes]))
        labels_iter = iter(self.__labels)
        for investment_universe in self.__investment_universes:
            label = next(labels_iter)
            self.__plot_investment_universe(investment_universe, label, feasible_portfolios_size)
        self.__plot_tickers(self.__investment_universes[0])

        self.__ax.legend(fontsize=15)

    ########################## PRIVATE ##########################

    def __assign_investment_universes(self, investment_universes):
        if isinstance(investment_universes, InvestmentUniverse):
            self.__investment_universes = [investment_universes]
        else:
            self.__investment_universes = investment_universes

    def __plot_investment_universe(self, investment_universe, label, feasible_portfolios_size):

        if investment_universe.feasible_portfolios and self.__feasible_portfolios_visible:
            self.__plot_feasible_portfolios(investment_universe, label, feasible_portfolios_size)

        if investment_universe.efficient_frontier and self.__efficient_frontier_visible:
            self.__plot_efficient_portfolios(investment_universe, label)

        if investment_universe.cal and self.__cal_visible:
            self.__plot_cal(investment_universe, label)

        if investment_universe.mvp and self.__mvp_visible:
            self.__plot_mvp(investment_universe, label)

        if investment_universe.r_f and self.__r_f_visible:
            self.__plot_r_f(investment_universe, label)

        if investment_universe.tangency_portfolio and self.__tangency_portfolio_visible:
            self.__plot_tangency_portfolio(investment_universe, label)

        if investment_universe.other_portfolios and self.__other_portfolios_visible:
            self.__plot_other_portfolios(investment_universe, label)

    def __plot_feasible_portfolios(self, investment_universe, label, feasible_portfolios_size):
        feasible_portfolios_r_bar_list = list(map(lambda x: x.expected_return,
                                                  investment_universe.feasible_portfolios))
        feasible_portfolios_std_list = list(map(lambda x: x.standard_deviation,
                                                investment_universe.feasible_portfolios))
        self.__ax.scatter(feasible_portfolios_std_list, feasible_portfolios_r_bar_list, s=feasible_portfolios_size,
                          alpha=self.__alpha,
                          label="Feasible portfolios - " + label)

    def __plot_efficient_portfolios(self, investment_universe, label):
        efficient_portfolios_r_bar_list = list(map(lambda x: x.expected_return,
                                                   investment_universe.efficient_frontier))
        efficient_portfolios_std_list = list(map(lambda x: x.standard_deviation,
                                                 investment_universe.efficient_frontier))
        self.__ax.scatter(efficient_portfolios_std_list, efficient_portfolios_r_bar_list, color="black", s=50,
                          label="Efficient frontier")

    def __plot_cal(self, investment_universe, label):
        cal_portfolios_r_bar_list = list(map(lambda x: x.expected_return,
                                             investment_universe.cal))
        cal_portfolios_std_list = list(map(lambda x: x.standard_deviation,
                                           investment_universe.cal))
        self.__ax.scatter(cal_portfolios_std_list, cal_portfolios_r_bar_list, s=20, label="CAL - " + label)

    def __plot_tickers(self, investment_universe):
        for i in range(0, len(investment_universe.tickers_list)):
            self.__ax.scatter(investment_universe.assets_std[i], investment_universe.r_bar[i], s=100,
                              label=investment_universe.tickers_list[i])

    def __plot_mvp(self, investment_universe, label):
        self.__ax.scatter(investment_universe.mvp.standard_deviation, investment_universe.mvp.expected_return, s=100,
                          label="MVP - " + label)

    def __plot_r_f(self, investment_universe, label):
        self.__ax.scatter(0, investment_universe.r_f, s=100, label="Risk-free rate - " + label)

    def __plot_tangency_portfolio(self, investment_universe, label):
        self.__ax.scatter(investment_universe.tangency_portfolio.standard_deviation,
                          investment_universe.tangency_portfolio.expected_return, s=100,
                          label="Tangency portfolio - " + label)

    def __plot_other_portfolios(self, investment_universe, label):
        for portfolio_name, portfolio in investment_universe.other_portfolios.items():
            self.__ax.scatter(portfolio.standard_deviation, portfolio.expected_return, s=100,
                              label=portfolio_name + " - " + label)

    def __check_bool(self, value, variable_name):
        if type(value) is not bool:
            raise TypeError("'{}' must be a boolean!".format(variable_name))
