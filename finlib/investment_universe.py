import numpy as np
import matplotlib.pyplot as plt
import math

from finlib.general import Portfolio
from finlib.general import calculate_portfolio_standard_deviation
from finlib.general import calculate_portfolio_sharpe_ratio

from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds


################################################################################################
##################################### INVESTMENT UNIVERSE ######################################
################################################################################################

class InvestmentUniverse:

    def __init__(self, tickers_list, mu, cov, r_f=None, min_weight=0,
                 optimization_tolerance=1e-6):
        # The argument "nb_assets" denotes the number of assets in the portfolio

        self.__tickers_list = tickers_list
        self.__mu = np.array(mu)
        self.__cov = np.array(cov)
        self.__assign_r_f(r_f)
        self.__assign_min_weight(min_weight)
        self.__assign_optimization_tolerance(optimization_tolerance)

        self.__nb_assets = len(mu)
        self.__calculate_assets_std()

        self.__feasible_portfolios = None
        self.__mvp = None
        self.__efficient_frontier = None
        self.__tangency_portfolio = None
        self.__cal = self.__cal_mu_list = self.__cal_std_list = None
        self.__other_portfolios = None

        self.__min_weight_bound = None
        self.__sum_weights_assets_equals_one_constraint = None

        self.__visualizer = None

    ################################# ATTRIBUTES #################################

    @property
    def tickers_list(self):
        return self.__tickers_list

    @tickers_list.setter
    def tickers_list(self, value):
        self.__tickers_list = value

    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, value):
        self.__mu = value

    @property
    def cov(self):
        return self.__cov

    @cov.setter
    def cov(self, value):
        self.__cov = value

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

    @property
    def visualizer(self):
        return self.__visualizer

    ################################# PUBLIC FUNCTIONS #################################

    def add_feasible_portfolios(self, nb_portfolios=20000):
        self.__feasible_portfolios = []
        for i in range(1, nb_portfolios):
            portfolio_weights = self.__add_random_portfolio_weights(
                self.__min_weight)
            portfolio = Portfolio(portfolio_weights, self.__mu, self.__cov)
            self.__feasible_portfolios.append(portfolio)

    def add_mvp(self, x0=None):

        # Initial guess (seed value)
        if x0 is None:
            x0 = np.ones(self.__nb_assets) / self.__nb_assets

        # Sum portfolio weights equals 1 constraint
        self.__sum_weights_assets_equals_one_constraint = LinearConstraint(
            np.ones(self.__nb_assets), 1, 1)
        self.__min_weight_bound = Bounds(self.__min_weight, np.inf)

        # Minimize
        mvp_result = minimize(
            lambda x: calculate_portfolio_standard_deviation(x, self.__cov),
            x0,
            bounds=self.__min_weight_bound,
            constraints=[self.__sum_weights_assets_equals_one_constraint],
            tol=self.__optimization_tolerance)

        # Assign results
        self.__mvp = Portfolio(mvp_result.x, self.__mu, self.__cov)

    def add_efficient_frontier(self, step=0.01, x0=None):

        # Initial guess (seed value)
        if x0 is None:
            x0 = np.ones(self.__nb_assets) / self.__nb_assets

        if not self.__mvp:
            self.add_mvp(x0)

        # Define the range of expected return over which to calculate the efficient frontier
        # Define the step between each calculation
        efficient_mu_min = self.__mvp.expected_return
        efficient_mu_max = (1 - self.__min_weight) * max(
            self.__mu) + self.__min_weight * min(self.__mu)
        efficient_mu_array = np.arange(efficient_mu_min,
                                       efficient_mu_max, step)

        # Calculate the efficient portfolios
        self.__efficient_frontier = []
        for efficient_mu in efficient_mu_array:
            efficient_mu_constraint = LinearConstraint(self.__mu.T,
                                                       efficient_mu,
                                                       efficient_mu)
            efficient_portfolio_result = minimize(
                lambda x: calculate_portfolio_standard_deviation(x,
                                                                 self.__cov),
                x0,
                bounds=self.__min_weight_bound,
                constraints=[self.__sum_weights_assets_equals_one_constraint,
                             efficient_mu_constraint],
                tol=self.__optimization_tolerance)
            if not efficient_portfolio_result.success:
                raise ValueError(
                    "minimize was not successful with bounds={} and constraints={}!"
                    .format(self.__min_weight_bound, efficient_mu))
            efficient_portfolio_weights = efficient_portfolio_result.x
            efficient_portfolio = Portfolio(efficient_portfolio_weights,
                                            self.__mu, self.__cov)
            self.__efficient_frontier.append(efficient_portfolio)

    def add_tangency_portfolio(self, x0=None):

        if self.__r_f is None:
            raise ValueError("You need to add a risk-free asset first!")

        # Initial guess (seed value)
        if x0 is None:
            x0 = np.ones(self.__nb_assets) / self.__nb_assets

        tangency_portfolio_result = minimize(
            lambda x: -calculate_portfolio_sharpe_ratio(x, self.__mu,
                                                        self.__cov,
                                                        self.__r_f),
            x0,
            bounds=self.__min_weight_bound,
            constraints=[self.__sum_weights_assets_equals_one_constraint],
            tol=self.__optimization_tolerance)

        tangency_portfolio_weights = tangency_portfolio_result.x
        self.__tangency_portfolio = Portfolio(tangency_portfolio_weights,
                                              self.__mu, self.__cov)

    def add_cal(self, min_fraction_tangency=0, max_fraction_tangency=3,
                step_fraction_tangency=0.001):

        if not self.__tangency_portfolio:
            self.add_tangency_portfolio()

        self.__cal = []
        for tangency_weight in np.arange(min_fraction_tangency,
                                         max_fraction_tangency,
                                         step_fraction_tangency):
            cal_portfolio_weights = np.concatenate(
                (tangency_weight * self.__tangency_portfolio.weights,
                 [1 - tangency_weight]))
            cal_portfolio = Portfolio(cal_portfolio_weights, self.__mu,
                                      self.__cov, self.__r_f)
            self.__cal.append(cal_portfolio)

    def add_portfolio(self, portfolio, portfolio_name=None):

        if type(portfolio) is Portfolio:
            portfolio_obj = portfolio
        elif type(portfolio) is list and len(portfolio) == self.__nb_assets:
            portfolio_obj = Portfolio(portfolio, self.__mu, self.__cov)
        else:
            raise TypeError("The variable 'portfolio' must be an object of "
                            "type Portfolio or a list of weights.")

        if self.__other_portfolios is None:
            self.__other_portfolios = {}
        if portfolio_name is None:
            portfolio_name = str(len(self.__other_portfolios) + 1)
        self.__other_portfolios[portfolio_name] = portfolio_obj

    def remove_other_portfolios(self, portfolio_name=None):
        if portfolio_name is None:
            self.__other_portfolios = None
        else:
            del self.__other_portfolios[portfolio_name]

    def calculate_portfolio(self, weights):
        return Portfolio(weights, self.__mu, self.__cov)

    def plot(self):
        if self.__visualizer is None:
            self.__visualizer = InvestmentUniverseVisualizer([self])
        self.__visualizer.plot()

    ########################## PRIVATE ##########################

    def __assign_r_f(self, r_f):
        if r_f is None or type(r_f) is float or type(r_f) is int:
            self.__r_f = r_f
        else:
            raise TypeError("The variable 'r_f' must be of type float or int.")

    def __assign_min_weight(self, min_weight):
        if type(min_weight) is float or type(min_weight) is int:
            self.__min_weight = min_weight
        else:
            raise TypeError(
                "The variable 'min_weight' must be of type float or int.")

    def __assign_optimization_tolerance(self, optimization_tolerance):
        if type(optimization_tolerance) is float:
            self.__optimization_tolerance = optimization_tolerance
        else:
            raise TypeError(
                "The variable 'optimization_tolerance' must be of type float.")

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
            self.__assets_std.append(math.sqrt(self.__cov[i][i]))
        return self.__assets_std


class InvestmentUniverseVisualizer:

    def __init__(self, investment_universes, labels=None,
                 default_visibility=True):

        if labels is None and len(investment_universes) > 1:
            labels = ["1", "2", "3", "4"]
        elif labels is None:
            labels = []

        self.__assign_investment_universes(investment_universes)

        self.__labels = labels
        self.__alpha = 1.0 / len(self.__investment_universes)

        self.__ax = None
        self.__fig = None

        self.__set_default_visibility(default_visibility)

        self.__calculate_visible_portfolios_mu_std()
        self.__set_default_std_limits()
        self.__set_default_mu_limits()

    ################################# ATTRIBUTES #################################

    @property
    def assets_visible(self):
        return self.__cal_visible

    @assets_visible.setter
    def assets_visible(self, value):
        self.__check_bool(value, "assets_visible")
        self.__assets_visible = value

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

    @property
    def min_mu(self):
        return self.__min_mu

    @min_mu.setter
    def min_mu(self, value):
        self.__min_mu = value

    @property
    def max_mu(self):
        return self.__max_mu

    @max_mu.setter
    def max_mu(self, value):
        self.__max_mu = value

    @property
    def min_std(self):
        return self.__min_std

    @min_std.setter
    def min_std(self, value):
        self.__min_std = value

    @property
    def max_std(self):
        return self.__max_std

    @max_mu.setter
    def max_std(self, value):
        self.__max_std = value

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
        self.__ax.set_ylim([self.__min_mu, self.__max_mu])
        self.__ax.grid()

        self.__ax.set_title("Risk-return tradeoff", fontsize=35)
        self.__ax.set_ylabel("Expected returns", fontsize=30)
        self.__ax.set_xlabel("Standard deviation", fontsize=30)
        self.__ax.tick_params(axis='both', labelsize=25)

        labels_iter = iter(self.__labels)
        for investment_universe in self.__investment_universes:
            label = next(labels_iter, None)
            self.__plot_investment_universe(investment_universe, label,
                                            feasible_portfolios_size)

        self.__ax.legend(fontsize=15)

    ########################## PRIVATE ##########################

    def __set_default_visibility(self, default_visibility):
        self.__assets_visible = default_visibility
        self.__feasible_portfolios_visible = default_visibility
        self.__mvp_visible = default_visibility
        self.__efficient_frontier_visible = default_visibility
        self.__tangency_portfolio_visible = default_visibility
        self.__cal_visible = default_visibility
        self.__r_f_visible = default_visibility
        self.__other_portfolios_visible = default_visibility

    def __set_default_mu_limits(self, default_border=0.1):

        mu_list = [mu for mu, std in self.__visible_portfolios_mu_std_list]
        border_abs_value = default_border * max(mu_list)

        self.__min_mu = min(0, min(mu_list) - border_abs_value)
        self.__max_mu = max(mu_list) + border_abs_value

    def __set_default_std_limits(self, default_border=0.1):

        std_list = [std for mu, std in self.__visible_portfolios_mu_std_list]
        border_abs_value = default_border * max(std_list)

        self.__min_std = min(0, min(std_list) - border_abs_value)
        self.__max_std = max(std_list) + border_abs_value

    def __calculate_visible_portfolios_mu_std(self):
        self.__visible_portfolios_mu_std_list = []

        remaining_ptfs_list = []
        for inv_uni in self.__investment_universes:
            self.__visible_portfolios_mu_std_list.extend(
                list(zip(inv_uni.mu, inv_uni.assets_std)))
            if inv_uni.mvp is not None:
                remaining_ptfs_list.append(inv_uni.mvp)
            if inv_uni.tangency_portfolio is not None:
                remaining_ptfs_list.append(inv_uni.tangency_portfolio)
            if inv_uni.other_portfolios is not None:
                other_portfolios = list(inv_uni.other_portfolios.values())
                remaining_ptfs_list.extend(other_portfolios)

        self.__visible_portfolios_mu_std_list.extend(
            [(ptf.expected_return, ptf.standard_deviation) for ptf
             in remaining_ptfs_list])

    def __assign_investment_universes(self, investment_universes):
        if isinstance(investment_universes, InvestmentUniverse):
            self.__investment_universes = [investment_universes]
        else:
            self.__investment_universes = investment_universes

    def __plot_investment_universe(self, investment_universe, label,
                                   feasible_portfolios_size):

        if investment_universe.feasible_portfolios and self.__feasible_portfolios_visible:
            self.__plot_feasible_portfolios(investment_universe, label,
                                            feasible_portfolios_size)

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

        if self.__assets_visible:
            self.__plot_tickers(investment_universe)

    def __plot_feasible_portfolios(self, investment_universe, label,
                                   feasible_portfolios_size):
        feasible_portfolios_mu_list = list(map(lambda x: x.expected_return,
                                               investment_universe.feasible_portfolios))
        feasible_portfolios_std_list = list(map(lambda x: x.standard_deviation,
                                                investment_universe.feasible_portfolios))

        legend_label = self.__complete_label("Feasible portfolios", label)
        self.__ax.scatter(feasible_portfolios_std_list,
                          feasible_portfolios_mu_list,
                          s=feasible_portfolios_size,
                          alpha=self.__alpha,
                          label=legend_label)

    def __plot_efficient_portfolios(self, investment_universe, label):
        efficient_portfolios_mu_list = list(map(lambda x: x.expected_return,
                                                investment_universe.efficient_frontier))
        efficient_portfolios_std_list = list(
            map(lambda x: x.standard_deviation,
                investment_universe.efficient_frontier))
        legend_label = self.__complete_label("Efficient frontier", label)
        self.__ax.scatter(efficient_portfolios_std_list,
                          efficient_portfolios_mu_list, color="black", s=50,
                          label=legend_label)

    def __plot_cal(self, investment_universe, label):
        cal_portfolios_mu_list = list(map(lambda x: x.expected_return,
                                          investment_universe.cal))
        cal_portfolios_std_list = list(map(lambda x: x.standard_deviation,
                                           investment_universe.cal))
        legend_label = self.__complete_label("CAL", label)
        self.__ax.scatter(cal_portfolios_std_list, cal_portfolios_mu_list,
                          s=20, label=legend_label)

    def __plot_tickers(self, investment_universe):
        for i in range(0, len(investment_universe.tickers_list)):
            self.__ax.scatter(investment_universe.assets_std[i],
                              investment_universe.mu[i], s=100,
                              label=investment_universe.tickers_list[i])

    def __plot_mvp(self, investment_universe, label):
        legend_label = self.__complete_label("MVP", label)
        self.__ax.scatter(investment_universe.mvp.standard_deviation,
                          investment_universe.mvp.expected_return, s=100,
                          label=legend_label)

    def __plot_r_f(self, investment_universe, label):
        legend_label = self.__complete_label("Risk-free rate", label)
        self.__ax.scatter(0, investment_universe.r_f, s=100,
                          label=legend_label)

    def __plot_tangency_portfolio(self, investment_universe, label):
        legend_label = self.__complete_label("Tangency portfolio", label)
        self.__ax.scatter(
            investment_universe.tangency_portfolio.standard_deviation,
            investment_universe.tangency_portfolio.expected_return, s=100,
            label=legend_label)

    def __plot_other_portfolios(self, investment_universe, label):

        for portfolio_name, portfolio in investment_universe.other_portfolios.items():
            legend_label = self.__complete_label(portfolio_name, label)
            self.__ax.scatter(portfolio.standard_deviation,
                              portfolio.expected_return, s=100,
                              label=legend_label)

    def __check_bool(self, value, variable_name):
        if type(value) is not bool:
            raise TypeError("'{}' must be a boolean!".format(variable_name))

    def __complete_label(self, initial_legend_label, additional_label):
        completed_legend_label = initial_legend_label
        if additional_label is not None:
            completed_legend_label += " - " + additional_label

        return completed_legend_label
