import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math

from finlib.general import Portfolio, calculate_portfolio_expected_return
from finlib.general import calculate_portfolio_standard_deviation
from finlib.general import calculate_portfolio_sharpe_ratio

from scipy.optimize import minimize, NonlinearConstraint
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds


################################################################################################
##################################### INVESTMENT UNIVERSE ######################################
################################################################################################

class InvestmentUniverse:
    MAX_NB_ITERATIONS = 100

    def __init__(self, tickers_list, mu, cov, r_f=None, min_weight=0,
                 optimization_tolerance=1e-8):
        # The argument "nb_assets" denotes the number of assets in the
        # portfolio

        self.__tickers_list = tickers_list
        self.__nb_assets = len(mu)
        self.mu = mu
        self.cov = cov
        self.__assign_r_f(r_f)
        self.__assign_min_weight(min_weight)
        self.__assign_optimization_tolerance(optimization_tolerance)

        self.__calculate_assets_std()

        self.__feasible_portfolios = None
        self.__mvp = None
        self.__efficient_frontier = None
        self.__tangency_portfolio = None
        self.__cal = self.__cal_mu_list = self.__cal_std_list = None
        self.__other_portfolios = None

        self.__min_weight_bound = None
        self.__sum_weights_assets_equals_one_constraint = None

        self.__efficient_mu_min = None
        self.__efficient_mu_max = None

        self.__visualizer = None

    ################################# ATTRIBUTES ##############################

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
        self.__mu = np.array(value)

    @property
    def cov(self):
        return self.__cov

    @cov.setter
    def cov(self, value):
        self.__cov = np.array(value)

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
    def std(self):
        return self.__std

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

    def calculate_feasible_portfolios(self, nb_portfolios=20000):
        self.__feasible_portfolios = []
        for i in range(1, nb_portfolios):
            portfolio_weights = self.__add_random_portfolio_weights(
                self.__min_weight)
            portfolio = Portfolio(portfolio_weights, self.__mu, self.__cov)
            self.__feasible_portfolios.append(portfolio)

        return self.__feasible_portfolios

    def calculate_mvp(self, x0=None):

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

        return self.__mvp

    def calculate_efficient_portfolio(self, mu=None, sigma=None, name=None,
                                      x0=None, tolerance=None):
        tolerance = self.__optimization_tolerance if tolerance is None \
            else tolerance

        # Initial guess (seed value)
        if x0 is None:
            x0 = np.ones(self.__nb_assets) / self.__nb_assets

        self.__calculate_efficient_mu_min_max()

        if mu is not None and sigma is not None:
            raise ValueError("Only one of 'mu' and 'sigma' must be passed as "
                             "argument.")
        elif mu is not None:
            efficient_portfolio = \
                self.__calculate_efficient_portfolio_from_mu(mu, x0, tolerance)
        elif sigma is not None:
            efficient_portfolio = \
                self.__calculate_efficient_portfolio_from_sigma(sigma, x0,
                                                                tolerance)
        else:
            raise ValueError("Either 'mu' or 'sigma' must be passed as "
                             "argument.")

        self.calculate_portfolio(efficient_portfolio, name)

        return efficient_portfolio

    def calculate_efficient_frontier(self, nb_portfolios=1000, x0=None,
                                     tolerance=None):

        tolerance = self.__optimization_tolerance if tolerance is None \
            else tolerance

        # Initial guess (seed value)
        if x0 is None:
            x0 = np.ones(self.__nb_assets) / self.__nb_assets

        if not self.__mvp:
            self.calculate_mvp(x0)

        self.__calculate_efficient_mu_min_max()
        efficient_mu_array = self.__calculate_efficient_mu_array(nb_portfolios)

        # Calculate the efficient portfolios
        self.__efficient_frontier = []
        for efficient_mu in efficient_mu_array:
            efficient_portfolio = \
                self.__calculate_efficient_portfolio_from_mu(efficient_mu, x0,
                                                             tolerance)
            self.__efficient_frontier.append(efficient_portfolio)

        return self.__efficient_frontier

    def calculate_tangency_portfolio(self, x0=None):

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

        return self.__tangency_portfolio

    def calculate_cal(self, min_fraction_tangency=0, max_fraction_tangency=3,
                      step_fraction_tangency=0.001):

        if not self.__tangency_portfolio:
            self.calculate_tangency_portfolio()

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

        return self.__cal

    def calculate_portfolio(self, portfolio, name=None):

        if isinstance(portfolio, Portfolio):
            portfolio_obj = portfolio
        elif isinstance(portfolio, list) and len(
                portfolio) == self.__nb_assets:
            portfolio_obj = Portfolio(portfolio, self.__mu, self.__cov)
        else:
            raise TypeError("The variable 'portfolio' must be an object of "
                            "type Portfolio or a list of weights.")

        if self.__other_portfolios is None:
            self.__other_portfolios = {}
        if name is None:
            name = str(len(self.__other_portfolios) + 1)
        self.__other_portfolios[name] = portfolio_obj

        return portfolio_obj

    def remove_other_portfolios(self, name=None):
        if name is None:
            self.__other_portfolios = None
        else:
            del self.__other_portfolios[name]

    def plot(self, compare_with=None, labels=None, weights_visible=True):
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
        self.__visualizer.plot()

    ########################## PRIVATE ##########################

    def __assign_r_f(self, r_f):
        if r_f is None or type(r_f) is float or type(r_f) is int:
            self.__r_f = r_f
        else:
            raise TypeError("The variable 'r_f' must be of type float or int.")

    def __assign_min_weight(self, min_weight):
        if type(min_weight) is float or type(min_weight) is int:
            self.__min_weight = min_weight * np.ones(self.__nb_assets)
        elif type(min_weight) is list or type(min_weight) is np.array:
            self.__min_weight = min_weight
        else:
            raise TypeError(
                "The variable 'min_weight' must be of type float ,int or "
                "list.")

    def __assign_optimization_tolerance(self, optimization_tolerance):
        if type(optimization_tolerance) is float:
            self.__optimization_tolerance = optimization_tolerance
        else:
            raise TypeError(
                "The variable 'optimization_tolerance' must be of type float.")

    # Portfolio weights generator
    def __add_random_portfolio_weights(self, smallest_weights_list):
        # This function adds random portfolio weights
        # The argument "smallest_weight" denotest the smallest weight admissible for a given asset
        # For example, "smallest_weight=0" indicates that short sales are not allowed,
        # and "smallest_weight=-1" implies that the weight of each asset in the portfolio must be equal or greater to -1
        # The function returns an array of portfolio weights

        weights = np.random.dirichlet(np.ones(self.__nb_assets), size=1)[0]
        norm_weights = \
            weights * (1 - sum(smallest_weights_list)) + smallest_weights_list

        return norm_weights

    def __calculate_assets_std(self):
        std = []
        for i in range(0, len(self.__tickers_list)):
            std.append(math.sqrt(self.__cov[i][i]))
        self.__std = np.array(std)
        return self.__std

    def __calculate_efficient_mu_min_max(self):
        mu_argmax = np.argmax(self.__mu)
        mu_max = self.__mu[mu_argmax]
        others_mu = np.delete(self.__mu, mu_argmax)
        others_min_weight = np.delete(self.__min_weight, mu_argmax)

        self.__efficient_mu_min = self.__mvp.expected_return
        self.__efficient_mu_max = (1 - sum(
            others_min_weight)) * mu_max + np.dot(
            others_mu, others_min_weight)

    def __calculate_efficient_mu_array(self, nb_portfolios):
        # Define the range of expected return over which to calculate the
        # efficient frontier.
        delta_mu = self.__efficient_mu_max - self.__efficient_mu_min
        step = delta_mu / nb_portfolios
        efficient_mu_array = np.arange(self.__efficient_mu_min,
                                       self.__efficient_mu_max, step)

        return efficient_mu_array

    def __calculate_efficient_portfolio_from_mu(self, mu, x0, tolerance):
        efficient_mu_constraint = LinearConstraint(self.__mu.T, mu, mu)
        efficient_portfolio_result = minimize(
            lambda x: calculate_portfolio_standard_deviation(x,
                                                             self.__cov),
            x0,
            bounds=self.__min_weight_bound,
            constraints=[self.__sum_weights_assets_equals_one_constraint,
                         efficient_mu_constraint],
            tol=tolerance)
        if not efficient_portfolio_result.success:
            raise ValueError(
                "minimize was not successful with bounds={} and constraints={}!"
                .format(self.__min_weight_bound, mu))
        efficient_portfolio_weights = efficient_portfolio_result.x
        efficient_portfolio = Portfolio(efficient_portfolio_weights,
                                        self.__mu, self.__cov)

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


class InvestmentUniverseVisualizer:
    MAX_NB_INV_UNIV = 4

    class VisualElement:
        def __init__(self, plot_function, priority, size,
                     investment_universe=None, label=None):
            self.__plot_function = plot_function
            self.__investment_universe = investment_universe
            self.__size = size
            self.__priority = priority
            self.__label = label

        def plot(self):
            if self.__investment_universe is not None:
                self.__plot_function(self.__investment_universe, self.__label,
                                     self.__size)
            else:
                self.__plot_function(self.__label, self.__size)

        def __lt__(self, other):
            return self.__priority < other.__priority

    def __init__(self, investment_universes, labels=None,
                 default_visibility=True, weights_visible=True):

        self.__assign_investment_universes(investment_universes)
        self.__assign_labels(labels)
        self.__generate_tickers_inv_univ_dict()

        self.__alpha = 1.0

        self.__ax = None
        self.__fig = None

        self.__set_default_visibility(default_visibility)
        self.__weights_visible = weights_visible

        self.__calculate_visible_portfolios_mu_std()
        self.__set_default_std_limits()
        self.__set_default_mu_limits()

        self.__set_default_colors()
        self.__set_default_visual_elements_properties()

        self.__visual_elements_list = None

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
    def weights_visible(self):
        return self.__weights_visible

    @weights_visible.setter
    def weights_visible(self, value):
        self.__weights_visible = value

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

    @max_std.setter
    def max_std(self, value):
        self.__max_std = value

    @property
    def investment_universes(self):
        return self.__investment_universes

    @investment_universes.setter
    def investment_universes(self, value):
        self.__assign_investment_universes(value)
        self.__assign_labels(None)

    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, value):
        self.__assign_labels(value)

    @property
    def visibility_priorities(self):
        return self.__visibility_priorities

    @visibility_priorities.setter
    def visibility_priorities(self, value):
        self.__visibility_priorities = value

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

        self.__generate_visual_elements_list()

        sorted_visual_elements = sorted(self.__visual_elements_list,
                                        reverse=True)
        for vis_elem in sorted_visual_elements:
            vis_elem.plot()

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
                list(zip(inv_uni.mu, inv_uni.std)))
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

    def __assign_labels(self, labels):
        if labels is None and len(self.__investment_universes) > 1:
            self.__labels = ["1", "2", "3", "4"]
        elif labels is None:
            self.__labels = []
        else:
            self.__labels = labels

    def __generate_tickers_inv_univ_dict(self):
        self.__tickers_inv_univ_dict = {}
        inv_univ_index = 0
        for investment_universe in self.__investment_universes:
            ticker_index = 0
            for ticker in investment_universe.tickers_list:
                if ticker not in self.__tickers_inv_univ_dict:
                    self.__tickers_inv_univ_dict[ticker] = (inv_univ_index,
                                                            ticker_index)
                ticker_index += 1
            inv_univ_index += 1

    def __plot_feasible_portfolios(self, investment_universe, label, size):
        # TODO: Make it more general
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["feasible"]

        feasible_portfolios_mu_list = list(map(lambda x: x.expected_return,
                                               investment_universe.feasible_portfolios))
        feasible_portfolios_std_list = list(map(lambda x: x.standard_deviation,
                                                investment_universe.feasible_portfolios))

        legend_label = self.__complete_label("Feasible portfolios", label)
        self.__ax.scatter(feasible_portfolios_std_list,
                          feasible_portfolios_mu_list,
                          s=size,
                          alpha=self.__alpha,
                          label=legend_label,
                          color=color)

    def __plot_efficient_portfolios(self, investment_universe, label, size=50):
        # TODO: Make it more general
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["efficient"]

        efficient_portfolios_mu_list = list(map(lambda x: x.expected_return,
                                                investment_universe.efficient_frontier))
        efficient_portfolios_std_list = list(
            map(lambda x: x.standard_deviation,
                investment_universe.efficient_frontier))
        legend_label = self.__complete_label("Efficient frontier", label)
        self.__ax.scatter(efficient_portfolios_std_list,
                          efficient_portfolios_mu_list, color=color, s=size)

    def __plot_cal(self, investment_universe, label, size=20):
        # TODO: Make it more general
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["cal"]

        cal_portfolios_mu_list = list(map(lambda x: x.expected_return,
                                          investment_universe.cal))
        cal_portfolios_std_list = list(map(lambda x: x.standard_deviation,
                                           investment_universe.cal))
        legend_label = self.__complete_label("CAL", label)
        self.__ax.scatter(cal_portfolios_std_list, cal_portfolios_mu_list,
                          s=size, label=legend_label, color=color)

    def __plot_tickers(self, label, size=200):
        # TODO: Make it more general
        color_label = label if label is not None else "1"
        color_iter = iter(self.__colors[color_label]["assets"])

        for ticker, (inv_univ_index, ticker_index) \
                in self.__tickers_inv_univ_dict.items():
            inv_univ = self.__investment_universes[inv_univ_index]
            color = next(color_iter)
            self.__ax.scatter(inv_univ.std[ticker_index],
                              inv_univ.mu[ticker_index], s=size,
                              label=inv_univ.tickers_list[ticker_index],
                              color=color)

    def __plot_mvp(self, investment_universe, label, size=200):
        # TODO: Make it more general
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["mvp"]

        legend_label = self.__complete_label("MVP", label)
        self.__ax.scatter(investment_universe.mvp.standard_deviation,
                          investment_universe.mvp.expected_return, s=size,
                          label=legend_label, color=color)

    def __plot_r_f(self, investment_universe, label, size=200):
        # TODO: Make it more general
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["r_f"]

        legend_label = self.__complete_label("Risk-free rate", label)
        self.__ax.scatter(0, investment_universe.r_f, s=size, color=color)

    def __plot_tangency_portfolio(self, investment_universe, label, size=200):
        # TODO: Make it more general
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["tangency"]

        legend_label = self.__complete_label("Tangency portfolio", label)
        self.__ax.scatter(
            investment_universe.tangency_portfolio.standard_deviation,
            investment_universe.tangency_portfolio.expected_return, s=size,
            label=legend_label, color=color)

    def __plot_other_portfolios(self, investment_universe, label, size=200):
        # TODO: Make it more general
        color_label = label if label is not None else "1"
        color_iter = iter(self.__colors[color_label]["others"])

        for portfolio_name, portfolio in investment_universe.other_portfolios.items():
            color = next(color_iter)
            weights = [round(weight, 2) for weight in portfolio.weights]
            name_weights = portfolio_name + " - " + str(weights) \
                if self.__weights_visible else portfolio_name
            legend_label = self.__complete_label(name_weights, label)
            self.__ax.scatter(portfolio.standard_deviation,
                              portfolio.expected_return, s=size,
                              label=legend_label, color=color)

    def __check_bool(self, value, variable_name):
        if type(value) is not bool:
            raise TypeError("'{}' must be a boolean!".format(variable_name))

    def __complete_label(self, initial_legend_label, additional_label):
        completed_legend_label = initial_legend_label
        if additional_label is not None:
            completed_legend_label += " - " + additional_label

        return completed_legend_label

    def __set_default_colors(self):

        tab20_cmap = matplotlib.cm.tab20
        tab20b_cmap = matplotlib.cm.tab20b

        colors1 = {
            'feasible': tab20_cmap(0),
            'mvp': tab20_cmap(6),
            'efficient': 'black',
            'tangency': tab20_cmap(4),
            'r_f': 'black',
            'cal': tab20_cmap(2),
            'assets': [tab20b_cmap(i) for i in range(0, 20, 4)],
            'others': [tab20b_cmap(i) for i in range(1, 20, 4)]
        }

        colors2 = {
            'feasible': tab20_cmap(1),
            'mvp': tab20_cmap(7),
            'efficient': 'black',
            'tangency': tab20_cmap(5),
            'r_f': 'black',
            'cal': tab20_cmap(3),
            'assets': [tab20b_cmap(i) for i in range(2, 20, 4)],
            'others': [tab20b_cmap(i) for i in range(3, 20, 4)]
        }

        self.__colors = {
            "1": colors1,
            "2": colors2
        }

    def __set_default_visual_elements_properties(self):

        self.__visual_elements_properties = {
            "tickers": {
                "priority": 10,
                "size": 200
            }
        }

        for inv_univ_index in range(0, self.MAX_NB_INV_UNIV):
            vis_elem_properties = {
                "others": {
                    "priority": 20 - inv_univ_index,
                    "size": 200
                },
                "r_f": {
                    "priority": 30 - inv_univ_index,
                    "size": 200
                },
                "tangency_portfolio": {
                    "priority": 40 - inv_univ_index,
                    "size": 200
                },
                "mvp": {
                    "priority": 50 - inv_univ_index,
                    "size": 200
                },
                "cal": {
                    "priority": 60 - inv_univ_index,
                    "size": 20
                },
                "efficient_portfolios": {
                    "priority": 70 - inv_univ_index,
                    "size": 50
                },
                "feasible_portfolios": {
                    "priority": 80 - inv_univ_index,
                    "size": 50
                }
            }
            self.__visual_elements_properties[
                inv_univ_index] = vis_elem_properties

    def __generate_visual_elements_list(self):

        self.__visual_elements_list = []

        tickers_properties = self.__visual_elements_properties["tickers"]
        if self.__assets_visible:
            self.__visual_elements_list.append(
                self.VisualElement(self.__plot_tickers,
                                   tickers_properties["priority"],
                                   tickers_properties["size"]))

        inv_univ_index = 0
        labels_iter = iter(self.__labels)
        for inv_univ in self.__investment_universes:
            label = next(labels_iter, None)
            properties = self.__visual_elements_properties[inv_univ_index]
            if inv_univ.other_portfolios \
                    and self.__other_portfolios_visible:
                self.__visual_elements_list.append(
                    self.VisualElement(self.__plot_other_portfolios,
                                       properties["others"]["priority"],
                                       properties["others"]["size"], inv_univ,
                                       label))
            if inv_univ.r_f and self.__r_f_visible:
                self.__visual_elements_list.append(
                    self.VisualElement(self.__plot_r_f,
                                       properties["r_f"]["priority"],
                                       properties["r_f"]["size"], inv_univ,
                                       label))
            if inv_univ.tangency_portfolio \
                    and self.__tangency_portfolio_visible:
                self.__visual_elements_list.append(
                    self.VisualElement(self.__plot_tangency_portfolio,
                                       properties["tangency_portfolio"][
                                           "priority"],
                                       properties["tangency_portfolio"][
                                           "size"], inv_univ, label))

            if inv_univ.mvp \
                    and self.__mvp_visible:
                self.__visual_elements_list.append(
                    self.VisualElement(self.__plot_mvp,
                                       properties["mvp"]["priority"],
                                       properties["mvp"]["size"], inv_univ,
                                       label))
            if inv_univ.cal \
                    and self.__cal_visible:
                self.__visual_elements_list.append(
                    self.VisualElement(self.__plot_cal,
                                       properties["cal"]["priority"],
                                       properties["cal"]["size"], inv_univ,
                                       label))
            if inv_univ.efficient_frontier \
                    and self.__efficient_frontier_visible:
                self.__visual_elements_list.append(
                    self.VisualElement(self.__plot_efficient_portfolios,
                                       properties["efficient_portfolios"][
                                           "priority"],
                                       properties["efficient_portfolios"][
                                           "size"], inv_univ, label))
            if inv_univ.feasible_portfolios \
                    and self.__feasible_portfolios_visible:
                self.__visual_elements_list.append(
                    self.VisualElement(self.__plot_feasible_portfolios,
                                       properties["feasible_portfolios"][
                                           "priority"],
                                       properties["feasible_portfolios"][
                                           "size"], inv_univ, label))
            inv_univ_index += 1
