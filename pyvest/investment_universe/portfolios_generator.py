import math
import itertools

import numpy as np

from scipy.optimize import LinearConstraint, minimize, Bounds

import pyvest.general as general


class PortfoliosGenerator:

    def generate_portfolios(self, nb_portfolios, distance):
        raise NotImplementedError(
            "PortfoliosGenerator.generate_portfolios has not been "
            "implemented.")


class FeasiblePortfoliosGenerator(PortfoliosGenerator):

    def __init__(self, investment_universe, with_r_f=False):
        self.__investment_universe = investment_universe
        self.__with_r_f = with_r_f

    def generate_portfolios(self, nb_portfolios, distance):
        raise NotImplementedError(
            "FeasiblePortfoliosGenerator.generate_portfolios has not been "
            "implemented.")

    @property
    def investment_universe(self):
        return self.__investment_universe

    @property
    def with_r_f(self):
        return self.__with_r_f


class FeasiblePortfoliosGeneratorRandom(FeasiblePortfoliosGenerator):

    def __init__(self, investment_universe, with_r_f=False):
        super().__init__(investment_universe, with_r_f)

        self.__mu = investment_universe.mu
        self.__cov = investment_universe.cov
        self.__r_f = investment_universe.r_f
        self.__assets = investment_universe.assets

        if with_r_f is False:
            self.__min_weight = investment_universe.min_weight
        else:
            self.__min_weight = np.append(investment_universe.min_weight,
                                          investment_universe.min_weight_r_f)

    def generate_portfolios(self, nb_portfolios, distance=None):

        feasible_portfolios = []
        for i in range(1, nb_portfolios):
            risky_assets_portfolio_weights = \
                self.__calculate_random_portfolio_weights()

            if not self.with_r_f and self.__r_f is not None:
                portfolio_weights = \
                    np.append(risky_assets_portfolio_weights, [0])
            else:
                portfolio_weights = risky_assets_portfolio_weights

            portfolio = general.Portfolio(portfolio_weights, self.__mu,
                                          self.__cov, r_f=self.__r_f,
                                          assets=self.__assets)
            feasible_portfolios.append(portfolio)

        return feasible_portfolios

    def __calculate_random_portfolio_weights(self):
        # This function adds random portfolio weights
        # The argument "smallest_weight" denotes the smallest weight
        # admissible for a given asset. For example, "smallest_weight=0"
        # indicates that short sales are not allowed, and "smallest_weight=-1"
        # implies that the weight of each asset in the portfolio must be equal
        # or greater to -1. The function returns an array of portfolio weights

        weights = np.random.dirichlet(np.ones(len(self.__min_weight)),
                                      size=1)[0]
        norm_weights = \
            weights * (1 - sum(self.__min_weight)) + self.__min_weight

        return norm_weights


class FeasiblePortfoliosGeneratorFrontier(FeasiblePortfoliosGenerator):

    def __init__(self, investment_universe, with_r_f=False):
        super().__init__(investment_universe, with_r_f)

        if with_r_f:
            self.__augmented_mu = investment_universe.augmented_mu
            self.__augmented_cov = investment_universe.augmented_cov

        self.__mu = investment_universe.mu
        self.__cov = investment_universe.cov

        self.__r_f = investment_universe.r_f
        self.__assets = investment_universe.assets

        self.__nb_risky_assets = len(self.__mu)
        self.__nb_assets = self.__nb_risky_assets if not with_r_f \
            else self.__nb_risky_assets + 1

        if not with_r_f:
            self.__min_weight = investment_universe.min_weight
        else:
            self.__min_weight = np.append(investment_universe.min_weight,
                                          investment_universe.min_weight_r_f)

        self.__portfolios_list = None
        self.__frontier = None
        self.__min_portfolios_list = None
        self.__max_portfolios_list = None
        self.__filtered_max_portfolios_list = None
        self.__filtered_min_portfolios_list = None

        self.__step = None
        self.__factor = None

        self.__max_portfolios_list = []

    @property
    def frontier(self):
        return self.__frontier

    def generate_portfolios(self, nb_portfolios=None, distance=None):

        self.__portfolios_list = []
        self.__frontier = []
        self.__min_portfolios_list = []

        x0 = np.ones(self.__nb_assets) / self.__nb_assets

        tolerance = 1e-8

        self.__calculate_max_portfolios()
        self.__filter_max_portfolios()

        self.__portfolios_list = []
        for ptf_max in self.__filtered_max_portfolios_list:

            mu = ptf_max.expected_return
            ptf_min = self.__calculate_frontier_portfolio_from_mu(
                mu, x0, tolerance, maximize=False)

            if ptf_min is not None:
                self.__min_portfolios_list.append(ptf_min)
                self.__portfolios_list.append(ptf_min)
                self.__portfolios_list.append(ptf_max)
                self.__frontier.append((ptf_min, ptf_max))

        return self.__portfolios_list

    def __calculate_frontier_portfolio_from_mu(self, mu, x0, tolerance,
                                               maximize=False):

        efficient_portfolio = None

        factor = -1 if maximize else 1

        self.__min_weights_bound = Bounds(self.__min_weight,
                                          np.inf)

        if self.with_r_f:
            # Sum portfolio weights equals 1 constraint
            self.__sum_weights_assets_equals_one_constraint = LinearConstraint(
                np.ones(self.__nb_assets), 1, 1)

            efficient_mu_constraint = LinearConstraint(self.__augmented_mu.T,
                                                       mu, mu)

            efficient_portfolio_result = minimize(
                lambda
                    x: factor * general.calculate_portfolio_standard_deviation(
                    x, self.__augmented_cov),
                x0,
                bounds=self.__min_weights_bound,
                constraints=[self.__sum_weights_assets_equals_one_constraint,
                             efficient_mu_constraint],
                tol=tolerance)
        else:
            # Sum portfolio weights equals 1 constraint
            self.__sum_weights_assets_equals_one_constraint = LinearConstraint(
                np.ones(self.__nb_assets), 1, 1)

            efficient_mu_constraint = LinearConstraint(self.__mu.T, mu, mu)
            efficient_portfolio_result = minimize(
                lambda
                    x: factor * general.calculate_portfolio_standard_deviation(
                    x, self.__cov),
                x0,
                bounds=self.__min_weights_bound,
                constraints=[self.__sum_weights_assets_equals_one_constraint,
                             efficient_mu_constraint],
                tol=tolerance)

        if not efficient_portfolio_result.success:
            print("minimize was not successful with bounds={} and "
                  "constraints={}!".format(self.__min_weights_bound, mu))
        else:

            if self.__r_f is not None and not self.with_r_f:
                efficient_portfolio_weights = \
                    np.append(efficient_portfolio_result.x, [0])
            else:
                efficient_portfolio_weights = efficient_portfolio_result.x

            efficient_portfolio = general.Portfolio(
                efficient_portfolio_weights,
                self.__mu, self.__cov,
                r_f=self.__r_f,
                assets=self.__assets)

        return efficient_portfolio

    def __calculate_max_portfolios(self):

        step = 0.0001

        index_pairs_list = itertools.combinations(range(self.__nb_assets), 2)

        for index_pair in index_pairs_list:
            weights_array = np.zeros(self.__nb_assets)
            for x in np.arange(0, 1 + step, step):
                weights_array[index_pair[0]] = x
                weights_array[index_pair[1]] = 1 - x
                portfolio = self.__create_portfolio(weights_array)
                self.__max_portfolios_list.append(portfolio)

    def __filter_max_portfolios(self):

        mu_step = 0.005

        highest_std_ptfs_dict = {}
        for ptf in self.__max_portfolios_list:
            mu_index = math.floor(ptf.expected_return / mu_step)
            if mu_index not in highest_std_ptfs_dict \
                    or (ptf.standard_deviation
                        > highest_std_ptfs_dict[mu_index].standard_deviation):
                highest_std_ptfs_dict[mu_index] = ptf

        self.__filtered_max_portfolios_list = highest_std_ptfs_dict.values()

    def __create_portfolio(self, normalized_weights):

        portfolio_weights = self.__inverse_normalize_weights(
            normalized_weights)

        if self.__r_f is not None and not self.with_r_f:
            portfolio_weights = \
                np.append(portfolio_weights, [0])

        portfolio = general.Portfolio(portfolio_weights,
                                      self.__mu,
                                      self.__cov, r_f=self.__r_f,
                                      assets=self.__assets)

        return portfolio

    def __inverse_normalize_weights(self, normalized_weights):

        weights = normalized_weights * (1 - sum(self.__min_weight)) \
                  + self.__min_weight

        return weights

    def __normalize_weights(self, weights):

        normalized_weights = (weights - self.__min_weight) \
                             / (1 - sum(self.__min_weight))

        return normalized_weights
