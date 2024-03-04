import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import pyvest


class InvestmentUniverseVisualizer:
    MAX_NB_INV_UNIV = 4

    class VisualElement:
        def __init__(self, plot_function, priority, size,
                     investment_universe=None, label=None, investors=None):
            self.__plot_function = plot_function
            self.__investment_universe = investment_universe
            self.__size = size
            self.__priority = priority
            self.__label = label
            self.__investors = investors

            self.__zorder = 1 / priority

        def plot(self):
            if self.__investment_universe is not None \
                    and self.__investors is not None:
                self.__plot_function(self.__investment_universe,
                                     self.__investors, self.__label,
                                     self.__size, self.__zorder)
            elif self.__investment_universe is not None:
                self.__plot_function(self.__investment_universe, self.__label,
                                     self.__size, self.__zorder)
            else:
                self.__plot_function(self.__label, self.__size, self.__zorder)

        def __lt__(self, other):
            return self.__priority < other.__priority

    def __init__(self, investment_universes, labels=None,
                 default_visibility=True, weights_visible=True,
                 nb_decimal_places=2):

        self.__assign_investment_universes(investment_universes)
        self.__assign_labels(labels)
        self.__generate_assets_inv_univ_dict()

        self.__alpha = 1.0

        self.__ax = None
        self.__fig = None

        self.__set_default_visibility(default_visibility)
        self.__weights_visible = weights_visible
        self.__nb_decimal_places = nb_decimal_places

        self.reset_colors()
        self.__set_default_visual_elements_properties()

        self.__visual_elements_list = None
        self.__investors_colors = {}

    ################################# ATTRIBUTES #################################

    @property
    def assets_visible(self):
        return self.__assets_visible

    @assets_visible.setter
    def assets_visible(self, value):
        self.__check_bool(value, "assets_visible")
        self.__assets_visible = value

    @property
    def feasible_portfolios_with_r_f_visible(self):
        return self.__feasible_portfolios_with_r_f_visible

    @feasible_portfolios_with_r_f_visible.setter
    def feasible_portfolios_with_r_f_visible(self, value):
        self.__check_bool(value, "feasible_portfolios_with_r_f_visible")
        self.__feasible_portfolios_with_r_f_visible = value

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
    def market_portfolio_visible(self):
        return self.__market_portfolio_visible

    @market_portfolio_visible.setter
    def market_portfolio_visible(self, value):
        self.__market_portfolio_visible = value

    @property
    def nb_decimal_places(self):
        return self.__nb_decimal_places

    @nb_decimal_places.setter
    def nb_decimal_places(self, value):
        self.__nb_decimal_places = value

    @property
    def investors(self):
        return self.__investors

    @investors.setter
    def investors(self, value):
        self.__check_bool(value, "investors")
        self.__investors = value

    @property
    def min_expected_return(self):
        return self.__min_expected_return

    @min_expected_return.setter
    def min_expected_return(self, value):
        self.__min_expected_return = value

    @property
    def max_mu(self):
        return self.__max_expected_return

    @max_mu.setter
    def max_mu(self, value):
        self.__max_expected_return = value

    @property
    def min_standard_deviation(self):
        return self.__min_standard_deviation

    @min_standard_deviation.setter
    def min_standard_deviation(self, value):
        self.__min_standard_deviation = value

    @property
    def max_standard_deviation(self):
        return self.__max_standard_deviation

    @max_standard_deviation.setter
    def max_standard_deviation(self, value):
        self.__max_standard_deviation = value

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

    ##################### colors ###################
    @property
    def colors(self):
        return self.__colors

    ##################### visual_elements_properties ###################
    @property
    def visual_elements_properties(self):
        return self.__visual_elements_properties

    ######################### PUBLIC FUNCTIONS ########################

    def plot(self, figsize=(16, 9), zoom_individual=False,
             min_expected_return=None, max_expected_return=None,
             min_standard_deviation=None, max_standard_deviation=None,
             investors=None, indifference_curves=None, legend='upper left'):

        self.__calculate_visible_portfolios_mu_std(zoom_individual, investors)
        self.__set_default_std_limits()
        self.__set_default_mu_limits()

        if min_expected_return is not None:
            self.__min_expected_return = min_expected_return
        if max_expected_return is not None:
            self.__max_expected_return = max_expected_return
        if min_standard_deviation is not None:
            self.__min_standard_deviation = min_standard_deviation
        if max_standard_deviation is not None:
            self.__max_standard_deviation = max_standard_deviation

        self.__fig, self.__ax = plt.subplots(figsize=figsize)

        self.__ax.set_xlim([self.__min_standard_deviation,
                            self.__max_standard_deviation])
        self.__ax.set_ylim([self.__min_expected_return,
                            self.__max_expected_return])
        self.__ax.grid()

        self.__ax.set_title("Risk-return tradeoff", fontsize=35)
        self.__ax.set_ylabel("Expected return", fontsize=30)
        self.__ax.set_xlabel("Standard deviation", fontsize=30)
        self.__ax.tick_params(axis='both', labelsize=25)

        self.__generate_visual_elements_list(investors, indifference_curves)

        sorted_visual_elements = sorted(self.__visual_elements_list,
                                        reverse=True)
        for vis_elem in sorted_visual_elements:
            vis_elem.plot()

        if type(legend) is str:
            self.__ax.legend(fontsize=15, loc=legend)

    ########################## PRIVATE ##########################

    def __set_default_visibility(self, default_visibility):
        self.__assets_visible = default_visibility
        self.__feasible_portfolios_visible = default_visibility
        self.__feasible_portfolios_with_r_f_visible = default_visibility
        self.__mvp_visible = default_visibility
        self.__efficient_frontier_visible = default_visibility
        self.__tangency_portfolio_visible = default_visibility
        self.__cal_visible = default_visibility
        self.__r_f_visible = default_visibility
        self.__other_portfolios_visible = default_visibility
        self.__market_portfolio_visible = default_visibility

    def __set_default_mu_limits(self, border_padding=0.1):

        mu_list = [mu for mu, std in self.__visible_portfolios_mu_std_list]
        border_abs_value = border_padding * max(mu_list)

        self.__min_expected_return = 0 if min(mu_list) >= 0 \
            else min(mu_list) - border_abs_value
        self.__max_expected_return = max(mu_list) + border_abs_value

    def __set_default_std_limits(self, border_padding=0.1):

        std_list = [std for mu, std in self.__visible_portfolios_mu_std_list]
        border_abs_value = border_padding * max(std_list)

        self.__min_standard_deviation = 0 if min(std_list) >= 0 \
            else min(std_list) - border_abs_value
        self.__max_standard_deviation = max(std_list) + border_abs_value

    def __calculate_visible_portfolios_mu_std(self, zoom_individual,
                                              investors):
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
                other_portfolios = [ptf_name_pair[0] for ptf_name_pair
                                    in list(inv_uni.other_portfolios.values())]
                remaining_ptfs_list.extend(other_portfolios)
            if investors is not None:
                investor_portfolios = [inv_uni.investors[x].portfolio
                                       for x in investors]
                remaining_ptfs_list.extend(investor_portfolios)

            if not zoom_individual \
                    and inv_uni.efficient_frontier is not None:
                remaining_ptfs_list.extend(inv_uni.efficient_frontier)
            if not zoom_individual \
                    and inv_uni.feasible_portfolios is not None:
                remaining_ptfs_list.extend(inv_uni.feasible_portfolios)

        self.__visible_portfolios_mu_std_list.extend(
            [(ptf.expected_return, ptf.standard_deviation) for ptf
             in remaining_ptfs_list])

    def __assign_investment_universes(self, investment_universes):
        if isinstance(investment_universes, pyvest.InvestmentUniverse):
            self.__investment_universes = [investment_universes]
        else:
            self.__investment_universes = investment_universes

    def __assign_labels(self, labels):
        generic_labels = ["1", "2", "3", "4"]
        if labels is None and len(self.__investment_universes) > 1:
            self.__labels = generic_labels
        elif labels is None:
            self.__labels = []
        else:
            self.__labels = labels + generic_labels[len(labels):]

    def __generate_assets_inv_univ_dict(self):
        self.__assets_inv_univ_dict = {}
        inv_univ_index = 0
        for investment_universe in self.__investment_universes:
            asset_index = 0
            for asset in investment_universe.assets:
                if asset not in self.__assets_inv_univ_dict:
                    self.__assets_inv_univ_dict[asset] = (inv_univ_index,
                                                          asset_index)
                asset_index += 1
            inv_univ_index += 1

    def __plot_feasible_portfolios(self, investment_universe, label, size, zorder):
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
                          color=color, zorder=zorder)

    def __plot_feasible_portfolios_with_r_f(self, investment_universe, label,
                                            size, zorder):
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["feasible_with_r_f"]

        feasible_portfolios_mu_list = \
            list(map(lambda x: x.expected_return,
                     investment_universe.feasible_portfolios_with_r_f))
        feasible_portfolios_std_list = \
            list(map(lambda x: x.standard_deviation,
                     investment_universe.feasible_portfolios_with_r_f))

        legend_label = self.__complete_label("Feasible portfolios with r_f",
                                             label)
        self.__ax.scatter(feasible_portfolios_std_list,
                          feasible_portfolios_mu_list,
                          s=size,
                          alpha=self.__alpha,
                          label=legend_label,
                          color=color, zorder=zorder)

    def __plot_feasible_portfolios_surface(self, investment_universe, label,
                                           size, zorder):
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["feasible"]

        # feasible_portfolios_std_list = []
        # feasible_portfolios_mu_list = []
        # for ptf_pair in investment_universe.feasible_portfolios_surface:
        #
        #     mu = ptf_pair[0].expected_return
        #     sigma_min = ptf_pair[0].standard_deviation
        #     sigma_max = ptf_pair[1].standard_deviation
        #
        #     feasible_portfolios_std_list.append(sigma_min)
        #     feasible_portfolios_mu_list.append(mu)
        #
        #     feasible_portfolios_std_list.append(sigma_max)
        #     feasible_portfolios_mu_list.append(mu)
        #
        #     for sigma in np.arange(sigma_min, sigma_max + 0.01, 0.01):
        #         feasible_portfolios_std_list.append(sigma)
        #         feasible_portfolios_mu_list.append(mu)

        mu_sigma_tuples_list = []
        for ptf_pair in investment_universe.feasible_portfolios_surface:

            mu = ptf_pair[0].expected_return
            sigma_min = ptf_pair[0].standard_deviation
            sigma_max = ptf_pair[1].standard_deviation

            mu_sigma_tuples_list.append((mu, sigma_min, sigma_max))

            # feasible_portfolios_mu_list.append(mu)
            # feasible_portfolios_std_min_list.append(sigma_min)
            # feasible_portfolios_std_max_list.append(sigma_max)

            # print(mu, sigma_min, sigma_max)

        mu_sigma_tuples_list = sorted(mu_sigma_tuples_list, key=lambda x: x[0])
        mu_sigma_min_max_list = list(zip(*mu_sigma_tuples_list))
        feasible_portfolios_mu_list = mu_sigma_min_max_list[0]
        feasible_portfolios_std_min_list = mu_sigma_min_max_list[1]
        feasible_portfolios_std_max_list = mu_sigma_min_max_list[2]

        legend_label = self.__complete_label("Feasible portfolios", label)
        # self.__ax.scatter(feasible_portfolios_std_list,
        #                   feasible_portfolios_mu_list,
        #                   s=size,
        #                   alpha=self.__alpha,
        #                   label=legend_label,
        #                   color=color, zorder=zorder)
        self.__ax.fill_betweenx(feasible_portfolios_mu_list,
                                feasible_portfolios_std_min_list,
                                feasible_portfolios_std_max_list,
                                linewidth=size,
                                alpha=self.__alpha,
                                label=legend_label,
                                color=color, zorder=zorder)

    def __plot_feasible_portfolios_surface_with_r_f(self, investment_universe,
                                                    label, size, zorder):
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["feasible_with_r_f"]

        mu_sigma_tuples_list = []
        for ptf_pair in investment_universe.feasible_portfolios_surface_with_r_f:
            mu = ptf_pair[0].expected_return
            sigma_min = ptf_pair[0].standard_deviation
            sigma_max = ptf_pair[1].standard_deviation

            mu_sigma_tuples_list.append((mu, sigma_min, sigma_max))

        mu_sigma_tuples_list = sorted(mu_sigma_tuples_list, key=lambda x: x[0])
        mu_sigma_min_max_list = list(zip(*mu_sigma_tuples_list))
        feasible_portfolios_mu_list = mu_sigma_min_max_list[0]
        feasible_portfolios_std_min_list = mu_sigma_min_max_list[1]
        feasible_portfolios_std_max_list = mu_sigma_min_max_list[2]

        legend_label = self.__complete_label("Feasible portfolios with r_f",
                                             label)
        self.__ax.fill_betweenx(feasible_portfolios_mu_list,
                                feasible_portfolios_std_min_list,
                                feasible_portfolios_std_max_list,
                                linewidth=size,
                                alpha=self.__alpha,
                                label=legend_label,
                                color=color, zorder=zorder)

    def __plot_feasible_portfolios_equation(self, investment_universe, label,
                                            size, zorder):
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["feasible"]

        nb_points = 100
        step = (self.__max_expected_return - self.__min_expected_return) / nb_points

        exp_ret_list = list(np.arange(self.__min_expected_return, self.__max_expected_return, step))
        std_list = [investment_universe.feasible_portfolios_equation(mu)
                    for mu in exp_ret_list]

        legend_label = self.__complete_label("Feasible portfolios", label)
        self.__ax.fill_betweenx(exp_ret_list, std_list, self.__max_standard_deviation,
                                linewidth=size,
                                alpha=self.__alpha,
                                label=legend_label,
                                color=color, zorder=zorder)

    def __plot_feasible_portfolios_equation_with_r_f(self, investment_universe,
                                                     label, size, zorder):
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["feasible_with_r_f"]

        nb_points = 100
        step = (self.__max_expected_return - self.__min_expected_return) / nb_points

        exp_ret_list = list(np.arange(self.__min_expected_return, self.__max_expected_return, step))
        std_list = [
            investment_universe.feasible_portfolios_equation_with_r_f(mu)
            for mu in exp_ret_list]

        legend_label = self.__complete_label("Feasible portfolios with r_f",
                                             label)
        self.__ax.fill_betweenx(exp_ret_list, std_list, self.__max_standard_deviation,
                                linewidth=size,
                                alpha=self.__alpha,
                                label=legend_label,
                                color=color, zorder=zorder)

    def __plot_efficient_frontier(self, investment_universe, label, size,
                                  zorder):
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["efficient"]

        efficient_portfolios_mu_list = list(map(lambda x: x.expected_return,
                                                investment_universe.efficient_frontier))
        efficient_portfolios_std_list = list(
            map(lambda x: x.standard_deviation,
                investment_universe.efficient_frontier))
        legend_label = self.__complete_label("Efficient frontier", label)

        self.__ax.plot(efficient_portfolios_std_list,
                       efficient_portfolios_mu_list,
                       linewidth=size, alpha=self.__alpha, label=legend_label,
                       color=color, zorder=zorder)

        # self.__ax.scatter(efficient_portfolios_std_list,
        #                   efficient_portfolios_mu_list, color=color, s=size,
        #                   label=legend_label, zorder=zorder)

    def __plot_efficient_frontier_equation(self, investment_universe, label,
                                           size, zorder):
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["efficient"]

        efficient_frontier_equation = \
            investment_universe.efficient_frontier_equation["equation"]
        mvp_mu_sigma = investment_universe.efficient_frontier_equation["mvp"]
        mvp_mu = mvp_mu_sigma[0]

        nb_points = 100
        step = (self.__max_expected_return - mvp_mu) / nb_points

        exp_ret_list = list(np.arange(mvp_mu, self.__max_expected_return, step))
        std_list = [efficient_frontier_equation(mu) for mu in exp_ret_list]

        legend_label = self.__complete_label("Efficient frontier", label)
        self.__ax.plot(std_list, exp_ret_list, linewidth=size,
                       alpha=self.__alpha, label=legend_label, color=color,
                       zorder=zorder)

    def __plot_cal(self, investment_universe, label, size, zorder):
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["cal"]

        cal_portfolios_mu_list = list(map(lambda x: x.expected_return,
                                          investment_universe.cal))
        cal_portfolios_std_list = list(map(lambda x: x.standard_deviation,
                                           investment_universe.cal))
        legend_label = self.__complete_label("CAL", label)
        self.__ax.plot(cal_portfolios_std_list, cal_portfolios_mu_list,
                       linewidth=size, alpha=self.__alpha, label=legend_label,
                       color=color, zorder=zorder)
        # self.__ax.scatter(cal_portfolios_std_list, cal_portfolios_mu_list,
        #                   s=size, label=legend_label, color=color,
        #                   zorder=zorder)

    def __plot_cal_equation(self, investment_universe, label, size, zorder):
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["cal"]

        nb_points = 100
        step = (self.__max_expected_return - self.__min_expected_return) / nb_points

        exp_ret_list = list(np.arange(investment_universe.r_f, self.__max_expected_return,
                                      step))
        std_list = [investment_universe.cal_equation(mu)
                    for mu in exp_ret_list]

        legend_label = self.__complete_label("CAL", label)
        self.__ax.plot(std_list, exp_ret_list, linewidth=size,
                       alpha=self.__alpha, label=legend_label, color=color,
                       zorder=zorder)

    def __plot_assets(self, label, size, zorder):
        color_label = label if label is not None else "1"
        color_iter = iter(self.__colors[color_label]["assets"])

        for asset, (inv_univ_index, asset_index) \
                in self.__assets_inv_univ_dict.items():
            inv_univ = self.__investment_universes[inv_univ_index]
            color = next(color_iter)

            weights = np.zeros(len(inv_univ.assets)) if inv_univ.r_f is None \
                else np.zeros(len(inv_univ.assets) + 1)
            weights[asset_index] = 1
            weights_str = \
                str([("{:." + str(self.__nb_decimal_places) + "f}").format(
                    round(weight, self.__nb_decimal_places)) for weight
                    in weights]).replace("'", "")

            legend_label = inv_univ.assets[asset_index] + " - " + weights_str

            self.__ax.scatter(inv_univ.std[asset_index],
                              inv_univ.mu[asset_index], s=size,
                              label=legend_label,
                              color=color, zorder=zorder)

    def __plot_mvp(self, investment_universe, label, size, zorder):
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["mvp"]

        weights = str([("{:." + str(self.__nb_decimal_places) + "f}").format(
            round(weight, self.__nb_decimal_places)) for weight
            in investment_universe.mvp.weights]).replace("'", "")

        legend_label = self.__complete_label("MVP - " + weights, label)
        self.__ax.scatter(investment_universe.mvp.standard_deviation,
                          investment_universe.mvp.expected_return, s=size,
                          label=legend_label, color=color, zorder=zorder)

    def __plot_r_f(self, investment_universe, label, size, zorder):
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["r_f"]

        weights = np.zeros(len(investment_universe.assets) + 1)
        weights[-1] = 1
        weights_str = \
            str([("{:." + str(self.__nb_decimal_places) + "f}").format(
                round(weight, self.__nb_decimal_places)) for weight
                in weights]).replace("'", "")

        legend_label = "r_f - " + weights_str

        self.__ax.scatter(0, investment_universe.r_f, s=size,
                          label=legend_label, color=color, zorder=zorder)

    def __plot_tangency_portfolio(self, investment_universe, label, size, zorder):
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["tangency"]

        weights = str([("{:." + str(self.__nb_decimal_places) + "f}").format(
            round(weight, self.__nb_decimal_places)) for weight
            in investment_universe.tangency_portfolio.weights]).replace(
            "'", "")

        legend_label = self.__complete_label("Tangency portfolio - " + weights,
                                             label)
        self.__ax.scatter(
            investment_universe.tangency_portfolio.standard_deviation,
            investment_universe.tangency_portfolio.expected_return, s=size,
            label=legend_label, color=color, zorder=zorder)

    def __plot_other_portfolios(self, investment_universe, label, size, zorder):
        color_label = label if label is not None else "1"
        color_iter = iter(self.__colors[color_label]["others"])

        for portfolio_weights, portfolio_with_name_pair \
                in investment_universe.other_portfolios.items():
            portfolio = portfolio_with_name_pair[0]
            portfolio_name = portfolio_with_name_pair[1]

            weights = str([("{:." + str(self.__nb_decimal_places) + "f}")
                          .format(round(weight, self.__nb_decimal_places))
                           for weight in portfolio_weights]).replace("'", "")

            if portfolio_name is not None:
                name_weights = portfolio_name + " - " + str(weights) \
                    if self.__weights_visible else portfolio_name
            else:
                name_weights = str(weights)

            legend_label = self.__complete_label(name_weights, label)

            color = next(color_iter)

            self.__ax.scatter(portfolio.standard_deviation,
                              portfolio.expected_return, s=size,
                              label=legend_label, color=color, zorder=zorder)

    def __plot_market_portfolio(self, investment_universe, label, size, zorder):
        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["market"]

        weights = str([("{:." + str(self.__nb_decimal_places) + "f}").format(
            round(weight, self.__nb_decimal_places)) for weight
            in investment_universe.market_portfolio.weights]).replace(
            "'", "")

        legend_label = self.__complete_label("Market portfolio - " + weights,
                                             label)
        self.__ax.scatter(
            investment_universe.market_portfolio.standard_deviation,
            investment_universe.market_portfolio.expected_return, s=size,
            label=legend_label, color=color, zorder=zorder)

    def __plot_investors_portfolio(self, investment_universe, investor_names,
                                   label, size, zorder):

        for investor_name in investor_names:
            if investor_name in investment_universe.investors:
                investor = investment_universe.investors[investor_name]
                weights = str(
                    [("{:." + str(self.__nb_decimal_places) + "f}").format(
                        round(weight, self.__nb_decimal_places)) for weight
                        in investor.portfolio.weights]).replace(
                    "'", "")

                legend_label = self.__complete_label(
                    investor_name + " - " + weights,
                    label)

                color = self.__get_investor_color(investor_name, label)
                self.__ax.scatter(
                    investor.portfolio.standard_deviation,
                    investor.portfolio.expected_return,
                    s=size, label=legend_label, color=color, zorder=zorder)

    def __plot_indifference_curves(self, investment_universe, investor_names,
                                   label, size, zorder):

        for investor_name in investor_names:
            if investor_name in investment_universe.investors:
                investor = investment_universe.investors[investor_name]
                portfolio_utility = investor.portfolio_utility

                utility_list = self.get_utility_list(portfolio_utility)

                color = self.__get_investor_color(investor_name, label)
                for utility in utility_list:
                    std_array, mu_array = \
                        investor.calculate_indifference_curve(
                            utility, min_std=self.__min_standard_deviation,
                            max_std=self.__max_standard_deviation)
                    self.__ax.plot(std_array, mu_array, linewidth=size,
                                   alpha=self.__alpha, color=color,
                                   zorder=zorder)

    def get_utility_list(self, portfolio_utility):
        nb_curves = 5

        min_utility = 2 * self.__min_expected_return - self.__max_expected_return
        max_utility = self.__max_expected_return
        utility_step = (self.__max_expected_return - self.__min_expected_return) / nb_curves

        utility_list = []

        utility = portfolio_utility
        utility_list.append(utility)
        while utility > min_utility:
            utility -= utility_step
            utility_list.append(utility)

        while utility < max_utility:
            utility += utility_step
            utility_list.append(utility)

        return utility_list

    def __check_bool(self, value, variable_name):
        if type(value) is not bool:
            raise TypeError("'{}' must be a boolean!".format(variable_name))

    def __complete_label(self, initial_legend_label, additional_label):
        completed_legend_label = initial_legend_label
        if additional_label is not None:
            completed_legend_label += " - " + additional_label

        return completed_legend_label

    def reset_colors(self):

        tab20_cmap = matplotlib.cm.tab20
        tab20b_cmap = matplotlib.cm.tab20b
        tab20c_cmap = matplotlib.cm.tab20c
        dark2_cmap = matplotlib.cm.Dark2
        accent_cmap = matplotlib.cm.Accent

        colors1 = {
            'feasible': tab20_cmap(0),
            'feasible_with_r_f': tab20_cmap(3),
            'mvp': tab20_cmap(6),
            'efficient': 'black',
            'tangency': tab20_cmap(4),
            'r_f': tab20c_cmap(16),
            'cal': tab20_cmap(2),
            'assets': [tab20b_cmap(i) for i in range(0, 20, 4)],
            'others': [tab20b_cmap(i) for i in range(1, 20, 4)] + [
                dark2_cmap(i) for i in range(0, 8)],
            'market': 'yellow',
            'investors': [dark2_cmap(i) for i in range(0, 8, 1)]
        }

        colors2 = {
            'feasible': tab20_cmap(1),
            'feasible_with_r_f': tab20_cmap(13),
            'mvp': tab20_cmap(7),
            'efficient': 'black',
            'tangency': tab20_cmap(5),
            'r_f': tab20c_cmap(16),
            'cal': tab20_cmap(12),
            'assets': [tab20b_cmap(i) for i in range(2, 20, 4)],
            'others': [tab20b_cmap(i) for i in range(3, 20, 4)] + [
                accent_cmap(i) for i in range(0, 8)],
            'market': 'lightskyblue',
            'investors': [dark2_cmap(i) for i in range(0, 8)]
        }

        # TODO: Modify colors3 and colors4 so that the colors are all
        #  different from colors1 and colors2
        colors3 = {
            'feasible': tab20_cmap(2),
            'feasible_with_r_f': tab20_cmap(15),
            'mvp': tab20_cmap(8),
            'efficient': 'black',
            'tangency': tab20_cmap(9),
            'r_f': tab20c_cmap(16),
            'cal': tab20_cmap(14),
            'assets': [tab20b_cmap(i) for i in range(3, 20, 4)],
            'others': [dark2_cmap(i) for i in range(0, 8)] + [
                accent_cmap(i) for i in range(0, 8)],
            'market': 'hotpink',
            'investors': [accent_cmap(i) for i in range(0, 8)]
        }

        colors4 = {
            'feasible': accent_cmap(0),
            'feasible_with_r_f': tab20_cmap(17),
            'mvp': accent_cmap(1),
            'efficient': 'black',
            'tangency': accent_cmap(2),
            'r_f': tab20c_cmap(16),
            'cal': tab20_cmap(16),
            'assets': [tab20b_cmap(i) for i in range(1, 20, 4)],
            'others': [dark2_cmap(i) for i in range(4, 8)] + [
                accent_cmap(i) for i in range(4, 8)],
            'market': 'palegreen',
            'investors': [accent_cmap(i) for i in range(4, 8)]
        }

        if len(self.__labels) > 0:
            self.__colors = {
                self.__labels[0]: colors1,
                self.__labels[1]: colors2,
                self.__labels[2]: colors3,
                self.__labels[3]: colors4
            }
        else:
            self.__colors = {
                "1": colors1,
                "2": colors2,
                "3": colors3,
                "4": colors4
            }

    def __set_default_visual_elements_properties(self):

        self.__visual_elements_properties = {
            "assets": {
                "priority": 10,
                "size": 200
            }
        }

        for inv_univ_index in range(0, self.MAX_NB_INV_UNIV):
            vis_elem_properties = {
                "r_f": {
                    "priority": 5 - inv_univ_index,
                    "size": 200
                },
                "investors": {
                    "priority": 15 - inv_univ_index,
                    "size": 200
                },
                "others": {
                    "priority": 20 - inv_univ_index,
                    "size": 200
                },
                "market_portfolio": {
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
                    "size": 6
                },
                "efficient_portfolios": {
                    "priority": 70 - inv_univ_index,
                    "size": 6
                },
                "feasible_portfolios": {
                    "priority": 80 - inv_univ_index,
                    "size": 0
                },
                "feasible_portfolios_with_r_f": {
                    "priority": 90 - inv_univ_index,
                    "size": 0
                },
                "indifference_curves": {
                    "priority": 100 - inv_univ_index,
                    "size": 20
                }
            }
            self.__visual_elements_properties[
                inv_univ_index] = vis_elem_properties

    def __generate_visual_elements_list(self, investors, indifference_curves):

        if indifference_curves is True:
            indifference_curves = investors
        elif indifference_curves is False:
            indifference_curves = []

        self.__visual_elements_list = []

        assets_properties = self.__visual_elements_properties["assets"]
        if self.__assets_visible:
            assets_label = self.__labels[0] if len(self.__labels) > 0 else None
            self.__visual_elements_list.append(
                self.VisualElement(self.__plot_assets,
                                   assets_properties["priority"],
                                   assets_properties["size"],
                                   label=assets_label))

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
            if inv_univ.cal_equation \
                    and self.__cal_visible:
                self.__visual_elements_list.append(
                    self.VisualElement(self.__plot_cal_equation,
                                       properties["cal"]["priority"],
                                       properties["cal"]["size"], inv_univ,
                                       label))
            if inv_univ.efficient_frontier \
                    and self.__efficient_frontier_visible:
                self.__visual_elements_list.append(
                    self.VisualElement(self.__plot_efficient_frontier,
                                       properties["efficient_portfolios"][
                                           "priority"],
                                       properties["efficient_portfolios"][
                                           "size"], inv_univ, label))
            if inv_univ.efficient_frontier_equation \
                    and self.__efficient_frontier_visible:
                self.__visual_elements_list.append(
                    self.VisualElement(self.__plot_efficient_frontier_equation,
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
            if inv_univ.feasible_portfolios_with_r_f \
                    and self.__feasible_portfolios_with_r_f_visible:
                self.__visual_elements_list.append(
                    self.VisualElement(
                        self.__plot_feasible_portfolios_with_r_f,
                        properties["feasible_portfolios_with_r_f"]["priority"],
                        properties["feasible_portfolios_with_r_f"]["size"],
                        inv_univ, label))
            if inv_univ.feasible_portfolios_surface \
                    and self.__feasible_portfolios_visible:
                self.__visual_elements_list.append(
                    self.VisualElement(self.__plot_feasible_portfolios_surface,
                                       properties["feasible_portfolios"][
                                           "priority"],
                                       properties["feasible_portfolios"][
                                           "size"], inv_univ, label))
            if inv_univ.feasible_portfolios_surface_with_r_f \
                    and self.__feasible_portfolios_visible:
                self.__visual_elements_list.append(
                    self.VisualElement(
                        self.__plot_feasible_portfolios_surface_with_r_f,
                        properties["feasible_portfolios_with_r_f"][
                            "priority"],
                        properties["feasible_portfolios_with_r_f"][
                            "size"], inv_univ, label))
            if inv_univ.feasible_portfolios_equation \
                    and self.__feasible_portfolios_visible:
                self.__visual_elements_list.append(
                    self.VisualElement(
                        self.__plot_feasible_portfolios_equation,
                        properties["feasible_portfolios"][
                            "priority"],
                        properties["feasible_portfolios"][
                            "size"], inv_univ, label))
            if inv_univ.feasible_portfolios_equation_with_r_f \
                    and self.__feasible_portfolios_visible:
                self.__visual_elements_list.append(
                    self.VisualElement(
                        self.__plot_feasible_portfolios_equation_with_r_f,
                        properties["feasible_portfolios_with_r_f"][
                            "priority"],
                        properties["feasible_portfolios_with_r_f"][
                            "size"], inv_univ, label))
            if inv_univ.market_portfolio \
                    and self.__market_portfolio_visible:
                self.__visual_elements_list.append(
                    self.VisualElement(self.__plot_market_portfolio,
                                       properties["market_portfolio"][
                                           "priority"],
                                       properties["market_portfolio"][
                                           "size"], inv_univ, label))
            if investors is not None:
                self.__visual_elements_list.append(
                    self.VisualElement(
                        self.__plot_investors_portfolio,
                        properties["investors"]["priority"],
                        properties["investors"]["size"],
                        inv_univ, label, investors))

            if indifference_curves is not None:
                self.__visual_elements_list.append(
                    self.VisualElement(
                        self.__plot_indifference_curves,
                        properties["indifference_curves"]["priority"],
                        properties["indifference_curves"]["size"],
                        inv_univ, label, indifference_curves))

            inv_univ_index += 1

    def __get_investor_color(self, investor_name, label):

        if investor_name in self.__investors_colors:
            investor_color = self.__investors_colors[investor_name]
        else:
            color_label = label if label is not None else "1"
            color_iter = iter(self.__colors[color_label]["investors"])

            investor_color = next(color_iter)
            while investor_color in self.__investors_colors.values():
                investor_color = next(color_iter)

            self.__investors_colors[investor_name] = investor_color

        return investor_color
