import matplotlib.pyplot as plt
import matplotlib

import pyvest


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
        self.__generate_assets_inv_univ_dict()

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

    def plot(self, figsize=(16, 9)):

        self.__fig, self.__ax = plt.subplots(figsize=figsize)

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
        if isinstance(investment_universes, pyvest.InvestmentUniverse):
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

    def __plot_assets(self, label, size=200):
        # TODO: Make it more general
        color_label = label if label is not None else "1"
        color_iter = iter(self.__colors[color_label]["assets"])

        for asset, (inv_univ_index, asset_index) \
                in self.__assets_inv_univ_dict.items():
            inv_univ = self.__investment_universes[inv_univ_index]
            color = next(color_iter)
            self.__ax.scatter(inv_univ.std[asset_index],
                              inv_univ.mu[asset_index], s=size,
                              label=inv_univ.assets[asset_index],
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
        dark2_cmap = matplotlib.cm.Dark2
        accent_cmap = matplotlib.cm.Accent

        colors1 = {
            'feasible': tab20_cmap(0),
            'mvp': tab20_cmap(6),
            'efficient': 'black',
            'tangency': tab20_cmap(4),
            'r_f': 'black',
            'cal': tab20_cmap(2),
            'assets': [tab20b_cmap(i) for i in range(0, 20, 4)],
            'others': [tab20b_cmap(i) for i in range(1, 20, 4)] + [
                dark2_cmap(i) for i in range(0, 8)]
        }

        colors2 = {
            'feasible': tab20_cmap(1),
            'mvp': tab20_cmap(7),
            'efficient': 'black',
            'tangency': tab20_cmap(5),
            'r_f': 'black',
            'cal': tab20_cmap(3),
            'assets': [tab20b_cmap(i) for i in range(2, 20, 4)],
            'others': [tab20b_cmap(i) for i in range(3, 20, 4)] + [
                accent_cmap(i) for i in range(0, 8)]
        }

        self.__colors = {
            "1": colors1,
            "2": colors2
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

        assets_properties = self.__visual_elements_properties["assets"]
        if self.__assets_visible:
            self.__visual_elements_list.append(
                self.VisualElement(self.__plot_assets,
                                   assets_properties["priority"],
                                   assets_properties["size"]))

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
