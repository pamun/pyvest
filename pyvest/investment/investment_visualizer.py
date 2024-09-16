import matplotlib
import matplotlib.pyplot as plt

import pyvest.investment as investment_module


class InvestmentVisualizer:
    MAX_NB_INVESTMENTS = 4

    class VisualElement:
        def __init__(self, plot_function, priority, data=None, style=None):
            self.__plot_function = plot_function
            self.__data = data
            self.__style = style
            self.__priority = priority

            self.__zorder = 1 / self.__priority

        def plot(self):
            self.__plot_function(self.__data, self.__style,
                                 self.__zorder)

        def __lt__(self, other):
            return self.__priority < other.__priority

    def __init__(self, investment, labels=None):

        self.__assign_investments(investment)
        self.__assign_labels(labels)
        self.__set_default_colors()
        self.__set_default_visual_elements_properties()

        self.__ax = None
        self.__fig = None

    @property
    def investments(self):
        return self.__investments

    @investments.setter
    def investments(self, value):
        self.__assign_investments(value)
        self.__assign_labels(None)

    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, value):
        self.__assign_labels(value)

    @property
    def fig(self):
        return self.__fig

    @property
    def ax(self):
        return self.__ax

    def plot_profit(self, start_date=None, end_date=None, figsize=(16, 9),
                    legend='best', show_transactions=False):
        self.__prepare_profit_plot(figsize)

        visual_elements = self.__generate_profits_visual_elements(
            start_date, end_date)

        transactions_visual_elements = []
        if show_transactions:
            transactions_visual_elements = \
                self.__generate_transactions_visual_elements(start_date,
                                                             end_date)

        sorted_visual_elements = \
            sorted(visual_elements + transactions_visual_elements,
                   reverse=True)
        for vis_elem in sorted_visual_elements:
            vis_elem.plot()

        if type(legend) is str:
            self.__ax.legend(fontsize=15, loc=legend)

    def plot_return(self, start_date=None, end_date=None, figsize=(16, 9),
                    legend='best', show_transactions=False):
        self.__prepare_return_plot(figsize)

        visual_elements = self.__generate_returns_visual_elements(
            start_date, end_date)

        transactions_visual_elements = []
        if show_transactions:
            transactions_visual_elements = \
                self.__generate_transactions_visual_elements(start_date,
                                                             end_date)

        sorted_visual_elements = \
            sorted(visual_elements + transactions_visual_elements,
                   reverse=True)
        for vis_elem in sorted_visual_elements:
            vis_elem.plot()

        if type(legend) is str:
            self.__ax.legend(fontsize=15, loc=legend)

    def plot_weight(self, start_date=None, end_date=None, figsize=(16, 9),
                    legend='best', show_transactions=False):
        self.__prepare_weight_plot(figsize)

        visual_elements = self.__generate_weight_visual_elements(
            start_date, end_date)

        transactions_visual_elements = []
        if show_transactions:
            transactions_visual_elements = \
                self.__generate_transactions_visual_elements(start_date,
                                                             end_date)

        sorted_visual_elements = \
            sorted(visual_elements + transactions_visual_elements,
                   reverse=True)
        for vis_elem in sorted_visual_elements:
            vis_elem.plot()

        if type(legend) is str:
            self.__ax.legend(fontsize=15, loc=legend)

    def plot_value(self, start_date=None, end_date=None, figsize=(16, 9),
                    legend='best', show_transactions=False):
        self.__prepare_value_plot(figsize)

        visual_elements = self.__generate_value_visual_elements(
            start_date, end_date)

        transactions_visual_elements = []
        if show_transactions:
            transactions_visual_elements = \
                self.__generate_transactions_visual_elements(start_date,
                                                             end_date)

        sorted_visual_elements = \
            sorted(visual_elements + transactions_visual_elements,
                   reverse=True)
        for vis_elem in sorted_visual_elements:
            vis_elem.plot()

        if type(legend) is str:
            self.__ax.legend(fontsize=15, loc=legend)

    def __prepare_profit_plot(self, figsize):
        self.__fig, self.__ax = plt.subplots(figsize=figsize)
        self.__ax.grid()

        self.__ax.set_title("Profit", fontsize=35)
        self.__ax.set_ylabel("Value ($)", fontsize=30)
        self.__ax.set_xlabel("Date", fontsize=30)
        self.__ax.tick_params(axis='both', labelsize=25)

    def __prepare_return_plot(self, figsize):
        self.__fig, self.__ax = plt.subplots(figsize=figsize)
        self.__ax.grid()

        self.__ax.set_title("Return", fontsize=35)
        self.__ax.set_ylabel("Percentage (%)", fontsize=30)
        self.__ax.set_xlabel("Date", fontsize=30)
        self.__ax.tick_params(axis='both', labelsize=25)

    def __prepare_weight_plot(self, figsize):
        self.__fig, self.__ax = plt.subplots(figsize=figsize)
        self.__ax.grid()

        self.__ax.set_title("Weight by asset", fontsize=35)
        self.__ax.set_ylabel("Weight", fontsize=30)
        self.__ax.set_xlabel("Date", fontsize=30)
        self.__ax.tick_params(axis='both', labelsize=25)

    def __prepare_value_plot(self, figsize):
        self.__fig, self.__ax = plt.subplots(figsize=figsize)
        self.__ax.grid()

        self.__ax.set_title("Value by asset", fontsize=35)
        self.__ax.set_ylabel("Value ($)", fontsize=30)
        self.__ax.set_xlabel("Date", fontsize=30)
        self.__ax.tick_params(axis='both', labelsize=25)

    def __assign_investments(self, investments):
        if isinstance(investments, investment_module.Investment):
            self.__investments = [investments]
        else:
            self.__investments = investments

    def __assign_labels(self, labels):
        generic_labels = ["1", "2", "3", "4"]
        if labels is None and len(self.__investments) > 1:
            self.__labels = generic_labels
        elif labels is None:
            self.__labels = []
        else:
            self.__labels = labels + generic_labels[len(labels):]

    def __set_default_colors(self):

        tab10_cmap = matplotlib.cm.tab10
        tab20b_cmap = matplotlib.cm.tab20b
        set2_cmap = matplotlib.cm.Set2
        dark2_cmap = matplotlib.cm.Dark2

        colors1 = {
            'unrealized_capital_gain': tab10_cmap(0),
            'realized_capital_gain': tab10_cmap(1),
            'dividends': tab10_cmap(2),
            'total_profits': tab10_cmap(3),
            'buy_transactions': tab10_cmap(0),
            'sell_transactions': tab10_cmap(1),
            'dividend_transactions': tab10_cmap(2),
            'rebalance_transactions': tab10_cmap(4),
            'total_returns': dark2_cmap(0),
            'assets': [tab20b_cmap(i) for i in range(0, 20, 4)]
        }

        colors2 = {
            'unrealized_capital_gain': set2_cmap(4),
            'realized_capital_gain': set2_cmap(5),
            'dividends': set2_cmap(6),
            'total_profits': set2_cmap(7),
            'buy_transactions': set2_cmap(4),
            'sell_transactions': set2_cmap(5),
            'dividend_transactions': set2_cmap(6),
            'rebalance_transactions': set2_cmap(0),
            'total_returns': dark2_cmap(1),
            'weight': dark2_cmap(1),
            'assets': [tab20b_cmap(i) for i in range(2, 20, 4)]
        }

        colors3 = {
            'unrealized_capital_gain': dark2_cmap(0),
            'realized_capital_gain': dark2_cmap(1),
            'dividends': dark2_cmap(2),
            'total_profits': dark2_cmap(3),
            'assets': [tab20b_cmap(i) for i in range(3, 20, 4)]
        }

        colors4 = {
            'unrealized_capital_gain': dark2_cmap(4),
            'realized_capital_gain': dark2_cmap(5),
            'dividends': dark2_cmap(6),
            'total_profits': dark2_cmap(7),
            'assets': [tab20b_cmap(i) for i in range(1, 20, 4)]
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

        self.__visual_elements_properties = {}

        for inv_univ_index in range(0, self.MAX_NB_INVESTMENTS):
            vis_elem_properties = {
                "unrealized_capital_gains": {
                    "priority": 30 - inv_univ_index,
                    "size": 3
                },
                "realized_capital_gains": {
                    "priority": 25 - inv_univ_index,
                    "size": 3
                },
                "dividends": {
                    "priority": 20 - inv_univ_index,
                    "size": 3
                },
                "total_profits": {
                    "priority": 15 - inv_univ_index,
                    "size": 3
                },
                "buy_transactions": {
                    "priority": 10 - inv_univ_index,
                    "size": 3
                },
                "sell_transactions": {
                    "priority": 10 - inv_univ_index,
                    "size": 3
                },
                "dividend_transactions": {
                    "priority": 10 - inv_univ_index,
                    "size": 3
                },
                "rebalance_transactions": {
                    "priority": 10 - inv_univ_index,
                    "size": 3
                },
                "total_returns": {
                    "priority": 15 - inv_univ_index,
                    "size": 3
                },
                "weight": {
                    "priority": 15 - inv_univ_index,
                    "size": 3
                },
                "value": {
                    "priority": 15 - inv_univ_index,
                    "size": 3
                }
            }
            self.__visual_elements_properties[
                inv_univ_index] = vis_elem_properties

    def __complete_label(self, initial_legend_label, additional_label):
        completed_legend_label = initial_legend_label
        if additional_label is not None:
            completed_legend_label += " - " + additional_label

        return completed_legend_label

    def __plot_unrealized_capital_gains(self, profits, style, zorder):

        label = style["label"]
        size = style["size"]

        dates = [profit.date for profit in profits]
        unrealized_capital_gains = [profit.unrealized_capital_gain
                                    for profit in profits]

        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["unrealized_capital_gain"]
        legend_label = self.__complete_label("Unrealized Capital Gains", label)

        self.__ax.plot(dates, unrealized_capital_gains, label=legend_label,
                       color=color, linewidth=size, zorder=zorder)

    def __plot_realized_capital_gains(self, profits, style, zorder):

        label = style["label"]
        size = style["size"]

        dates = [profit.date for profit in profits]
        realized_capital_gains = [profit.realized_capital_gain
                                  for profit in profits]

        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["realized_capital_gain"]
        legend_label = self.__complete_label("Realized Capital Gains", label)

        self.__ax.plot(dates, realized_capital_gains, label=legend_label,
                       color=color, linewidth=size, zorder=zorder)

    def __plot_dividends(self, data, style, zorder):

        label = style["label"]
        size = style["size"]

        profits = data["profits"]
        reinvest_dividends = data["reinvest_dividends"]

        dates = [profit.date for profit in profits]

        if reinvest_dividends:
            dividends = [profit.reinvested_dividends for profit in profits]
        else:
            dividends = [profit.non_reinvested_dividends for profit in profits]

        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["dividends"]
        legend_label = self.__complete_label("Dividends", label)

        self.__ax.plot(dates, dividends, label=legend_label,
                       color=color, linewidth=size, zorder=zorder)

    def __plot_total_profits(self, profits, style, zorder):

        label = style["label"]
        size = style["size"]

        dates = [profit.date for profit in profits]

        total_profits = [profit.non_reinvested_dividends
                         + profit.reinvested_dividends
                         + profit.unrealized_capital_gain
                         + profit.realized_capital_gain
                         for profit in profits]

        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["total_profits"]
        legend_label = self.__complete_label("Total Profits", label)

        self.__ax.plot(dates, total_profits, label=legend_label, color=color,
                       linewidth=size, zorder=zorder)

    def __plot_buy_transactions(self, data_dict, style, zorder):

        label = style["label"]
        size = style["size"]

        investment = data_dict["investment"]
        start_date = data_dict["start_date"]
        end_date = data_dict["end_date"]

        transactions = investment.get_transactions(transaction_type="BUY",
                                                   start_date=start_date,
                                                   end_date=end_date)

        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["buy_transactions"]
        legend_label = self.__complete_label("Buy", label)

        nb_lines = 0
        for date, transactions_list in transactions.items():
            vline_label = legend_label if nb_lines == 0 else None
            self.__ax.axvline(x=date, label=vline_label, color=color,
                              linestyle="dotted",
                              linewidth=size, zorder=zorder)
            nb_lines += 1

    def __plot_sell_transactions(self, data_dict, style, zorder):

        label = style["label"]
        size = style["size"]

        investment = data_dict["investment"]
        start_date = data_dict["start_date"]
        end_date = data_dict["end_date"]

        transactions = investment.get_transactions(transaction_type="SELL",
                                                   start_date=start_date,
                                                   end_date=end_date)

        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["sell_transactions"]
        legend_label = self.__complete_label("Sell", label)

        nb_lines = 0
        for date, transactions_list in transactions.items():
            vline_label = legend_label if nb_lines == 0 else None
            self.__ax.axvline(x=date, label=vline_label, color=color,
                              linestyle="dotted",
                              linewidth=size, zorder=zorder)
            nb_lines += 1

    def __plot_dividend_transactions(self, data_dict, style, zorder):

        label = style["label"]
        size = style["size"]

        investment = data_dict["investment"]
        start_date = data_dict["start_date"]
        end_date = data_dict["end_date"]

        transaction_type = "REINVESTED_DIVIDEND" \
            if investment.reinvest_dividends \
            else investment.reinvest_dividends
        transactions = investment.get_transactions(
            transaction_type=transaction_type, start_date=start_date,
            end_date=end_date)

        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["dividend_transactions"]
        legend_label = self.__complete_label("Dividend", label)

        nb_lines = 0
        for date, transactions_list in transactions.items():
            vline_label = legend_label if nb_lines == 0 else None
            self.__ax.axvline(x=date, label=vline_label, color=color,
                              linestyle="dotted",
                              linewidth=size, zorder=zorder)
            nb_lines += 1

    def __plot_rebalance_transactions(self, data_dict, style, zorder):

        label = style["label"]
        size = style["size"]

        investment = data_dict["investment"]
        start_date = data_dict["start_date"]
        end_date = data_dict["end_date"]

        transactions = investment.get_transactions(
            transaction_type="REBALANCE", start_date=start_date,
            end_date=end_date)

        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["rebalance_transactions"]
        legend_label = self.__complete_label("Rebalance", label)

        nb_lines = 0
        for date, transactions_list in transactions.items():
            vline_label = legend_label if nb_lines == 0 else None
            self.__ax.axvline(x=date, label=vline_label, color=color,
                              linestyle="dotted",
                              linewidth=size, zorder=zorder)
            nb_lines += 1

    def __plot_total_returns(self, returns, style, zorder):

        label = style["label"]
        size = style["size"]

        dates = [inv_return.date for inv_return in returns]
        total_returns = [inv_return.total_return for inv_return in returns]

        color_label = label if label is not None else "1"
        color = self.__colors[color_label]["total_returns"]
        legend_label = self.__complete_label("Total Return", label)

        self.__ax.plot(dates, total_returns, label=legend_label,
                       color=color, linewidth=size, zorder=zorder)

    def __plot_weight(self, dated_weights_list, style, zorder):

        label = style["label"]
        size = style["size"]
        color = style["color"]

        dates = [dated_weight.date for dated_weight in dated_weights_list]
        weights = [dated_weight.weight for dated_weight
                   in dated_weights_list]

        if len(dated_weights_list) > 0:
            ticker = dated_weights_list[0].ticker
        else:
            ticker = ""

        legend_label = self.__complete_label(ticker, label)

        self.__ax.plot(dates, weights, label=legend_label,
                       color=color, linewidth=size, zorder=zorder)

    def __plot_value(self, dated_values_list, style, zorder):

        label = style["label"]
        size = style["size"]
        color = style["color"]

        dates = [dated_value.date for dated_value in dated_values_list]
        values = [dated_value.value for dated_value
                   in dated_values_list]

        if len(dated_values_list) > 0:
            ticker = dated_values_list[0].ticker
        else:
            ticker = ""

        legend_label = self.__complete_label(ticker, label)

        self.__ax.plot(dates, values, label=legend_label,
                       color=color, linewidth=size, zorder=zorder)

    def __generate_profits_visual_elements(self, start_date, end_date):
        visual_elements = []

        investment_index = 0
        labels_iter = iter(self.__labels)
        for investment in self.__investments:
            profits = investment.calculate_profit_range(start_date=start_date,
                                                        end_date=end_date)

            label = next(labels_iter, None)
            properties = self.__visual_elements_properties[investment_index]

            visual_elements.append(
                self.VisualElement(self.__plot_unrealized_capital_gains,
                                   properties["unrealized_capital_gains"][
                                       "priority"],
                                   data=profits,
                                   style={
                                       "size": properties[
                                           "unrealized_capital_gains"]["size"],
                                       "label": label
                                   }))

            visual_elements.append(
                self.VisualElement(self.__plot_realized_capital_gains,
                                   properties["realized_capital_gains"][
                                       "priority"],
                                   data=profits,
                                   style={
                                       "size": properties[
                                           "realized_capital_gains"]["size"],
                                       "label": label
                                   }))

            visual_elements.append(
                self.VisualElement(self.__plot_dividends,
                                   properties["dividends"][
                                       "priority"],
                                   data={
                                       "profits": profits,
                                       "reinvest_dividends":
                                           investment.reinvest_dividends
                                   },
                                   style={
                                       "size": properties[
                                           "dividends"]["size"],
                                       "label": label
                                   }))

            visual_elements.append(
                self.VisualElement(self.__plot_total_profits,
                                   properties["total_profits"]["priority"],
                                   data=profits,
                                   style={
                                       "size": properties[
                                           "total_profits"]["size"],
                                       "label": label
                                   }))

            investment_index += 1

        return visual_elements

    def __generate_transactions_visual_elements(self, start_date, end_date):
        visual_elements = []

        investment_index = 0
        labels_iter = iter(self.__labels)
        for investment in self.__investments:
            data_dict = {"investment": investment,
                         "start_date": start_date,
                         "end_date": end_date}

            label = next(labels_iter, None)
            properties = self.__visual_elements_properties[investment_index]

            visual_elements.append(
                self.VisualElement(self.__plot_buy_transactions,
                                   properties["buy_transactions"][
                                       "priority"],
                                   data=data_dict,
                                   style={
                                       "size": properties[
                                           "buy_transactions"]["size"],
                                       "label": label
                                   }))

            visual_elements.append(
                self.VisualElement(self.__plot_sell_transactions,
                                   properties["sell_transactions"][
                                       "priority"],
                                   data=data_dict,
                                   style={
                                       "size": properties[
                                           "sell_transactions"]["size"],
                                       "label": label
                                   }))

            visual_elements.append(
                self.VisualElement(self.__plot_dividend_transactions,
                                   properties["dividend_transactions"][
                                       "priority"],
                                   data=data_dict,
                                   style={
                                       "size": properties[
                                           "dividend_transactions"]["size"],
                                       "label": label
                                   }))

            visual_elements.append(
                self.VisualElement(self.__plot_rebalance_transactions,
                                   properties["rebalance_transactions"][
                                       "priority"],
                                   data=data_dict,
                                   style={
                                       "size": properties[
                                           "rebalance_transactions"]["size"],
                                       "label": label
                                   }))

            investment_index += 1

        return visual_elements

    def __generate_returns_visual_elements(self, start_date, end_date):
        visual_elements = []

        investment_index = 0
        labels_iter = iter(self.__labels)
        for investment in self.__investments:
            returns = investment.calculate_return_range(start_date=start_date,
                                                        end_date=end_date)

            label = next(labels_iter, None)
            properties = self.__visual_elements_properties[investment_index]

            visual_elements.append(
                self.VisualElement(self.__plot_total_returns,
                                   properties["total_returns"][
                                       "priority"],
                                   data=returns,
                                   style={
                                       "size": properties[
                                           "total_returns"]["size"],
                                       "label": label
                                   }))

            investment_index += 1

        return visual_elements

    def __generate_weight_visual_elements(self, start_date, end_date):
        visual_elements = []

        investment_index = 0
        labels_iter = iter(self.__labels)
        for investment in self.__investments:
            dated_weight_by_ticker_list = investment.calculate_weight_range(
                start_date=start_date, end_date=end_date)

            label = next(labels_iter, None)
            properties = self.__visual_elements_properties[investment_index]

            dated_weights_list_by_ticker = {}
            for dated_weight_by_ticker in dated_weight_by_ticker_list:
                for ticker, dated_weight in dated_weight_by_ticker.items():
                    if ticker not in dated_weights_list_by_ticker:
                        dated_weights_list_by_ticker[ticker] = []
                    dated_weights_list_by_ticker[ticker].append(dated_weight)

            color_label = label if label is not None else "1"
            color_iter = iter(self.__colors[color_label]["assets"])
            for ticker, dated_weights_list \
                    in dated_weights_list_by_ticker.items():
                color = next(color_iter)
                visual_elements.append(
                    self.VisualElement(self.__plot_weight,
                                       properties["weight"][
                                           "priority"],
                                       data=dated_weights_list,
                                       style={
                                           "size": properties[
                                               "total_returns"]["size"],
                                           "label": label,
                                           "color": color
                                       }))

            investment_index += 1

        return visual_elements

    def __generate_value_visual_elements(self, start_date, end_date):
        visual_elements = []

        investment_index = 0
        labels_iter = iter(self.__labels)
        for investment in self.__investments:
            dated_value_by_ticker_list = investment.calculate_value_range(
                start_date=start_date, end_date=end_date)

            label = next(labels_iter, None)
            properties = self.__visual_elements_properties[investment_index]

            dated_values_list_by_ticker = {}
            for dated_value_by_ticker in dated_value_by_ticker_list:
                for ticker, dated_value in dated_value_by_ticker.items():
                    if ticker not in dated_values_list_by_ticker:
                        dated_values_list_by_ticker[ticker] = []
                    dated_values_list_by_ticker[ticker].append(dated_value)

            color_label = label if label is not None else "1"
            color_iter = iter(self.__colors[color_label]["assets"])
            for ticker, dated_values_list \
                    in dated_values_list_by_ticker.items():
                color = next(color_iter)
                visual_elements.append(
                    self.VisualElement(self.__plot_value,
                                       properties["value"][
                                           "priority"],
                                       data=dated_values_list,
                                       style={
                                           "size": properties[
                                               "total_returns"]["size"],
                                           "label": label,
                                           "color": color
                                       }))

            investment_index += 1

        return visual_elements
