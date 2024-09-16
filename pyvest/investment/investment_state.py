import numpy as np

from pyvest.investment import Profit


class InvestmentState:
    def __init__(self, investment, quantity_by_ticker=None,
                 total_cost_by_ticker=None, average_cost_by_ticker=None,
                 total_commission_by_ticker=None, asset_price_by_ticker=None,
                 cash=None, total_invested=None, profit=None,
                 investment_state_to_copy=None):

        self.__investment = investment

        self.__init_values_by_ticker()

        self.__cash = 0
        self.__total_invested = 0
        self.__profit = Profit(self.__investment)

        if investment_state_to_copy is not None:
            self.__copy(investment_state_to_copy)

        self.__initialize_attributes(quantity_by_ticker, total_cost_by_ticker,
                                     average_cost_by_ticker,
                                     total_commission_by_ticker,
                                     asset_price_by_ticker, cash,
                                     total_invested, profit)

    def __repr__(self):
        return self.__generate_output()

    def __str__(self):
        return self.__generate_output()

    @property
    def quantity_by_ticker(self):
        return self.__quantity_by_ticker

    @quantity_by_ticker.setter
    def quantity_by_ticker(self, value):
        self.__quantity_by_ticker = value

    @property
    def total_cost_by_ticker(self):
        return self.__total_cost_by_ticker

    @total_cost_by_ticker.setter
    def total_cost_by_ticker(self, value):
        self.__total_cost_by_ticker = value

    @property
    def average_cost_by_ticker(self):
        return self.__average_cost_by_ticker

    @average_cost_by_ticker.setter
    def average_cost_by_ticker(self, value):
        self.__average_cost_by_ticker = value

    @property
    def total_commission_by_ticker(self):
        return self.__total_commission_by_ticker

    @total_commission_by_ticker.setter
    def total_commission_by_ticker(self, value):
        self.__total_commission_by_ticker = value

    @property
    def asset_price_by_ticker(self):
        return self.__asset_price_by_ticker

    @asset_price_by_ticker.setter
    def asset_price_by_ticker(self, value):
        self.__asset_price_by_ticker = value

    @property
    def value_by_ticker(self):
        return self.__calculate_value_by_ticker()

    @property
    def weight_by_ticker(self):
        return self.__calculate_weight_by_ticker()

    @property
    def cash(self):
        return self.__cash

    @cash.setter
    def cash(self, value):
        self.__cash = value

    @property
    def total_invested(self):
        return self.__total_invested

    @total_invested.setter
    def total_invested(self, value):
        self.__total_invested = value

    @property
    def profit(self):
        return self.__profit

    @profit.setter
    def profit(self, value):
        self.__profit = value

    def __init_values_by_ticker(self):

        self.__total_commission_by_ticker = {}
        self.__average_cost_by_ticker = {}
        self.__total_cost_by_ticker = {}
        self.__quantity_by_ticker = {}
        self.__asset_price_by_ticker = {}

        for ticker in self.__investment.assets.keys():
            self.__total_commission_by_ticker[ticker] = 0
            self.__average_cost_by_ticker[ticker] = 0
            self.__total_cost_by_ticker[ticker] = 0
            self.__quantity_by_ticker[ticker] = 0
            self.__asset_price_by_ticker[ticker] = 0

    def __copy(self, investment_state_to_copy):
        self.__quantity_by_ticker = \
            investment_state_to_copy.quantity_by_ticker.copy()
        self.__total_cost_by_ticker = \
            investment_state_to_copy.total_cost_by_ticker.copy()
        self.__average_cost_by_ticker = \
            investment_state_to_copy.average_cost_by_ticker.copy()
        self.__total_commission_by_ticker = \
            investment_state_to_copy.total_commission_by_ticker.copy()
        self.__asset_price_by_ticker = \
            investment_state_to_copy.asset_price_by_ticker.copy()
        self.__cash = investment_state_to_copy.cash
        self.__total_invested = investment_state_to_copy.total_invested

        self.__profit = Profit(self.__investment,
                               profit_to_copy=investment_state_to_copy.profit)

    def __initialize_attributes(self, quantity_by_ticker, total_cost_by_ticker,
                                average_cost_by_ticker,
                                total_commission_by_ticker,
                                asset_price_by_ticker, cash, total_invested,
                                profit):
        if quantity_by_ticker is not None:
            for ticker, _ in quantity_by_ticker.items():
                self.__quantity_by_ticker[ticker] = quantity_by_ticker[ticker]

        if total_cost_by_ticker is not None:
            for ticker, _ in total_cost_by_ticker.items():
                self.__total_cost_by_ticker[ticker] = \
                    total_cost_by_ticker[ticker]

        if average_cost_by_ticker is not None:
            for ticker, _ in average_cost_by_ticker.items():
                self.__average_cost_by_ticker[ticker] = \
                    average_cost_by_ticker[ticker]

        if total_commission_by_ticker is not None:
            for ticker, _ in total_commission_by_ticker.items():
                self.__total_commission_by_ticker[ticker] = \
                    total_commission_by_ticker[ticker]

        if asset_price_by_ticker is not None:
            for ticker, _ in asset_price_by_ticker.items():
                self.__asset_price_by_ticker[ticker] = asset_price_by_ticker[
                    ticker]

        if cash is not None:
            self.__cash = cash

        if total_invested is not None:
            self.__total_invested = total_invested

        if profit is not None:
            self.__profit = profit

    def __calculate_value_by_ticker(self):
        value_by_ticker = {}
        for ticker, quantity in self.__quantity_by_ticker.items():
            value_by_ticker[ticker] = \
                quantity * self.__asset_price_by_ticker[ticker]

        return value_by_ticker

    def __calculate_weight_by_ticker(self):
        weight_by_ticker = {}
        value_by_ticker = self.__calculate_value_by_ticker()
        total_value = sum(value_by_ticker.values())
        for ticker, quantity in self.__quantity_by_ticker.items():
            value = quantity * self.__asset_price_by_ticker[ticker]
            weight_by_ticker[ticker] = value / total_value if total_value > 0 \
                else np.NaN

        return weight_by_ticker

    def __generate_output(self):
        output = ""
        for ticker in self.__quantity_by_ticker.keys():
            output += ticker + "\n" \
                      + "  Quantity: " \
                      + str(self.__quantity_by_ticker[ticker]) + "\n" \
                      + "  Value: " \
                      + str(self.value_by_ticker[ticker]) + "\n" \
                      + "  Weight: " \
                      + str(self.weight_by_ticker[ticker]) + "\n" \
                      + "  Total Cost: " \
                      + str(self.__total_cost_by_ticker[ticker]) + "\n" \
                      + "  Average Cost: " \
                      + str(self.__average_cost_by_ticker[ticker]) + "\n" \
                      + "  Total Commission: " \
                      + str(self.__total_commission_by_ticker[ticker]) + "\n"
        output += "Cash: " + str(self.__cash) + "\n" \
                  + "Total Invested: " + str(self.__total_invested)

        if self.__profit is not None:
            output += "\n" + "Profit: \n" + str(self.__profit) + "\n" \
                      + "-------------------------------------\n"

        return output
