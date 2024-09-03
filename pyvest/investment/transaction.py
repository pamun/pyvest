import pyvest.investment as investment_module
from pyvest.investment.profit import Profit


class Transaction:
    def __init__(self, type, investment, asset, quantity=None,
                 asset_price=None, commission=None, date=None):
        self.__type = type

        self.__asset = asset

        if isinstance(asset, dict):
            self.__ticker = list(asset.keys())
        else:
            self.__ticker = self.__asset.ticker

        self.__investment = investment

        self._quantity = quantity
        self._asset_price = asset_price
        self._commission = commission
        self._date = date

    def __repr__(self):
        return self.__generate_output()

    def __str__(self):
        return self.__generate_output()

    @property
    def type(self):
        return self.__type

    @property
    def investment(self):
        return self.__investment

    @property
    def quantity(self):
        return self._quantity

    @property
    def asset_price(self):
        return self._asset_price

    @property
    def commission(self):
        return self._commission

    @property
    def date(self):
        return self._date

    @property
    def ticker(self):
        return self.__ticker

    def update_investment_state(self, investment_state_old):
        raise NotImplementedError("Transaction.update_investment_state has not"
                                  " been implemented.")

    def _create_profit(self, ticker, realized_capital_gain=None,
                       unrealized_capital_gain=None,
                       reinvested_dividends=None,
                       non_reinvested_dividends=None,
                       profit_to_copy=None):

        realized_capital_gain_by_ticker = \
            {ticker: realized_capital_gain} \
                if realized_capital_gain is not None else None

        unrealized_capital_gain_by_ticker = \
            {ticker: unrealized_capital_gain} \
                if unrealized_capital_gain is not None else None

        reinvested_dividends_by_ticker = \
            {ticker: reinvested_dividends} \
                if reinvested_dividends is not None else None

        non_reinvested_dividends_by_ticker = \
            {ticker: non_reinvested_dividends} \
                if non_reinvested_dividends is not None else None

        profit = Profit(self.__investment, date=self.date,
                        realized_capital_gain_by_ticker=
                        realized_capital_gain_by_ticker,
                        unrealized_capital_gain_by_ticker=
                        unrealized_capital_gain_by_ticker,
                        reinvested_dividends_by_ticker=
                        reinvested_dividends_by_ticker,
                        non_reinvested_dividends_by_ticker=
                        non_reinvested_dividends_by_ticker,
                        profit_to_copy=profit_to_copy)

        return profit

    def _create_investment_state(self, ticker, quantity=None, total_cost=None,
                                 average_cost=None, total_commission=None,
                                 asset_price=None, cash=None,
                                 total_invested=None, profit=None,
                                 investment_state_to_copy=None):
        quantity_by_ticker = {ticker: quantity} if quantity is not None \
            else None
        total_cost_by_ticker = {ticker: total_cost} \
            if total_cost is not None else None
        average_cost_by_ticker = {ticker: average_cost} \
            if average_cost is not None else None
        total_commission_by_ticker = {ticker: total_commission} \
            if total_commission is not None else None
        asset_price_by_ticker = {ticker: asset_price} \
            if asset_price is not None else None

        investment_state = investment_module.InvestmentState(
            self.__investment,
            quantity_by_ticker=quantity_by_ticker,
            total_cost_by_ticker=total_cost_by_ticker,
            average_cost_by_ticker=average_cost_by_ticker,
            total_commission_by_ticker=total_commission_by_ticker,
            asset_price_by_ticker=asset_price_by_ticker, cash=cash,
            total_invested=total_invested, profit=profit,
            investment_state_to_copy=investment_state_to_copy)

        return investment_state

    def __generate_output(self):
        asset_str = str(self.__asset)
        asset_str = asset_str.replace("\n", "\n  ")

        output = "Transaction type: " + self.__type + "\n" \
                 + "Asset: \n  " + asset_str + "\n" \
                 + "Quantity: " + str(self._quantity) + "\n" \
                 + "Price: " + str(self._asset_price) + "\n" \
                 + "Commission: " + str(self._commission)

        if self._date is not None:
            output += "\n" + "Date: " + str(self._date)

        return output


class BuyTransaction(Transaction):
    def __init__(self, investment, asset, quantity, asset_price, commission,
                 date=None):
        super().__init__("BUY", investment, asset, quantity, asset_price,
                         commission, date)

    def update_investment_state(self, investment_state_old):
        # Single asset transaction
        ticker = self.ticker

        total_cost_new = \
            investment_state_old.total_cost_by_ticker[ticker] \
            + self.quantity * self.asset_price
        total_commission_new = \
            investment_state_old.total_commission_by_ticker[ticker] \
            + self.commission
        quantity_new = investment_state_old.quantity_by_ticker[ticker] \
                       + self.quantity
        average_cost_new = total_cost_new / quantity_new

        # TODO: CHECK average_cost_new
        unrealized_capital_gain_new = \
            quantity_new * (self.asset_price - average_cost_new)

        cash_new = investment_state_old.cash - (
                self.quantity * self.asset_price + self.commission)
        additional_investment = self.quantity * self.asset_price \
                                + self.commission
        total_invested_new = \
            investment_state_old.total_invested + additional_investment

        profit = self._create_profit(
            ticker, unrealized_capital_gain=unrealized_capital_gain_new,
            profit_to_copy=investment_state_old.profit)

        investment_state = self._create_investment_state(
            ticker,
            quantity=quantity_new, total_cost=total_cost_new,
            average_cost=average_cost_new,
            total_commission=total_commission_new,
            asset_price=self.asset_price, cash=cash_new,
            total_invested=total_invested_new, profit=profit,
            investment_state_to_copy=investment_state_old)

        return investment_state


class SellTransaction(Transaction):
    def __init__(self, investment, asset, quantity, asset_price, commission,
                 date=None):
        super().__init__("SELL", investment, asset, quantity, asset_price,
                         commission, date)

    def update_investment_state(self, investment_state_old):
        # Single asset transaction
        ticker = self.ticker

        total_cost_new = \
            investment_state_old.total_cost_by_ticker[ticker] \
            - self.quantity \
            * investment_state_old.average_cost_by_ticker[ticker]
        total_commission_new = \
            investment_state_old.total_commission_by_ticker[ticker] \
            + self.commission
        quantity_new = investment_state_old.quantity_by_ticker[ticker] \
                       - self.quantity

        unrealized_capital_gain_new = \
            quantity_new * (self.asset_price
                            - investment_state_old.average_cost_by_ticker[
                                ticker])

        add_realized_capital_gain = \
            self.quantity * (self.asset_price -
                             investment_state_old.average_cost_by_ticker[
                                 ticker])
        realized_capital_gain_new = \
            investment_state_old.profit.realized_capital_gain_by_ticker[
                ticker] + add_realized_capital_gain

        cash_new = investment_state_old.cash + (
                self.quantity * self.asset_price - self.commission)

        profit = self._create_profit(
            ticker, realized_capital_gain=realized_capital_gain_new,
            unrealized_capital_gain=unrealized_capital_gain_new,
            profit_to_copy=investment_state_old.profit)

        investment_state = self._create_investment_state(
            ticker,
            quantity=quantity_new, total_cost=total_cost_new,
            total_commission=total_commission_new,
            asset_price=self.asset_price, cash=cash_new, profit=profit,
            investment_state_to_copy=investment_state_old)

        return investment_state


class DividendTransaction(Transaction):
    def __init__(self, investment, asset, asset_price, commission,
                 dividend_by_share, quantity=None, date=None):
        super().__init__("DIVIDEND", investment, asset, quantity, asset_price,
                         commission, date)

        self.__dividend_by_share = dividend_by_share

    def update_investment_state(self, investment_state_old):
        # Single asset transaction
        ticker = self.ticker

        self._quantity = investment_state_old.quantity_by_ticker[ticker]

        total_commission_new = \
            investment_state_old.total_commission_by_ticker[ticker] \
            + self.commission
        non_reinvested_dividends_new = \
            investment_state_old.profit.non_reinvested_dividends_by_ticker[
                ticker] + self._quantity * self.__dividend_by_share

        unrealized_capital_gain_new = \
            investment_state_old.quantity_by_ticker[ticker] \
            * (self.asset_price - investment_state_old.average_cost_by_ticker[
                ticker])

        cash_new = investment_state_old.cash + (
                self._quantity * self.__dividend_by_share - self.commission)

        profit = self._create_profit(
            ticker, unrealized_capital_gain=unrealized_capital_gain_new,
            non_reinvested_dividends=non_reinvested_dividends_new,
            profit_to_copy=investment_state_old.profit)

        investment_state = self._create_investment_state(
            ticker,
            total_commission=total_commission_new,
            asset_price=self.asset_price, cash=cash_new,
            profit=profit, investment_state_to_copy=investment_state_old)

        return investment_state


class ReinvestedDividendTransaction(Transaction):
    def __init__(self, investment, asset, asset_price, commission,
                 dividend_by_share, quantity=None, date=None):
        super().__init__("REINVESTED_DIVIDEND", investment, asset, quantity,
                         asset_price, commission, date)

        self.__dividend_by_share = dividend_by_share

    def update_investment_state(self, investment_state_old):
        # Single asset transaction
        ticker = self.ticker

        self._quantity = investment_state_old.quantity_by_ticker[ticker]

        total_commission_new = \
            investment_state_old.total_commission_by_ticker[ticker] \
            + self.commission
        additional_reinvested_dividends = \
            self._quantity * self.__dividend_by_share
        reinvested_dividends_new = \
            investment_state_old.profit.reinvested_dividends_by_ticker[
                ticker] + additional_reinvested_dividends

        total_cost_new = \
            investment_state_old.total_cost_by_ticker[ticker] \
            + additional_reinvested_dividends

        additional_quantity = \
            additional_reinvested_dividends / self.asset_price
        quantity_new = \
            investment_state_old.quantity_by_ticker[ticker] \
            + additional_quantity

        average_cost_new = total_cost_new / quantity_new if quantity_new > 0 \
            else 0

        unrealized_capital_gain_new = \
            quantity_new * (self.asset_price
                            - investment_state_old.average_cost_by_ticker[
                                ticker])

        additional_investment = \
            additional_reinvested_dividends + self.commission
        total_invested_new = \
            investment_state_old.total_invested + additional_investment

        profit = self._create_profit(
            ticker, unrealized_capital_gain=unrealized_capital_gain_new,
            reinvested_dividends=reinvested_dividends_new,
            profit_to_copy=investment_state_old.profit)

        investment_state = self._create_investment_state(
            ticker,
            quantity=quantity_new, total_cost=total_cost_new,
            average_cost=average_cost_new,
            total_commission=total_commission_new,
            asset_price=self.asset_price, total_invested=total_invested_new,
            profit=profit, investment_state_to_copy=investment_state_old)

        return investment_state


class RebalanceTransaction(Transaction):
    def __init__(self, investment, assets, asset_price, commission,
                 quantity=None, date=None):
        super().__init__("REBALANCE", investment, assets, quantity,
                         asset_price, commission, date)

        self.__value_by_ticker = None
        self.__rebalanced_value_by_ticker = None
        self.__difference_value_by_ticker = None
        self.__rebalanced_quantity_by_ticker = None
        self.__difference_quantity_by_ticker = None
        self.__total_cost_by_ticker = None
        self.__average_cost_by_ticker = None
        self.__total_commission_by_ticker = None
        self.__total_invested = None

        self.__realized_capital_gain_by_ticker = None
        self.__unrealized_capital_gain_by_ticker = None

    def update_investment_state(self, investment_state_old):

        self.__calculate_value_by_ticker(investment_state_old)

        self.__calculate_rebalanced_value_by_ticker(investment_state_old)
        self.__calculate_realized_capital_gain_by_ticker(investment_state_old)
        self.__calculate_cost_by_ticker(investment_state_old)
        self.__calculate_unrealized_capital_gain_by_ticker()
        self.__calculate_total_commission_by_ticker(investment_state_old)
        self.__calculate_total_invested_by_ticker(investment_state_old)

        cash_new = 0

        profit = Profit(self.investment, date=self.date,
                        realized_capital_gain_by_ticker=
                        self.__realized_capital_gain_by_ticker,
                        unrealized_capital_gain_by_ticker=
                        self.__unrealized_capital_gain_by_ticker,
                        profit_to_copy=investment_state_old.profit)

        investment_state = investment_module.InvestmentState(
            self.investment,
            quantity_by_ticker=self.__rebalanced_quantity_by_ticker,
            total_cost_by_ticker=self.__total_cost_by_ticker,
            average_cost_by_ticker=self.__average_cost_by_ticker,
            total_commission_by_ticker=self.__total_commission_by_ticker,
            asset_price_by_ticker=self.asset_price, cash=cash_new,
            total_invested=self.__total_invested,
            profit=profit,
            investment_state_to_copy=investment_state_old)

        return investment_state

    def __calculate_value_by_ticker(self, investment_state):
        self.__value_by_ticker = {}
        for ticker in self.ticker:
            quantity = investment_state.quantity_by_ticker[ticker]
            asset_price = self.asset_price[ticker]
            self.__value_by_ticker[ticker] = quantity * asset_price

    def __calculate_rebalanced_value_by_ticker(self, investment_state):

        total_value = sum(self.__value_by_ticker.values()) \
                      + investment_state.cash

        self.__rebalanced_value_by_ticker = {}
        self.__difference_value_by_ticker = {}
        self.__rebalanced_quantity_by_ticker = {}
        self.__difference_quantity_by_ticker = {}
        for ticker in self.ticker:
            weight = self.investment.weights[ticker]
            asset_price = self.asset_price[ticker]

            self.__rebalanced_value_by_ticker[ticker] = total_value * weight
            self.__rebalanced_quantity_by_ticker[ticker] = \
                self.__rebalanced_value_by_ticker[ticker] / asset_price
            self.__difference_value_by_ticker[ticker] = \
                self.__rebalanced_value_by_ticker[ticker] \
                - self.__value_by_ticker[ticker]
            self.__difference_quantity_by_ticker[ticker] = \
                self.__difference_value_by_ticker[ticker] / asset_price

    def __calculate_realized_capital_gain_by_ticker(self, investment_state):

        self.__realized_capital_gain_by_ticker = {}
        for ticker, diff_value in self.__difference_value_by_ticker.items():
            if diff_value > 0:
                # Buy
                self.__realized_capital_gain_by_ticker[ticker] = \
                    investment_state.profit.realized_capital_gain_by_ticker[ticker]
            else:
                # Sell
                self.__realized_capital_gain_by_ticker[ticker] = \
                    investment_state.profit.realized_capital_gain_by_ticker[ticker] \
                    - diff_value

    def __calculate_cost_by_ticker(self, investment_state):
        self.__total_cost_by_ticker = {}
        self.__average_cost_by_ticker = {}
        for ticker, diff_value in self.__difference_value_by_ticker.items():
            if diff_value > 0:
                # Buy
                self.__total_cost_by_ticker[ticker] = \
                    investment_state.total_cost_by_ticker[ticker] + diff_value
            else:
                # Sell
                self.__total_cost_by_ticker[ticker] = \
                    investment_state.total_cost_by_ticker[ticker] \
                    - self.__difference_quantity_by_ticker[ticker] \
                    * investment_state.average_cost_by_ticker[ticker]

            self.__average_cost_by_ticker[ticker] = \
                self.__total_cost_by_ticker[ticker] \
                / self.__rebalanced_quantity_by_ticker[ticker]

    def __calculate_unrealized_capital_gain_by_ticker(self):

        self.__unrealized_capital_gain_by_ticker = {}
        for ticker, diff_value in self.__difference_value_by_ticker.items():
            quantity = self.__rebalanced_quantity_by_ticker[ticker]
            asset_price = self.asset_price[ticker]

            self.__unrealized_capital_gain_by_ticker[ticker] = \
                quantity * (asset_price
                            - self.__average_cost_by_ticker[ticker])

    def __calculate_total_commission_by_ticker(self, investment_state):

        self.__total_commission_by_ticker = {}
        for ticker, diff_value in self.__difference_value_by_ticker.items():
            self.__total_commission_by_ticker[ticker] = \
                investment_state.total_commission_by_ticker[
                    ticker] + self.commission

    def __calculate_total_invested_by_ticker(self, investment_state):

        self.__total_invested = investment_state.total_invested
        for ticker, diff_value in self.__difference_value_by_ticker.items():
            asset_price = self.asset_price[ticker]
            if diff_value > 0:
                # Buy
                additional_investment = \
                    self.__difference_quantity_by_ticker[ticker] \
                    * asset_price + self.commission
                self.__total_invested += additional_investment
