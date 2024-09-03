import pandas as pd

import pyvest.investment as investment_module
from pyvest import YFDataReader
from pyvest.investment.transaction import BuyTransaction, SellTransaction, \
    DividendTransaction, ReinvestedDividendTransaction, RebalanceTransaction


class Investment:
    def __init__(self, tickers, cash, weights=None, start_date=None,
                 end_date=None, interval="1d", dividend_commission=0,
                 rebalance_commission=0, reinvest_dividends=True,
                 rebalance=False, from_historical_data=False,
                 data_reader=None):

        self.__tickers = [tickers] if isinstance(tickers, str) else tickers

        self.__cash = cash
        self.__start_date = self.__preprocess_date(start_date)
        self.__end_date = self.__preprocess_date(end_date)
        self.__interval = interval
        self.__assets = {}

        self.__weights = weights
        self.__reinvest_dividends = reinvest_dividends
        self.__rebalance = rebalance

        if data_reader is None:
            self.__data_reader = YFDataReader()
        else:
            self.__data_reader = data_reader

        self.__unprocessed_transactions = {}
        self.__initialize_state_history()

        if from_historical_data and self.__start_date is not None \
                and self.__end_date is not None:
            self.__create_assets_from_historical_data()
            self.__initialize_dates()
            self.__create_dividend_transactions(dividend_commission,
                                                reinvest_dividends)
            if weights is not None:
                self.rebalance(self.__start_date, rebalance_commission)
            if weights is not None and rebalance:
                self.__create_rebalance_transactions(rebalance_commission)
        else:
            self.__assets = tickers



    def __repr__(self):
        return self.__generate_output()

    def __str__(self):
        return self.__generate_output()

    @property
    def current_state(self):
        return self.__current_state

    @property
    def assets(self):
        return self.__assets

    @property
    def weights(self):
        return self.__weights

    @property
    def transactions(self):
        return self.__transactions_history

    @property
    def history(self):
        return self.__state_history

    @property
    def reinvest_dividends(self):
        return self.__reinvest_dividends

    def get_most_recent_state(self, date):

        most_recent_state = None

        for state_date, states_list in self.__state_history.items():
            if state_date > date:
                break
            most_recent_state = states_list[-1]

        return most_recent_state

    def buy(self, ticker, date, quantity, price=None, commission=0):

        date = self.__preprocess_date(date)

        if price is not None:
            buy_transaction = BuyTransaction(self, self.__assets[ticker],
                                             quantity, price, commission, date)
        elif date is not None:
            price = self.__get_asset_price(ticker, date)
            buy_transaction = BuyTransaction(self, self.__assets[ticker],
                                             quantity, price, commission, date)
        else:
            raise ValueError("Either both 'quantity' and 'price' or 'date' "
                             "must be different from None.")

        self.__process_transaction(date, buy_transaction)

        return buy_transaction

    def sell(self, ticker, date, quantity, price=None, commission=0):

        date = self.__preprocess_date(date)

        if price is not None:
            sell_transaction = SellTransaction(self, self.__assets[ticker],
                                               quantity, price, commission,
                                               date)
        elif date is not None:
            price = self.__get_asset_price(ticker, date)
            sell_transaction = SellTransaction(self, self.__assets[ticker],
                                               quantity, price, commission,
                                               date)
        else:
            raise ValueError("Either both 'quantity' and 'price' or 'date' "
                             "must be different from None.")

        self.__process_transaction(date, sell_transaction)

        return sell_transaction

    def add_dividend(self, ticker, date, dividend_by_share, commission=0,
                     reinvest_dividends=True):

        date = self.__preprocess_date(date)

        asset_price = self.__get_asset_price(ticker, date)

        if reinvest_dividends:
            dividend_transaction = ReinvestedDividendTransaction(
                self, self.__assets[ticker], asset_price, commission,
                dividend_by_share, date=date)
        else:
            dividend_transaction = DividendTransaction(
                self, self.__assets[ticker], asset_price, commission,
                dividend_by_share, date=date)

        self.__process_transaction(date, dividend_transaction)

        return dividend_transaction

    def rebalance(self, date, commission=0):
        date = self.__preprocess_date(date)

        asset_price_by_ticker = {}
        for ticker in self.__tickers:
            asset_price = self.__get_asset_price(ticker, date)
            asset_price_by_ticker[ticker] = asset_price

        rebalance_transaction = RebalanceTransaction(
            self, self.__assets, asset_price_by_ticker, commission, date=date)

        self.__process_transaction(date, rebalance_transaction)

        return rebalance_transaction

    def calculate_transactions(self):

        self.__initialize_state_history()

        unprocessed_trans_dates = \
            self.__get_sorted_dates_from_map(self.__unprocessed_transactions)

        for trans_date in unprocessed_trans_dates:
            transactions_list = self.__unprocessed_transactions[trans_date]

            if trans_date not in self.__state_history:
                self.__state_history[trans_date] = []

            if trans_date not in self.__transactions_history:
                self.__transactions_history[trans_date] = []

            for transaction in transactions_list:
                new_state = transaction.update_investment_state(
                    self.__current_state)

                self.__state_history[trans_date].append(new_state)
                self.__current_state = new_state

                self.__transactions_history[trans_date].append(transaction)

        end_state = self.__calculate_end_state()
        self.__state_history[self.__end_date] = [end_state]

    def calculate_profit(self, date, asset_prices=None):

        if type(date) == str:
            date = self.__preprocess_date(date)

        if asset_prices is None:
            asset_prices = {}
            for ticker in self.__tickers:
                asset_prices[ticker] = self.__get_asset_price(ticker, date)

        state = self.__get_most_recent_state(date)

        # Update capital gains
        unrealized_capital_gain_by_ticker = {}
        for ticker in self.__tickers:
            unrealized_capital_gain = \
                state.quantity_by_ticker[ticker] \
                * (asset_prices[ticker] - state.average_cost_by_ticker[ticker])
            unrealized_capital_gain_by_ticker[ticker] = unrealized_capital_gain

        profit = investment_module.Profit(
            self,
            date=date,
            unrealized_capital_gain_by_ticker=
            unrealized_capital_gain_by_ticker,
            profit_to_copy=state.profit)

        return profit

    def calculate_profit_range(self, start_date=None, end_date=None):

        if start_date is None:
            start_date = self.__start_date

        if end_date is None:
            end_date = self.__end_date

        processed_start_date = pd.to_datetime(start_date).tz_localize(None)
        processed_end_date = pd.to_datetime(end_date).tz_localize(None)

        profits_list = []
        for date in self.__history_dates:
            processed_date = pd.to_datetime(date).tz_localize(None)
            if processed_start_date <= processed_date <= processed_end_date:
                profit = self.calculate_profit(date=processed_date)
                profits_list.append(profit)

        return profits_list

    def get_transactions(self, transaction_type=None, start_date=None,
                         end_date=None):

        processed_start_date = pd.to_datetime(start_date).tz_localize(None) \
            if start_date is not None else None
        processed_end_date = pd.to_datetime(end_date).tz_localize(None) \
            if end_date is not None else None

        transactions_sub_dict = {}
        for date, transactions_list in self.__transactions_history.items():
            if (start_date is None or date >= processed_start_date) \
                    and (end_date is None or date <= processed_end_date):
                for transaction in transactions_list:
                    if transaction_type is None \
                            or transaction.type == transaction_type:
                        if date not in transactions_sub_dict:
                            transactions_sub_dict[date] = []
                        transactions_sub_dict[date].append(transaction)

        return transactions_sub_dict

    def calculate_return_range(self, start_date=None, end_date=None,
                               percentage=True):

        if start_date is None:
            start_date = self.__start_date

        if end_date is None:
            end_date = self.__end_date

        processed_start_date = pd.to_datetime(start_date).tz_localize(None)
        processed_end_date = pd.to_datetime(end_date).tz_localize(None)

        returns_list = []
        for date in self.__history_dates:
            processed_date = pd.to_datetime(date).tz_localize(None)
            if processed_start_date <= processed_date <= processed_end_date:
                inv_return = self.__calculate_return_on_date(
                    processed_date, percentage=percentage)
                returns_list.append(inv_return)

        return returns_list

    def get_state(self, date):

        processed_date = pd.to_datetime(date).tz_localize(None)

        most_recent_state = self.__get_most_recent_state(processed_date)
        profit = self.calculate_profit(processed_date)

        asset_price_by_ticker = {}
        for ticker in self.__tickers:
            asset_price_by_ticker[ticker] = self.__get_asset_price(
                ticker, processed_date)

        state = investment_module.InvestmentState(
            self, asset_price_by_ticker=asset_price_by_ticker, profit=profit,
            investment_state_to_copy=most_recent_state)

        return state

    def __initialize_state_history(self):
        self.__transactions_history = {}
        self.__current_state = investment_module.InvestmentState(
            self, cash=self.__cash)
        self.__state_history = {self.__start_date: [self.__current_state]}

    def __preprocess_date(self, date):

        date = pd.to_datetime(date).tz_localize(None)

        return date

    def __process_transaction(self, date, transaction):

        if date in self.__unprocessed_transactions:
            self.__unprocessed_transactions[date].append(transaction)
        else:
            self.__unprocessed_transactions[date] = [transaction]

        self.calculate_transactions()

    def __get_sorted_dates_from_map(self, dates_map):
        return sorted(list(dates_map.keys()))

    def __create_assets_from_historical_data(self):
        for ticker in self.__tickers:
            asset = investment_module.Stock(ticker,
                                            start_date=self.__start_date,
                                            end_date=self.__end_date,
                                            interval=self.__interval,
                                            data_reader=self.__data_reader)
            self.__assets[ticker] = asset

    def __create_dividend_transactions(self, commission, reinvest_dividends):
        for ticker, asset in self.__assets.items():
            for date, dividend_amount in asset.dividends.items():
                self.add_dividend(ticker, date.tz_localize(None),
                                  dividend_amount,
                                  commission=commission,
                                  reinvest_dividends=reinvest_dividends)

    def __create_rebalance_transactions(self, commission):
        end_of_month_dates = self.__get_end_of_month_dates()
        for date in end_of_month_dates:
            self.rebalance(date, commission)

    def __get_end_of_month_dates(self):
        end_of_month_dates = []
        previous_date = None
        previous_date_month = None
        for date in self.__history_dates:
            current_date_month = date.month
            if previous_date_month is not None \
                    and current_date_month != previous_date_month:
                end_of_month_dates.append(previous_date)
            previous_date = date
            previous_date_month = current_date_month

        return end_of_month_dates

    def __get_asset_price(self, ticker, date, column="Close"):
        asset_price = self.__assets[ticker].history.loc[date][column]

        return asset_price

    def __get_most_recent_state(self, date):

        most_recent_state = None

        most_recent_date = None
        state_dates = self.__get_sorted_dates_from_map(self.__state_history)
        for state_date in state_dates:
            if date < state_date:
                break
            most_recent_date = state_date

        if most_recent_date is not None:
            most_recent_state = self.__state_history[most_recent_date][-1]

        return most_recent_state

    def __calculate_end_state(self):

        profit = self.calculate_profit(self.__end_date)
        most_recent_state = self.__get_most_recent_state(self.__end_date)

        end_state = investment_module.InvestmentState(
            self, profit=profit, investment_state_to_copy=most_recent_state)

        return end_state

    def __initialize_dates(self):
        self.__start_date = self.__preprocess_date(
            self.__assets[self.__tickers[0]].history.index[0])
        self.__end_date = self.__preprocess_date(
            self.__assets[self.__tickers[0]].history.index[-1])
        self.__history_dates = self.__assets[self.__tickers[0]].history.index

    def __calculate_return_on_date(self, date, percentage=True):

        # state = self.get_most_recent_state(date)

        profit = self.calculate_profit(date)

        total_profit = \
            profit.non_reinvested_dividends + profit.realized_capital_gain \
            + profit.reinvested_dividends + profit.unrealized_capital_gain

        total_return_value = total_profit / self.current_state.total_invested

        if percentage:
            total_return_value *= 100

        total_return = investment_module.investment_return.InvestmentReturn(
            profit.date, total_return_value)

        return total_return

    def __generate_output(self):
        output = str(self.__current_state)

        return output
