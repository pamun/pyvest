import pandas as pd
import datetime

import pyvest.investment as investment_module
from pyvest.investment.probability_table import SingleProbabilityTable, \
    ProbabilityTable, JointProbabilityTable
from pyvest.investment.simulation import ProbabilityTableSimulation
from pyvest.investment.transaction import BuyTransaction, SellTransaction, \
    DividendTransaction, ReinvestedDividendTransaction, RebalanceTransaction


class Investment:
    def __init__(self, tickers, cash, weights=None, start_date=None,
                 end_date=None, frequency=None, dividend_commission=0,
                 rebalance_commission=0, reinvest_dividends=True,
                 rebalance=False, from_historical_data=True,
                 data_reader=None, name=None):

        self.__tickers = [tickers] if isinstance(tickers, str) else tickers

        self.__cash = cash

        self.__assets = {}
        self.__asset_types = {}

        self.__weights = weights
        self.__reinvest_dividends = reinvest_dividends
        self.__rebalance = rebalance

        self.__data_reader = data_reader

        self.__name = name

        self.__unprocessed_transactions = {}

        if from_historical_data and start_date is not None \
                and end_date is not None:
            self.__initialize_frequency(frequency)
            self.__start_date = self.__preprocess_date(start_date)
            self.__end_date = self.__preprocess_date(end_date)

            self.__initialize_state_history()

            self.__create_assets_from_historical_data()
            self.__initialize_dates()
            self.__create_dividend_transactions(dividend_commission,
                                                reinvest_dividends)
            if weights is not None:
                self.rebalance(self.__start_date, rebalance_commission)
            if weights is not None and rebalance:
                self.__create_rebalance_transactions(rebalance_commission)
        else:
            pass
            # raise ValueError("from_historical_data must be True and "
            #                  "start_date and end_date must be provided.")

        self.__probability_table = None
        self.__visualizer = None

    def __repr__(self):
        return self.__generate_output()

    def __str__(self):
        return self.__generate_output()

    @property
    def name(self):
        return self.__name

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

    @property
    def start_date(self):
        return self.__start_date

    @property
    def end_date(self):
        return self.__end_date

    @property
    def probability_table(self):
        return self.__probability_table

    @property
    def visualizer(self):
        return self.__visualizer

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

    def calculate_weight(self, date):

        processed_date = pd.to_datetime(date).tz_localize(None)

        state = self.get_state(date)

        weight_by_ticker = state.weight_by_ticker

        dated_weight_by_ticker = {}
        for ticker, weight in weight_by_ticker.items():
            dated_weight = \
                investment_module.investment_dated_info.InvestmentDatedInfo(
                    processed_date, ticker=ticker,  weight=weight)
            dated_weight_by_ticker[ticker] = dated_weight

        return dated_weight_by_ticker

    def calculate_weight_range(self, start_date=None, end_date=None):

        if start_date is None:
            start_date = self.__start_date

        if end_date is None:
            end_date = self.__end_date

        processed_start_date = pd.to_datetime(start_date).tz_localize(None)
        processed_end_date = pd.to_datetime(end_date).tz_localize(None)

        dated_weights_list = []
        for date in self.__history_dates:
            processed_date = pd.to_datetime(date).tz_localize(None)
            if processed_start_date <= processed_date <= processed_end_date:
                dated_weight_by_ticker = self.calculate_weight(processed_date)
                dated_weights_list.append(dated_weight_by_ticker)

        return dated_weights_list

    def calculate_value(self, date):

        processed_date = pd.to_datetime(date).tz_localize(None)

        state = self.get_state(date)

        value_by_ticker = state.value_by_ticker

        dated_value_by_ticker = {}
        for ticker, value in value_by_ticker.items():
            dated_value = \
                investment_module.investment_dated_info.InvestmentDatedInfo(
                    processed_date, ticker=ticker,  value=value)
            dated_value_by_ticker[ticker] = dated_value

        return dated_value_by_ticker

    def calculate_value_range(self, start_date=None, end_date=None):

        if start_date is None:
            start_date = self.__start_date

        if end_date is None:
            end_date = self.__end_date

        processed_start_date = pd.to_datetime(start_date).tz_localize(None)
        processed_end_date = pd.to_datetime(end_date).tz_localize(None)

        dated_values_list = []
        for date in self.__history_dates:
            processed_date = pd.to_datetime(date).tz_localize(None)
            if processed_start_date <= processed_date <= processed_end_date:
                dated_value_by_ticker = self.calculate_value(processed_date)
                dated_values_list.append(dated_value_by_ticker)

        return dated_values_list

    def add_probability_table(self, states, probabilities, returns):
        tickers = None
        if isinstance(returns, dict):
            tickers = self.__tickers

        self.__probability_table = SingleProbabilityTable(tickers,
                                                          name=self.name)
        self.__probability_table.add_states(states, probabilities)

        if tickers is not None:
            for ticker, asset_returns in returns.items():
                self.__probability_table.add_returns(asset_returns, ticker)
        else:
            self.__probability_table.add_returns(returns)

    def simulate(self, nb_periods, against=None):

        if against is None:
            probability_table = self.__probability_table
        elif isinstance(against, Investment):
            probability_table = JointProbabilityTable(
                {self.__probability_table.name: self.__probability_table,
                 against.probability_table.name: against.probability_table})
        else:
            single_probability_tables = \
                [self.__probability_table] \
                + [x.probability_table for x in against]
            probability_table = JointProbabilityTable(
                single_probability_tables)

        simulation = ProbabilityTableSimulation(probability_table)

        return simulation.simulate(nb_periods)

    ############################## PLOT METHODS ###############################

    def plot_profit(self, compare_with=None,
                    start_date=None, end_date=None, figsize=(16, 9),
                    legend='best', show_transactions=False):

        self.__prepare_visualizer(compare_with)
        self.__visualizer.plot_profit(start_date, end_date, figsize, legend,
                                      show_transactions)

    def plot_return(self, compare_with=None,
                    start_date=None, end_date=None, figsize=(16, 9),
                    legend='best', show_transactions=False):
        self.__prepare_visualizer(compare_with)
        self.__visualizer.plot_return(start_date, end_date, figsize, legend,
                                      show_transactions)

    def plot_weight(self, compare_with=None,
                    start_date=None, end_date=None, figsize=(16, 9),
                    legend='best', show_transactions=False):
        self.__prepare_visualizer(compare_with)
        self.__visualizer.plot_weight(start_date, end_date, figsize, legend,
                                      show_transactions)

    def plot_value(self, compare_with=None,
                   start_date=None, end_date=None, figsize=(16, 9),
                   legend='best', show_transactions=False):
        self.__prepare_visualizer(compare_with)
        self.__visualizer.plot_value(start_date, end_date, figsize, legend,
                                      show_transactions)

    ################################ PRIVATE ##################################

    def __initialize_frequency(self, frequency):

        is_portfolio = False
        for ticker in self.__tickers:
            if self.__is_portfolio(ticker):
                is_portfolio = True

        if is_portfolio and (frequency is None or frequency == "1mo"):
            # Portfolios are only available at a monthly frequency.
            self.__frequency = "1mo"
        elif is_portfolio:
            raise ValueError("Fama-French portfolios are only available at a "
                             "monthly frequency.")
        elif frequency is None:
            self.__frequency = "1d"
        else:
            self.__frequency = frequency

    def __initialize_state_history(self):
        self.__transactions_history = {}
        self.__current_state = investment_module.InvestmentState(
            self, cash=self.__cash)
        self.__state_history = {self.__start_date: [self.__current_state]}

    def __preprocess_date(self, date):

        if isinstance(date, str):
            full_date = self.__preprocess_string_date(date)
        else:
            full_date = date

        preprocessed_date = pd.to_datetime(full_date).tz_localize(None)

        return preprocessed_date

    def __preprocess_string_date(self, date_str):

        full_date_str = date_str
        if self.__frequency == "1mo":
            if len(date_str) == 7 and date_str[4] == "-":
                full_date_str = date_str + "-01"
            else:
                raise ValueError("Monthly dates must be in the format YYYY-MM")

        return full_date_str

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
            if self.__is_portfolio(ticker):
                # Portfolio (FF)
                asset = investment_module.StockPortfolio(
                    ticker, start_date=self.__start_date,
                    end_date=self.__end_date, frequency=self.__frequency,
                    data_reader=self.__data_reader)
                asset_type = "portfolio"
            else:
                # Stock (yfinance)
                asset = investment_module.Stock(ticker,
                                                start_date=self.__start_date,
                                                end_date=self.__end_date,
                                                frequency=self.__frequency,
                                                data_reader=self.__data_reader)
                asset_type = "stock"

            self.__assets[ticker] = asset
            self.__asset_types[ticker] = asset_type

    def __create_dividend_transactions(self, commission, reinvest_dividends):
        for ticker, asset in self.__assets.items():
            if self.__asset_types[ticker] == "stock":
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

        last_date = previous_date
        last_date_month = previous_date_month

        if self.__frequency == "1d":
            day_after_last_date = last_date + datetime.timedelta(days=1)
            day_after_last_date_month = day_after_last_date.month
            if day_after_last_date_month != last_date_month:
                end_of_month_dates.append(last_date)
        elif self.__frequency == "1mo" or self.__frequency == "3mo":
            end_of_month_dates.append(last_date)

        return end_of_month_dates

    def __get_asset_price(self, ticker, date, column="Close"):
        asset_price = self.__assets[ticker].get_price(date)

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
            self.__assets[self.__tickers[0]].start_date)
        self.__end_date = self.__preprocess_date(
            self.__assets[self.__tickers[0]].end_date)
        self.__history_dates = self.__assets[self.__tickers[0]].dates

    def __calculate_return_on_date(self, date, percentage=True):

        profit = self.calculate_profit(date)

        total_profit = \
            profit.non_reinvested_dividends + profit.realized_capital_gain \
            + profit.reinvested_dividends + profit.unrealized_capital_gain

        total_return_value = total_profit / self.current_state.total_invested

        if percentage:
            total_return_value *= 100

        total_return = \
            investment_module.investment_dated_info.InvestmentDatedInfo(
            profit.date, total_return=total_return_value)

        return total_return

    def __generate_output(self):
        output = str(self.__current_state)

        return output

    def __is_portfolio(self, ticker):

        is_portfolio = False

        factor_tickers = ["Mkt-RF", "SMB", "HML", "RF", "Mkt"]
        portfolio_tickers = ["beta_" + str(i) for i in range(1, 11)] + \
                            ["size_" + str(i) for i in range(1, 11)] + \
                            ["value_" + str(i) for i in range(1, 11)] + \
                            ["momentum_" + str(i) for i in range(1, 11)]

        if ticker in factor_tickers or ticker in portfolio_tickers:
            is_portfolio = True

        return is_portfolio

    def __prepare_visualizer(self, compare_with):
        investment = [self]
        if isinstance(compare_with, Investment):
            investment.append(compare_with)
        elif isinstance(compare_with, list):
            investment.extend(compare_with)

        if self.__visualizer is None:
            self.__visualizer = \
                investment_module.InvestmentVisualizer(investment)
        else:
            self.__visualizer.investments = investment
