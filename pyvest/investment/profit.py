class Profit:
    def __init__(self, investment, date=None,
                 realized_capital_gain_by_ticker=None,
                 unrealized_capital_gain_by_ticker=None,
                 reinvested_dividends_by_ticker=None,
                 non_reinvested_dividends_by_ticker=None,
                 profit_to_copy=None):

        self.__investment = investment

        self.__date = date

        self.__init_values_by_tickers()

        self.__realized_capital_gain = None
        self.__unrealized_capital_gain = None
        self.__reinvested_dividends = None
        self.__non_reinvested_dividends = None

        if profit_to_copy is not None:
            self.__copy(profit_to_copy)

        self.__initialize_attributes(date, realized_capital_gain_by_ticker,
                                     unrealized_capital_gain_by_ticker,
                                     reinvested_dividends_by_ticker,
                                     non_reinvested_dividends_by_ticker)

        self.__calculate_investment_profit()

    def __repr__(self):
        return self.__generate_output()

    def __str__(self):
        return self.__generate_output()

    @property
    def date(self):
        return self.__date

    @property
    def realized_capital_gain_by_ticker(self):
        return self.__realized_capital_gain_by_ticker

    @property
    def unrealized_capital_gain_by_ticker(self):
        return self.__unrealized_capital_gain_by_ticker

    @property
    def reinvested_dividends_by_ticker(self):
        return self.__reinvested_dividends_by_ticker

    @property
    def non_reinvested_dividends_by_ticker(self):
        return self.__non_reinvested_dividends_by_ticker

    @property
    def realized_capital_gain(self):
        return self.__realized_capital_gain

    @property
    def unrealized_capital_gain(self):
        return self.__unrealized_capital_gain

    @property
    def reinvested_dividends(self):
        return self.__reinvested_dividends

    @property
    def non_reinvested_dividends(self):
        return self.__non_reinvested_dividends

    def __init_values_by_tickers(self):

        self.__realized_capital_gain_by_ticker = {}
        self.__unrealized_capital_gain_by_ticker = {}
        self.__reinvested_dividends_by_ticker = {}
        self.__non_reinvested_dividends_by_ticker = {}

        for ticker in self.__investment.assets.keys():
            self.__realized_capital_gain_by_ticker[ticker] = 0
            self.__unrealized_capital_gain_by_ticker[ticker] = 0
            self.__reinvested_dividends_by_ticker[ticker] = 0
            self.__non_reinvested_dividends_by_ticker[ticker] = 0

    def __copy(self, profit_to_copy):
        self.__date = profit_to_copy.date
        self.__realized_capital_gain_by_ticker = \
            profit_to_copy.realized_capital_gain_by_ticker.copy()
        self.__unrealized_capital_gain_by_ticker = \
            profit_to_copy.unrealized_capital_gain_by_ticker.copy()
        self.__reinvested_dividends_by_ticker = \
            profit_to_copy.reinvested_dividends_by_ticker.copy()
        self.__non_reinvested_dividends_by_ticker = \
            profit_to_copy.non_reinvested_dividends_by_ticker.copy()

    def __initialize_attributes(self, date, realized_capital_gain_by_ticker,
                                unrealized_capital_gain_by_ticker,
                                reinvested_dividends_by_ticker,
                                non_reinvested_dividends_by_ticker):

        if date is not None:
            self.__date = date

        if realized_capital_gain_by_ticker is not None:
            for ticker, value in realized_capital_gain_by_ticker.items():
                self.__realized_capital_gain_by_ticker[ticker] = value

        if unrealized_capital_gain_by_ticker is not None:
            for ticker, value in unrealized_capital_gain_by_ticker.items():
                self.__unrealized_capital_gain_by_ticker[ticker] = value

        if reinvested_dividends_by_ticker is not None:
            for ticker, value in reinvested_dividends_by_ticker.items():
                self.__reinvested_dividends_by_ticker[ticker] = value

        if non_reinvested_dividends_by_ticker is not None:
            for ticker, value in non_reinvested_dividends_by_ticker.items():
                self.__non_reinvested_dividends_by_ticker[ticker] = value

    def __calculate_investment_profit(self):

        self.__realized_capital_gain = 0
        for ticker, value in self.__realized_capital_gain_by_ticker.items():
            self.__realized_capital_gain += value

        self.__unrealized_capital_gain = 0
        for ticker, value in self.__unrealized_capital_gain_by_ticker.items():
            self.__unrealized_capital_gain += value

        self.__reinvested_dividends = 0
        for ticker, value in self.__reinvested_dividends_by_ticker.items():
            self.__reinvested_dividends += value

        self.__non_reinvested_dividends = 0
        for ticker, value in self.__non_reinvested_dividends_by_ticker.items():
            self.__non_reinvested_dividends += value

    def __generate_output(self):
        output = ""

        if self.__date is not None:
            output += "Date: " + str(self.__date)

        if self.__realized_capital_gain is not None:
            if len(output) > 0:
                output += "\n"
            output += "Realized Capital Gain: " \
                      + str(self.__realized_capital_gain)

        if self.__unrealized_capital_gain is not None:
            if len(output) > 0:
                output += "\n"
            output += "Unrealized Capital Gain: " \
                      + str(self.__unrealized_capital_gain)

        if self.__reinvested_dividends is not None:
            if len(output) > 0:
                output += "\n"
            output += "Reinvested Dividends: " \
                      + str(self.__reinvested_dividends)

        if self.__non_reinvested_dividends is not None:
            if len(output) > 0:
                output += "\n"
            output += "Non Reinvested Dividends: " \
                      + str(self.__non_reinvested_dividends)

        return output
