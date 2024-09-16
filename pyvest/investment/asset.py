import pandas as pd

from pyvest import FamaFrenchDataReader, YFDataReader


class Asset:
    def __init__(self, ticker):
        self.__ticker = ticker

    @property
    def ticker(self):
        return self.__ticker

    @property
    def start_date(self):
        raise NotImplementedError("Asset.start_date has not been implemented.")

    @property
    def end_date(self):
        raise NotImplementedError("Asset.start_date has not been implemented.")

    @property
    def dates(self):
        raise NotImplementedError("Asset.dates has not been implemented.")

    def get_price(self, date):
        raise NotImplementedError("Asset.get_price has not been implemented.")

    def __repr__(self):
        return self.__generate_output()

    def __str__(self):
        return self.__generate_output()

    def __generate_output(self):
        output = "Ticker: " + self.__ticker

        return output


class Stock(Asset):
    def __init__(self, ticker, start_date=None, end_date=None,
                 frequency="1d", adjusted_close=False, data_reader=None):
        super().__init__(ticker)

        self.__start_date = start_date
        self.__end_date = end_date
        self.__frequency = frequency
        self.__adjusted_close = adjusted_close

        self.__data_reader = YFDataReader() if data_reader is None \
            else data_reader

        self.__history_df = None

        if self.__data_reader is not None and self.__start_date is not None \
                and self.__end_date is not None:
            self.__read_history()
            self.__extract_dividends_from_history()

    def __repr__(self):
        return super().__repr__() + "\n" + self.__generate_output()

    def __str__(self):
        return super().__str__() + "\n" + self.__generate_output()

    @property
    def start_date(self):
        return self.__history_df.index[0]

    @property
    def end_date(self):
        return self.__history_df.index[-1]

    @property
    def dates(self):
        return self.__history_df.index

    @property
    def dividends(self):
        return self.__dividends_series

    def get_price(self, date):
        column = "Adj Close" if self.__adjusted_close else "Close"
        asset_price = self.__history_df.loc[date][column]

        return asset_price

    def __read_history(self):
        self.__history_df = self.__data_reader.read_history(
            self.ticker, self.__start_date, self.__end_date,
            interval=self.__frequency)
        self.__history_df.index = \
            self.__history_df.index.map(lambda x: x.replace(tzinfo=None))

    def __extract_dividends_from_history(self):
        self.__dividends_series = self.__history_df["Dividends"][
            self.__history_df["Dividends"] > 0]

        return self.__dividends_series

    def __generate_output(self):
        output = ""

        return output


class StockPortfolio(Asset):
    def __init__(self, ticker, start_date=None, end_date=None,
                 frequency="monthly", initial_price=1.0, data_reader=None):

        super().__init__(ticker)

        self.__start_date = start_date
        self.__end_date = end_date
        self.__frequency = frequency
        self.__initial_price = initial_price

        self.__data_reader = FamaFrenchDataReader() if data_reader is None \
            else data_reader

        self.__history_df = None

        if self.__data_reader is not None and self.__start_date is not None \
                and self.__end_date is not None:
            self.__read_returns()
            self.__generate_price_history()

    def __repr__(self):
        return super().__repr__() + "\n" + self.__generate_output()

    def __str__(self):
        return super().__str__() + "\n" + self.__generate_output()

    @property
    def start_date(self):
        return self.__history_df.index[0]

    @property
    def end_date(self):
        return self.__history_df.index[-1]

    @property
    def dates(self):
        return self.__history_df.index

    def get_price(self, date):
        asset_price = self.__history_df.loc[date]["price"]

        return asset_price

    def __read_returns(self):

        if self.__is_factor(self.ticker):
            self.__read_factor()
        else:
            self.__read_portfolio()

    def __is_factor(self, ticker):

        is_factor = False

        factors_tickers = ["Mkt-RF", "SMB", "HML", "RF", "Mkt"]
        if ticker in factors_tickers:
            is_factor = True

        return is_factor

    def __read_factor(self):

        factors_df = self.__data_reader.read_factors(self.__start_date,
                                                     self.__end_date)

        if self.ticker == "Mkt":
            self.__history_df = pd.DataFrame(factors_df["Mkt-RF"]
                                             + factors_df["RF"],
                                             columns=["return"])
        else:
            column_name = self.ticker
            self.__history_df = factors_df[[column_name]]
            self.__history_df = self.__history_df.rename(
                columns={column_name: "return"})

        self.__history_df.index = self.__history_df.index.to_timestamp()

    def __read_portfolio(self):

        data_set, portfolio_index = self.__get_portfolio_name_from_ticker()
        portfolios_df = self.__data_reader.read_returns(data_set,
                                                        self.__start_date,
                                                        self.__end_date)
        column_name = portfolio_index
        self.__history_df = portfolios_df[[column_name]]
        self.__history_df = self.__history_df.rename(
            columns={column_name: "return"})
        self.__history_df.index = self.__history_df.index.to_timestamp()

    def __get_portfolio_name_from_ticker(self):

        portfolio_mapping_dict = {
            "beta": 'Portfolios_Formed_on_BETA',
            "size": 'Portfolios_Formed_on_ME',
            "value": 'Portfolios_Formed_on_BE-ME',
            "momentum": '10_Portfolios_Prior_12_2'
        }

        data_set = portfolio_mapping_dict[self.ticker.split("_")[0]]

        portfolio_index = self.ticker.split("_")[1]

        return data_set, portfolio_index

    def __generate_price_history(self):
        returns_series = self.__history_df["return"]
        # The return of the first month is omitted since the first price should
        # correspond to the last day of the first month.
        returns_series.iloc[0] = 0.0
        prices_series = (1.0 + returns_series * 0.01).cumprod()

        self.__history_df["price"] = prices_series

    def __generate_output(self):
        output = ""

        return output
