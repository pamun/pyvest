class Asset:
    def __init__(self, ticker):
        self.__ticker = ticker

    @property
    def ticker(self):
        return self.__ticker

    def __repr__(self):
        return self.__generate_output()

    def __str__(self):
        return self.__generate_output()

    def __generate_output(self):
        output = "Ticker: " + self.__ticker

        return output


class Stock(Asset):
    def __init__(self, ticker, start_date=None, end_date=None,
                 interval="1d", data_reader=None):
        super().__init__(ticker)

        self.__start_date = start_date
        self.__end_date = end_date
        self.__interval = interval
        self.__data_reader = data_reader

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
    def history(self):
        return self.__history_df

    @property
    def dividends(self):
        return self.__dividends_series

    def __read_history(self):
        self.__history_df = self.__data_reader.read_history(
            self.ticker, self.__start_date, self.__end_date,
            interval=self.__interval)

    def __extract_dividends_from_history(self):
        self.__dividends_series = self.__history_df["Dividends"][
            self.__history_df["Dividends"] > 0]

        return self.__dividends_series

    def __generate_output(self):
        output = ""

        return output
