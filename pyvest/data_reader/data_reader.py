import pandas as pd

import yfinance as yf


class DataReader:
    def __init__(self):
        pass

    def read_returns(self, tickers, start_date, end_date):
        raise NotImplementedError("read_returns has not been implemented.")


class YFDataReader(DataReader):
    def __init__(self):
        super().__init__()

    def read_returns(self, tickers, start_date, end_date, freq='1M'):

        if isinstance(tickers, str):
            tickers = [tickers]

        monthly_returns_df = pd.DataFrame()
        for ticker in tickers:
            monthly_returns_df[ticker] = self.__read_ticker_returns(
                ticker, start_date, end_date, freq)

        return monthly_returns_df

    def __read_ticker_returns(self, ticker, start_date, end_date, freq):
        history_df = yf.Ticker(ticker).history(start=start_date, end=end_date,
                                               auto_adjust=False)
        monthly_history_df = history_df.resample(freq).last()
        monthly_returns_series = monthly_history_df[
                                     'Adj Close'].pct_change() * 100
        monthly_returns_series = monthly_returns_series[1:]

        return monthly_returns_series
