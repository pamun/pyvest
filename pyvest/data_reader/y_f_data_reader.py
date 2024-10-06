from datetime import timedelta

from pyvest.data_reader import ReturnsDataReader
from pyvest.data_reader.history_data_reader import HistoryDataReader

import pandas as pd
import yfinance as yf


class YFDataReader(ReturnsDataReader, HistoryDataReader):
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

    def read_history(self, ticker, start_date, end_date, interval='1d'):

        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).tz_localize(None)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).tz_localize(None)

        end_date += timedelta(days=1)

        history_df = yf.Ticker(ticker).history(start=start_date, end=end_date,
                                               interval=interval,
                                               auto_adjust=False)

        return history_df

    def __read_ticker_returns(self, ticker, start_date, end_date, freq):
        history_df = yf.Ticker(ticker).history(start=start_date, end=end_date,
                                               auto_adjust=False)
        monthly_history_df = history_df.resample(freq).last()
        monthly_returns_series = monthly_history_df[
                                     'Adj Close'].pct_change() * 100
        monthly_returns_series = monthly_returns_series[1:]

        return monthly_returns_series
