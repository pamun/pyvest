import numpy as np
import pandas as pd
import math

import yfinance as yf


###############################################################################
############################# GENERAL FUNCTIONS ###############################
###############################################################################

def calculate_portfolio_expected_return(weights, mu):
    # This function calculate the expected return of a portfolio
    # The first argument, "weights", is a column vector that contains the
    # fraction of wealth invested in the underlying assets
    # The second argument, "mu", is a column vector containing the expected
    # return of the underlying assets

    return float(np.dot(weights, mu))


def calculate_portfolio_standard_deviation(weights, cov):
    # This function calculate the standard deviation of a portfolio
    # The first argument, "weights", is a column vector that contains the
    # fraction of wealth invested in the underlying assets
    # The second argument, "cov", is a variance-covariance matrix

    return math.sqrt(np.dot(weights, np.dot(cov, weights)))


def calculate_portfolio_sharpe_ratio(x, mu, cov, r_f):
    r_p = calculate_portfolio_expected_return(x, mu)
    cov_p = calculate_portfolio_standard_deviation(x, cov)
    return (r_p - r_f) / cov_p


def standard_utility_function(x, mu, cov, gamma):
    mu_p = calculate_portfolio_expected_return(x, mu)
    cov_p = calculate_portfolio_standard_deviation(x, cov)
    return mu_p - gamma * cov_p ** 2


################################################################################################
######################## DOWNLOAD AND TRANSFORM DATA FROM YAHOO FINANCE ########################
################################################################################################


def get_monthly_returns_series(ticker, start_date, end_date):
    history_df = yf.Ticker(ticker).history(start=start_date, end=end_date,
                                           auto_adjust=False)
    monthly_history_df = history_df.resample('1M').last()
    monthly_returns_series = monthly_history_df['Adj Close'].pct_change() * 100

    return monthly_returns_series


def get_monthly_returns(tickers_list, start_date, end_date):
    monthly_returns_df = pd.DataFrame()
    for ticker in tickers_list:
        monthly_returns_series = get_monthly_returns_series(ticker, start_date,
                                                            end_date)
        monthly_returns_df[ticker] = monthly_returns_series

    return monthly_returns_df
