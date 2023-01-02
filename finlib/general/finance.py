import numpy as np
import pandas as pd
import math

import yfinance as yf


###############################################################################
############################# GENERAL FUNCTIONS ###############################
###############################################################################

def calculate_portfolio_expected_return(weights, r_bar):
    # This function calculate the expected return of a portfolio
    # The first argument, "weights", is a column vector that contains the
    # fraction of wealth invested in the underlying assets
    # The second argument, "r_bar", is a column vector containing the expected
    # return of the underlying assets

    return float(np.dot(weights, r_bar))


def calculate_portfolio_standard_deviation(weights, sigma):
    # This function calculate the standard deviation of a portfolio
    # The first argument, "weights", is a column vector that contains the
    # fraction of wealth invested in the underlying assets
    # The second argument, "Sigma", is a variance-covariance matrix

    return math.sqrt(np.dot(weights, np.dot(sigma, weights)))


def calculate_portfolio_sharpe_ratio(x, r_bar, sigma, r_f):
    r_p = calculate_portfolio_expected_return(x, r_bar)
    sigma_p = calculate_portfolio_standard_deviation(x, sigma)
    return (r_p - r_f) / sigma_p


def standard_utility_function(x, r_bar, sigma, gamma):
    r_bar_p = calculate_portfolio_expected_return(x, r_bar)
    sigma_p = calculate_portfolio_standard_deviation(x, sigma)
    return r_bar_p - gamma * sigma_p ** 2


################################################################################################
######################## DOWNLOAD AND TRANSFORM DATA FROM YAHOO FINANCE ########################
################################################################################################

# def get_monthly_returns_series(ticker, date_start, date_end):
#     daily_raw_data_df = pdr.get_data_yahoo(ticker, start=date_start, end=date_end)  # Import daily data
#
#     daily_adj_close_seris = daily_raw_data_df["Adj Close"]
#     monthly_adj_close_series = daily_adj_close_seris.resample('M').last()  # Resample at the monthly frequency
#     monthly_returns_series = monthly_adj_close_series / monthly_adj_close_series.shift(1) - 1  # Calculate returns
#     monthly_returns_series = monthly_returns_series[1:]  # Eliminate the first observation
#
#     return monthly_returns_series

# def get_monthly_returns_df(tickers_list, date_start, date_end):
#     ticker_monthly_returns_dict = {}
#     for ticker in tickers_list:
#         monthly_returns_series = get_monthly_returns_series(ticker, date_start, date_end)
#         ticker_monthly_returns_dict[ticker] = monthly_returns_series
#     monthly_returns_df = pd.DataFrame(ticker_monthly_returns_dict)
#
#     return monthly_returns_df


def get_monthly_returns_series(ticker, start_date, end_date):
    history_df = yf.Ticker(ticker).history(start=start_date, end=end_date,
                                           auto_adjust=False)
    monthly_history_df = history_df.resample('1M').last()
    monthly_returns_series = monthly_history_df['Adj Close'].pct_change() * 100

    return monthly_returns_series


def get_monthly_returns_df(tickers_list, start_date, end_date):
    monthly_returns_df = pd.DataFrame()
    for ticker in tickers_list:
        monthly_returns_series = get_monthly_returns_series(ticker, start_date,
                                                            end_date)
        monthly_returns_df[ticker] = monthly_returns_series

    return monthly_returns_df


###############################################################################
################################# PORTFOLIO ###################################
###############################################################################

class Portfolio:

    def __init__(self, weights, r_bar, sigma, r_f=None, expense_ratios=None,
                 expense_ratio_r_f=None):

        self.__r_bar = np.array(r_bar)
        self.__sigma = np.array(sigma)
        self.__r_f = r_f
        self.__assign_weights(weights, r_f)
        self.__assign_expense_ratios(expense_ratios, expense_ratio_r_f)

    def __repr__(self):
        return str(self.__weights)

    def __str__(self):
        return str(self.__weights)

    ##################### r_bar ###################    
    @property
    def r_bar(self):
        return self.__r_bar

    @r_bar.setter
    def r_bar(self, value):
        if type(value) is float or type(value) is int:
            self.__r_bar = value
        else:
            raise TypeError(
                "The attribute 'r_bar' must be of type float or int.")

    ##################### sigma ###################    
    @property
    def sigma(self):
        return self.__sigma

    @sigma.setter
    def sigma(self, value):
        self.__sigma = value

    ##################### r_f ###################
    @property
    def r_f(self):
        return self.__r_f

    @r_f.setter
    def r_f(self, value):
        if type(value) is float or type(value) is int:
            self.__r_f = value
        else:
            raise TypeError(
                "The attribute 'r_f' must be of type float or int.")

    ##################### weights ###################
    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, value):
        self.__assign_weights(value, self.__r_f)

    ##################### expense_ratios ###################    
    @property
    def expense_ratios(self):
        return self.__expense_ratios

    @expense_ratios.setter
    def expense_ratios(self, value):
        self.__assign_expense_ratios(value, self.__expense_ratio_r_f)

    ##################### expense_ratio_r_f ###################    
    @property
    def expense_ratio_r_f(self):
        return self.__expense_ratio_r_f

    @expense_ratio_r_f.setter
    def expense_ratio_r_f(self, value):
        self.__assign_expense_ratios(self.__expense_ratios, value)

    ##################### read-only attributes ###################    
    @property
    def expected_return(self):
        effective_r_bar = self.__calculate_effective_r_bar(
            self.__augmented_r_bar, self.__expense_ratios,
            self.__expense_ratio_r_f)
        return calculate_portfolio_expected_return(self.__augmented_weights,
                                                   effective_r_bar)

    @property
    def standard_deviation(self):
        return calculate_portfolio_standard_deviation(self.__augmented_weights,
                                                      self.__augmented_sigma)

    @property
    def augmented_weights(self):
        return self.__augmented_weights

    @property
    def augmented_r_bar(self):
        return self.__augmented_r_bar

    @property
    def augmented_sigma(self):
        return self.__augmented_sigma

    ########################## PRIVATE ########################## 

    def __assign_weights(self, weights, r_f):
        if type(weights) is not list and type(weights) is not np.ndarray:
            raise TypeError("The attribute 'weights' has to be a list or a "
                            "np.array.")
        if len(weights) != len(self.__r_bar) and r_f is None:
            raise ValueError("The length of 'weights' must be equal to {}"
                             .format(len(self.__r_bar)))
        if len(weights) != len(self.__r_bar) + 1 and r_f is not None:
            raise ValueError("The length of 'weights' must be equal to {}"
                             .format(len(self.__r_bar) + 1))
        if not math.isclose(sum(weights), 1.0):
            raise ValueError("The sum of the weights must be equal to 1. "
                             "sum(weights)={}".format(sum(weights)))

        self.__weights = np.array(weights)

        if self.__r_f is not None:
            self.__augmented_r_bar = np.concatenate((self.__r_bar,
                                                     [self.__r_f]))
            self.__augmented_weights = self.__weights
        else:
            self.__augmented_r_bar = np.concatenate((self.__r_bar, [0.0]))
            self.__augmented_weights = np.concatenate((self.__weights, [0.0]))

        zeros_column = (np.zeros((len(self.__sigma), 1)))
        zeros_row = (np.zeros((1, len(self.__sigma) + 1)))
        self.__augmented_sigma = np.concatenate((np.concatenate((self.__sigma,
                                                                 zeros_column),
                                                                axis=1),
                                                 zeros_row))

        return self.__weights

    def __assign_expense_ratios(self, expense_ratios, expense_ratio_r_f):

        if expense_ratio_r_f is None:
            self.__expense_ratio_r_f = expense_ratio_r_f
        elif type(expense_ratio_r_f) is int or type(expense_ratio_r_f) \
                is float:
            self.__expense_ratio_r_f = expense_ratio_r_f
        else:
            raise TypeError("The attribute 'expense_ratio_r_f' has to be of "
                            "type 'int' or 'float'.")

        if expense_ratios is None:
            self.__expense_ratios = expense_ratios
        elif type(expense_ratios) is int or type(expense_ratios) is float:
            self.__expense_ratios = np.repeat(expense_ratios,
                                              len(self.__r_bar))
        elif type(expense_ratios) is not list and type(expense_ratios) \
                is not np.array:
            raise TypeError("The attribute 'weights' has to be of type 'int', "
                            "'float', 'list' or 'np.array'.")
        elif len(expense_ratios) != len(self.__r_bar):
            raise ValueError("The length of 'expense_ratios' must be equal to"
                             " {}".format(len(self.__r_bar)))
        else:
            self.__expense_ratios = np.array(expense_ratios)

        return self.__expense_ratios, self.__expense_ratio_r_f

    def __calculate_effective_r_bar(self, augmented_r_bar, expense_ratios,
                                    expense_ratio_r_f):

        effective_r_bar = augmented_r_bar.copy()
        if expense_ratios is not None:
            effective_r_bar[:-1] = \
                (1 + augmented_r_bar[:-1]) * (1 - expense_ratios) - 1
        if expense_ratio_r_f is not None:
            effective_r_bar[-1] = \
                (1 + augmented_r_bar[-1]) * (1 - expense_ratio_r_f) - 1

        return effective_r_bar
