from pyvest.data_reader.returns_data_reader import ReturnsDataReader
from pyvest.data_reader.factors_data_reader import FactorsDataReader

import pandas_datareader.data as web
from pandas_datareader.famafrench import get_available_datasets

# TODO: To be removed if/when pandas_datareader is updated.
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


class FamaFrenchDataReader(FactorsDataReader, ReturnsDataReader):
    def __init__(self):
        super().__init__()

        self.__define_datasets()
        self.__define_portfolios()
        self.__define_renamed_portfolios()

    def read_factors(self, start_date, end_date, factors=None):
        # Description: returns the monthly Fama-French factors from Kenneth
        # French's website.
        # Inputs:
        #   -start_date: start date of the factors timeseries
        #   -end_date: end date of the factors timeseries
        #   -factor: list of strings representing the factors that should be
        #   included in the output DataFrame. By default, returns all the
        #   available factors.
        # Output:
        #   -factors_df: a Dataframe that contains the monthly factors

        data_factors_dict = web.DataReader('F-F_Research_Data_Factors',
                                           'famafrench', start=start_date,
                                           end=end_date)

        if factors is None:
            factors = list(data_factors_dict[0].columns)

        factors_df = data_factors_dict[0][factors]

        return factors_df

    def read_returns(self, data_set, start_date, end_date, freq="monthly",
                     weighting="value", portfolios=None, decile_only=True,
                     rename_portfolios=True):

        portfolios_dict_key = self.__get_portfolios_dict_key(freq, weighting)

        portfolios_dict = web.DataReader(data_set, 'famafrench',
                                         start=start_date, end=end_date)

        all_portfolios_df = portfolios_dict[portfolios_dict_key]

        if portfolios is not None:
            portfolios_df = all_portfolios_df[portfolios]
        else:
            portfolios = self.get_portfolio_names(data_set, decile_only)
            if portfolios is not None:
                portfolios_df = all_portfolios_df[portfolios]
            else:
                portfolios_df = all_portfolios_df

        if rename_portfolios:
            portfolios_df.columns = self.__rename_portfolios(data_set)

        return portfolios_df

    def get_available_datasets(self):
        return get_available_datasets()

    def get_portfolio_names(self, data_set, decile_only=False):

        portfolios = None
        if data_set in self.__quintile_decile_data_sets and not decile_only:
            portfolios = self.__quintile_portfolios \
                         + self.__decile_portfolios
        elif data_set in self.__quintile_decile_data_sets and decile_only:
            portfolios = self.__decile_portfolios
        # elif data_set in self.__five_by_five_data_sets:
        #     portfolios = self.__five_by_five_portfolios

        return portfolios

    ################################ PRIVATE ##################################

    def __get_portfolios_dict_key(self, freq, weighting):
        portfolios_dict_key = None
        if freq == "monthly" and weighting == "value":
            portfolios_dict_key = 0
        elif freq == "monthly" and weighting == "equal":
            portfolios_dict_key = 1
        elif freq == "annual" and weighting == "value":
            portfolios_dict_key = 2
        elif freq == "annual" and weighting == "equal":
            portfolios_dict_key = 3

        return portfolios_dict_key

    def __get_default_portfolios(self, data_set):

        portfolios = None
        if data_set in self.__quintile_decile_data_sets:
            portfolios = self.__decile_portfolios
        elif data_set in self.__five_by_five_data_sets:
            portfolios = ['SMALL LoBETA', 'ME1 BETA2', 'ME1 BETA3',
                          'ME1 BETA4', 'SMALL HiBETA', 'ME2 BETA1',
                          'ME2 BETA2', 'ME2 BETA3', 'ME2 BETA4',
                          'ME2 BETA5', 'ME3 BETA1', 'ME3 BETA2',
                          'ME3 BETA3', 'ME3 BETA4', 'ME3 BETA5',
                          'ME4 BETA1', 'ME4 BETA2', 'ME4 BETA3',
                          'ME4 BETA4', 'ME4 BETA5', 'BIG LoBETA',
                          'ME5 BETA2', 'ME5 BETA3', 'ME5 BETA4',
                          'BIG HiBETA']

        return portfolios

    def __rename_portfolios(self, data_set):
        # Description: rename the columns of "portfolios_data_set" for
        # better readability in plots
        # Input:
        #   -portfolios_data_set: the name of a dataset on Kenneth
        #   French's website
        # Output: a list of portfolio names

        if data_set in self.__quintile_decile_data_sets \
                or data_set in self.__prior_data_sets:
            new_portfolio_names = ['1', '2', '3', '4', '5', '6', '7', '8',
                                   '9', '10']
        # elif data_set in self.__five_by_five_data_sets:
        else:
            new_portfolio_names = ['(' + str(i) + ',' + str(j) + ')'
                                   for i in range(1, 6) for j in range(1, 6)]

        return new_portfolio_names

    def __define_datasets(self):
        self.__quintile_decile_data_sets = ['Portfolios_Formed_on_BETA',
                                            'Portfolios_Formed_on_ME',
                                            'Portfolios_Formed_on_BE-ME']

        self.__prior_data_sets = ['10_Portfolios_Prior_12_2',
                                  '10_Portfolios_Prior_1_0',
                                  '10_Portfolios_Prior_60_13']

        self.__five_by_five_data_sets = ['25_Portfolios_ME_BETA_5x5',
                                         '25_Portfolios_5x5',
                                         '25_Portfolios_ME_Prior_12_2',
                                         '25_Portfolios_ME_Prior_1_0',
                                         '25_Portfolios_ME_Prior_60_13']

    def __define_portfolios(self):

        self.__quintile_portfolios = [
            'Lo 20', 'Qnt 2', 'Qnt 3', 'Qnt 4', 'Hi 20'
        ]

        self.__decile_portfolios = [
            'Lo 10', 'Dec 2', 'Dec 3', 'Dec 4', 'Dec 5', 'Dec 6', 'Dec 7',
            'Dec 8', 'Dec 9', 'Hi 10'
        ]

        self.__prior_portfolios = [
            'Lo PRIOR', 'PRIOR 2', 'PRIOR 3', 'PRIOR 4', 'PRIOR 5', 'PRIOR 6',
            'PRIOR 7', 'PRIOR 8', 'PRIOR 9', 'Hi PRIOR'
        ]

        # TODO: NOT GENERAL ENOUGH
        self.__five_by_five_portfolios = [
            'SMALL LoBETA', 'ME1 BETA2', 'ME1 BETA3', 'ME1 BETA4',
            'SMALL HiBETA', 'ME2 BETA1', 'ME2 BETA2', 'ME2 BETA3', 'ME2 BETA4',
            'ME2 BETA5', 'ME3 BETA1', 'ME3 BETA2', 'ME3 BETA3', 'ME3 BETA4',
            'ME3 BETA5', 'ME4 BETA1', 'ME4 BETA2', 'ME4 BETA3', 'ME4 BETA4',
            'ME4 BETA5', 'BIG LoBETA', 'ME5 BETA2', 'ME5 BETA3', 'ME5 BETA4',
            'BIG HiBETA'
        ]

    def __define_renamed_portfolios(self):

        self.renamed_portfolios_dict = {
            'Lo 20': "Q1",
            'Qnt 2': "Q2",
            'Qnt 3': "Q3",
            'Qnt 4': "Q4",
            'Hi 20': "Q5",
            'Lo 10': "1",
            'Dec 2': "2",
            'Dec 3': "3",
            'Dec 4': "4",
            'Dec 5': "5",
            'Dec 6': "6",
            'Dec 7': "7",
            'Dec 8': "8",
            'Dec 9': "9",
            'Hi 10': "10"
        }
