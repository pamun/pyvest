import math
import numpy as np

from pyvest.general.general import calculate_portfolio_expected_return
from pyvest.general.general import calculate_portfolio_standard_deviation


class Portfolio:

    def __init__(self, weights, mu, cov, r_f=None, expense_ratios=None,
                 expense_ratio_r_f=None, assets=None, nb_decimal_places=4):

        self.__mu = np.array(mu)
        self.__cov = np.array(cov)
        self.__r_f = r_f
        self.__assign_weights(weights, r_f)
        self.__assign_expense_ratios(expense_ratios, expense_ratio_r_f)
        self.__assets = assets
        self.__nb_decimal_places = nb_decimal_places

        if self.__assets is not None:
            self.__assets_weights = {asset: weight for (asset, weight)
                                     in zip(self.__assets, self.__weights)}
            if r_f is not None:
                self.__assets_weights["r_f"] = self.__weights[-1]
        else:
            self.__assets_weights = list(self.__weights)

    def __repr__(self):
        return self.__generate_output()

    def __str__(self):
        return self.__generate_output()

    ##################### mu ###################
    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, value):
        self.__mu = value
        # TODO: Test whether value is an 'array'.
        # if type(value) is float or type(value) is int:
        #     self.__mu = value
        # else:
        #     raise TypeError(
        #         "The attribute 'mu' must be of type float or int.")

    ##################### cov ###################
    @property
    def cov(self):
        return self.__cov

    @cov.setter
    def cov(self, value):
        self.__cov = value

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

    ##################### nb_decimal_places ###################
    @property
    def nb_decimal_places(self):
        return self.__nb_decimal_places

    @nb_decimal_places.setter
    def nb_decimal_places(self, value):
        self.__nb_decimal_places = value

    ##################### read-only attributes ###################
    @property
    def expected_return(self):
        effective_mu = self.__calculate_effective_mu(
            self.__augmented_mu, self.__expense_ratios,
            self.__expense_ratio_r_f)
        return calculate_portfolio_expected_return(self.__augmented_weights,
                                                   effective_mu)

    @property
    def standard_deviation(self):
        return calculate_portfolio_standard_deviation(self.__augmented_weights,
                                                      self.__augmented_cov)

    @property
    def augmented_weights(self):
        return self.__augmented_weights

    @property
    def augmented_mu(self):
        return self.__augmented_mu

    @property
    def augmented_cov(self):
        return self.__augmented_cov

    ########################## PRIVATE ##########################

    def __assign_weights(self, weights, r_f):
        if type(weights) is not list and type(weights) is not np.ndarray:
            raise TypeError("The attribute 'weights' has to be a list or a "
                            "np.array.")
        if len(weights) != len(self.__mu) and r_f is None:
            raise ValueError("The length of 'weights' must be equal to {}"
                             .format(len(self.__mu)))
        if len(weights) != len(self.__mu) + 1 and r_f is not None:
            raise ValueError("The length of 'weights' must be equal to {}"
                             .format(len(self.__mu) + 1))
        if not math.isclose(sum(weights), 1.0):
            raise ValueError("The sum of the weights must be equal to 1. "
                             "sum(weights)={}".format(sum(weights)))

        self.__weights = np.array(weights)

        if self.__r_f is not None:
            self.__augmented_mu = np.concatenate((self.__mu,
                                                  [self.__r_f]))
            self.__augmented_weights = self.__weights
        else:
            self.__augmented_mu = np.concatenate((self.__mu, [0.0]))
            self.__augmented_weights = np.concatenate((self.__weights, [0.0]))

        zeros_column = (np.zeros((len(self.__cov), 1)))
        zeros_row = (np.zeros((1, len(self.__cov) + 1)))
        self.__augmented_cov = np.concatenate((np.concatenate((self.__cov,
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
                                              len(self.__mu))
        elif type(expense_ratios) is not list and type(expense_ratios) \
                is not np.array:
            raise TypeError("The attribute 'weights' has to be of type 'int', "
                            "'float', 'list' or 'np.array'.")
        elif len(expense_ratios) != len(self.__mu):
            raise ValueError("The length of 'expense_ratios' must be equal to"
                             " {}".format(len(self.__mu)))
        else:
            self.__expense_ratios = np.array(expense_ratios)

        return self.__expense_ratios, self.__expense_ratio_r_f

    def __calculate_effective_mu(self, augmented_mu, expense_ratios,
                                 expense_ratio_r_f):

        effective_mu = augmented_mu.copy()
        if expense_ratios is not None:
            effective_mu[:-1] = \
                (1 + augmented_mu[:-1]) * (1 - expense_ratios) - 1
        if expense_ratio_r_f is not None:
            effective_mu[-1] = \
                (1 + augmented_mu[-1]) * (1 - expense_ratio_r_f) - 1

        return effective_mu

    def __generate_output(self):
        if type(self.__assets_weights) is list:
            rounded_assets_weights = [round(weight, self.__nb_decimal_places)
                                      for weight in self.__assets_weights]
        else:
            rounded_assets_weights = \
                {asset: round(weight, self.__nb_decimal_places)
                 for asset, weight in self.__assets_weights.items()}

        rounded_expected_return = round(self.expected_return,
                                        self.__nb_decimal_places)

        rounded_standard_deviation = round(self.standard_deviation,
                                           self.__nb_decimal_places)

        output = "weights: {}\nexpected return: {}\nstandard deviation: " \
                 "{}".format(str(rounded_assets_weights),
                             str(rounded_expected_return),
                             str(rounded_standard_deviation))

        return output
