class RegressionResults:
    def __init__(self, regression_results):
        self.__regression_results = regression_results

        estimated_parameters = regression_results.params
        self.__alpha = estimated_parameters[0]
        self.__beta = estimated_parameters[1:]

        self.__conf_int_df = None
        self.__lower_error_bar = None
        self.__upper_error_bar = None

        # Default confidence level
        self.__confidence_level = 0.95

    def __str__(self):
        return self.regression_results.summary().__str__()

    def __repr__(self):
        return self.regression_results.summary().__repr__()

    def _repr_html_(self):
        return self.regression_results.summary().as_html()

    def _repr_latex_(self):
        return self.regression_results.summary().as_latex()

    @property
    def regression_results(self):
        return self.__regression_results

    @property
    def alpha(self):
        return self.__alpha

    @property
    def beta(self):
        return self.__beta

    @property
    def confidence_interval(self):
        if self.__conf_int_df is None:
            self.calculate_confidence_interval(return_results=False)

        return self.__conf_int_df

    @property
    def error_bar(self):
        if self.__lower_error_bar is None or self.__upper_error_bar is None:
            self.calculate_error_bar(return_results=False)

        return self.__lower_error_bar, self.__upper_error_bar

    @property
    def lower_error_bar(self):
        if self.__lower_error_bar is None:
            self.calculate_error_bar(return_results=False)

        return self.__lower_error_bar

    @property
    def upper_error_bar(self):
        if self.__upper_error_bar is None:
            self.calculate_error_bar(return_results=False)

        return self.__upper_error_bar

    @property
    def confidence_level(self):
        return self.__confidence_level

    def calculate_confidence_interval(self, confidence_level=None,
                                      return_results=True):

        if confidence_level is not None:
            self.__confidence_level = confidence_level

        self.__conf_int_df = self.__regression_results.conf_int(
            1 - self.__confidence_level)
        self.__conf_int_df.rename({0: 'Lower bound', 1: 'Upper bound'}, axis=1,
                                  inplace=True)

        if return_results:
            return self.__conf_int_df

    def calculate_error_bar(self, confidence_level=None, return_results=True):

        conf_int_df = self.calculate_confidence_interval(confidence_level)
        self.__upper_error_bar = \
            conf_int_df.loc['const']['Upper bound'] - self.__alpha
        self.__lower_error_bar = self.__alpha - conf_int_df.loc['const'][
            'Lower bound']

        if return_results:
            return self.__lower_error_bar, self.__upper_error_bar
