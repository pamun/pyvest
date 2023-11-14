import numpy as np
import statsmodels.api as sm

from pyvest.factor_model.factor_model_visualizer import FactorModelVisualizer


class FactorModel:

    def __init__(self, r_f, factors, returns, name=None):

        if self.__check_r_f(r_f):
            self.__r_f = r_f
        if self.__check_factors(factors):
            self.__factors = factors
        if self.__check_returns(returns):
            self.__returns = returns

        self.__X = self.__factors
        self.__X = sm.add_constant(self.__X)
        self.__Y = self.__returns.sub(self.__r_f, axis=0)

        self.__regression_results_dict = None
        self.__estimated_alpha_dict = None
        self.__estimated_betas_dict = None
        self.__realized_average_returns_list = None
        self.__predicted_average_returns_list = None
        self.__upper_error_bars_list = None
        self.__lower_error_bars_list = None
        self.__name = name

        self.__visualizer = None

    ##################### X ###################    
    @property
    def X(self):
        return self.__X

    @X.setter
    def X(self, value):
        self.__X = value

        ##################### Y ###################

    @property
    def Y(self):
        return self.__Y

    @Y.setter
    def Y(self, value):
        self.__Y = value

    ##################### r_f ###################    
    @property
    def r_f(self):
        return self.__r_f

    ##################### regression_results ###################    
    @property
    def regression_results(self):
        return self.__regression_results_dict

    ##################### estimated_alpha ###################    
    @property
    def estimated_alpha(self):
        return self.__estimated_alpha_dict

    ##################### estimated_betas ###################    
    @property
    def estimated_betas(self):
        return self.__estimated_betas_dict

    ##################### realized_average_returns ###################    
    @property
    def realized_average_returns(self):
        return self.__realized_average_returns_list

    ##################### predicted_average_returns ###################    
    @property
    def predicted_average_returns(self):
        return self.__predicted_average_returns_list

    ##################### upper_error_bars ###################    
    @property
    def upper_error_bars(self):
        return self.__upper_error_bars_list

    ##################### lower_error_bars ###################    
    @property
    def lower_error_bars(self):
        return self.__lower_error_bars_list

    ##################### name ###################
    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        self.__name = value

    ##################### visualizer ###################
    @property
    def visualizer(self):
        return self.__visualizer

    ########################## PUBLIC ##########################    

    def calculate_regression(self, return_results=True):
        # Description: estimates CAPM regressions for all self.___Y in the
        # self.___Y DataFrame and returns dictionaries with estimated alphas,
        # estimated betas, and associated confidence interval of every
        # self.___Y.
        # Output: dictionaries containing the estimated alpha, beta, and
        # confidence interval of all self.___Y
        self.__regression_results_dict = {}
        self.__estimated_alpha_dict = {}
        self.__estimated_betas_dict = {}

        for column_name in self.__Y:
            regression_results, estimated_alpha, estimated_betas = \
                self.__calculate_single_regression(self.__Y[column_name])
            self.__regression_results_dict[column_name] = regression_results
            self.__estimated_alpha_dict[column_name] = estimated_alpha
            self.__estimated_betas_dict[column_name] = estimated_betas

        if return_results:
            return self.__regression_results_dict, \
                   self.__estimated_alpha_dict, \
                   self.__estimated_betas_dict

    def calculate_realized_vs_predicted_average_returns(self,
                                                        return_results=True):

        if not self.__regression_results_dict:
            self.calculate_regression(return_results=False)

        rf_mean = self.__r_f.mean()

        factors_mean = self.__factors.mean()
        if len(factors_mean.shape) == 0:
            factors_mean = np.array([factors_mean])

        estimated_betas_list = list(self.__estimated_betas_dict.values())
        estimated_alpha_list = list(self.__estimated_alpha_dict.values())

        self.__realized_average_returns_list = []
        self.__predicted_average_returns_list = []
        for i in range(0, len(estimated_alpha_list)):
            betas_dot_factors_mean = np.dot(factors_mean,
                                            estimated_betas_list[i])
            realized_average_returns = estimated_alpha_list[
                                           i] + rf_mean + betas_dot_factors_mean
            predicted_average_returns = rf_mean + betas_dot_factors_mean

            self.__realized_average_returns_list.append(
                realized_average_returns)
            self.__predicted_average_returns_list.append(
                predicted_average_returns)

        if return_results:
            return self.__realized_average_returns_list, \
                   self.__predicted_average_returns_list

    def calculate_error_bars(self, confidence_level=0.95,
                             return_results=True):

        if not self.__regression_results_dict:
            self.calculate_regression(return_results=False)

        self.__upper_error_bars_list = []
        self.__lower_error_bars_list = []
        for column_name, estimated_alpha in self.__estimated_alpha_dict.items():
            conf_int_df = self.__calculate_single_conf_int(
                self.__regression_results_dict[column_name],
                confidence_level)
            upper_error_bar = conf_int_df.loc['const'][
                                  'Upper bound'] - estimated_alpha
            lower_error_bar = estimated_alpha - conf_int_df.loc['const'][
                'Lower bound']
            self.__upper_error_bars_list.append(upper_error_bar)
            self.__lower_error_bars_list.append(lower_error_bar)

        if return_results:
            return self.__lower_error_bars_list, self.__upper_error_bars_list

    def plot(self, compare_with=None, labels=None, colors=None,
             error_bars_colors=None, legend='upper left', min_return=0,
             max_return=1.5, confidence_level=0.95, sml=False, beta_min=0,
             beta_max=2):

        self.__perform_required_calculations(confidence_level)

        self.__construct_visualizer(compare_with=compare_with, labels=labels,
                                    colors=colors,
                                    error_bars_colors=error_bars_colors)

        if sml:
            self.visualizer.plot_sml(beta_min=beta_min, beta_max=beta_max,
                                     legend=legend)
        else:
            self.visualizer.plot_realized_vs_predicted_average_return(
                min_return=min_return, max_return=max_return, legend=legend)

    ########################## PRIVATE ##########################

    def __perform_required_calculations(self, confidence_level):
        self.calculate_realized_vs_predicted_average_returns(
            return_results=False)
        self.calculate_error_bars(confidence_level, return_results=False)

    def __construct_visualizer(self, compare_with=None, labels=None,
                               colors=None, error_bars_colors=None):
        factor_models = [self]
        if isinstance(compare_with, FactorModel):
            factor_models.append(compare_with)
        elif isinstance(compare_with, list):
            factor_models.extend(compare_with)

        if labels is None:
            labels_iter = iter([1, 2, 3, 4])
            labels = [fm.name if fm.name is not None else next(labels_iter)
                      for fm in factor_models]

        self.__visualizer = FactorModelVisualizer(
            factor_models, labels=labels, colors=colors,
            error_bars_colors=error_bars_colors)

    def __check_r_f(self, r_f):
        if len(r_f.shape) != 1 and (len(r_f.shape) != 2 or r_f.shape[1] != 1):
            raise ValueError("The argument of 'r_f' must be one-dimensional.")
        return True

    def __check_factors(self, factors):
        if len(factors) != len(self.__r_f):
            raise ValueError(
                "The arguments of 'factors' and 'rf' must be of the same length.")
        return True

    def __check_returns(self, returns):
        if len(returns) != len(self.__r_f):
            raise ValueError(
                "The arguments of 'returns' and 'rf' must be of the same length.")
        return True

    def __calculate_single_regression(self, y):
        linear_model = sm.OLS(y, self.__X)
        regression_results = linear_model.fit()

        estimated_parameters = regression_results.params
        estimated_alpha = estimated_parameters[0]
        estimated_betas = estimated_parameters[1:]

        return regression_results, estimated_alpha, estimated_betas

    def __calculate_single_conf_int(self, regression_results,
                                    confidence_level):
        conf_int_df = regression_results.conf_int(1 - confidence_level)
        conf_int_df.rename({0: 'Lower bound', 1: 'Upper bound'}, axis=1,
                           inplace=True)

        return conf_int_df
