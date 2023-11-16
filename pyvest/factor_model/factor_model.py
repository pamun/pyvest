import numpy as np
import statsmodels.api as sm

from pyvest.factor_model.factor_model_visualizer import FactorModelVisualizer
from pyvest.factor_model.regression import Regression


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
        self.__name = name

        self.__regressions_dict = None
        self.__construct_regressions()
        self.__realized_average_returns = None
        self.__predicted_average_returns = None

        self.__portfolios = self.__Y.columns

        self.__visualizer = None

    def __getitem__(self, key):
        regressions = None
        if self.__regressions_dict is not None:
            regressions = self.__regressions_dict[key]

        return regressions

    def __iter__(self):
        return iter(self.__regressions_dict)

    def keys(self):
        return self.__regressions_dict.keys()

    def items(self):
        return self.__regressions_dict.items()

    def values(self):
        return self.__regressions_dict.values()

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

    ##################### regressions ###################
    @property
    def regressions(self):
        return self.__regressions_dict

    ##################### regressions_calculated ###################
    @property
    def regressions_calculated(self):
        ret = True if len(self.values()) > 0 else False
        for regression in self.values():
            if regression.regression_results is None:
                ret = False

        return ret

    ##################### realized_average_returns ###################    
    @property
    def realized_average_returns(self):
        return self.__realized_average_returns

    ##################### predicted_average_returns ###################    
    @property
    def predicted_average_returns(self):
        return self.__predicted_average_returns

    ##################### name ###################
    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        self.__name = value

    ##################### portfolios ###################
    @property
    def portfolios(self):
        return list(self.__Y.columns)

    ##################### visualizer ###################
    @property
    def visualizer(self):
        return self.__visualizer

    ########################## PUBLIC ##########################    

    def calculate_regressions(self, return_results=True):
        # Description: estimates CAPM regressions for all self.__Y in the
        # self.__Y DataFrame and returns dictionaries with estimated alphas,
        # estimated betas, and associated confidence interval of every
        # self.__Y.
        # Output: dictionaries containing the estimated alpha, beta, and
        # confidence interval of all self.__Y

        for regression in self.values():
            regression.calculate_regression(return_results=False)

        if return_results:
            return self.__regressions_dict

    def calculate_realized_vs_predicted_average_returns(self,
                                                        return_results=True):
        if not self.regressions_calculated:
            self.calculate_regressions(return_results=False)

        rf_mean = self.__r_f.mean()

        factors_mean = self.__factors.mean()
        if len(factors_mean.shape) == 0:
            factors_mean = np.array([factors_mean])

        beta_list = [rr.beta for rr
                     in self.__regressions_dict.values()]
        alpha_list = [rr.alpha for rr
                      in self.__regressions_dict.values()]

        self.__realized_average_returns = {}
        self.__predicted_average_returns = {}
        for ptf, i in zip(self.portfolios, range(len(self.portfolios))):
            betas_dot_factors_mean = np.dot(factors_mean, beta_list[i])
            realized_average_returns = \
                alpha_list[i] + rf_mean + betas_dot_factors_mean
            predicted_average_returns = rf_mean + betas_dot_factors_mean

            self.__realized_average_returns[ptf] = realized_average_returns
            self.__predicted_average_returns[ptf] = predicted_average_returns

        if return_results:
            return self.__realized_average_returns, \
                   self.__predicted_average_returns

    def plot(self, compare_with=None, labels=None, colors=None,
             error_bars_colors=None, portfolios=None, legend='upper left',
             min_return=0, max_return=1.5, confidence_level=None, sml=False,
             beta_min=0, beta_max=2):

        self.__construct_visualizer(compare_with=compare_with, labels=labels,
                                    colors=colors,
                                    error_bars_colors=error_bars_colors)

        if sml:
            self.visualizer.plot_sml(beta_min=beta_min, beta_max=beta_max,
                                     legend=legend, portfolios=portfolios,
                                     confidence_level=confidence_level)
        else:
            self.visualizer.plot_realized_vs_predicted_average_return(
                min_return=min_return, max_return=max_return, legend=legend,
                portfolios=portfolios, confidence_level=confidence_level)

    ########################## PRIVATE ##########################

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

    def __construct_regressions(self):
        self.__regressions_dict = {}
        for column_name in self.__Y.columns:
            regression = Regression(self.__X, self.__Y[column_name],
                                    name=self.__name + " - " + column_name)
            self.__regressions_dict[column_name] = regression

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
