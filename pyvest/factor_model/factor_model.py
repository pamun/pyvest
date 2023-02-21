import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm


#########################################################################
#                             FactorModel                               #
#########################################################################

class FactorModel:

    def __init__(self, r_f, factors, returns):

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

    ########################## PUBLIC ##########################    

    def calculate_regression(self):
        # Description: estimates CAPM regressions for all self.___Y in the self.___Y DataFrame and 
        # returns dictionaries with estimated alphas, estimated betas, and associated confidence interval of every self.___Y
        # Output: dictionaries containing the estimated alpha, beta, and confidence interval of all self.___Y
        self.__regression_results_dict = {}
        self.__estimated_alpha_dict = {}
        self.__estimated_betas_dict = {}

        for column_name in self.__Y:
            regression_results, estimated_alpha, estimated_betas = self.__calculate_single_regression(
                self.__Y[column_name])
            self.__regression_results_dict[column_name] = regression_results
            self.__estimated_alpha_dict[column_name] = estimated_alpha
            self.__estimated_betas_dict[column_name] = estimated_betas

        return self.__regression_results_dict, self.__estimated_alpha_dict, self.__estimated_betas_dict

    def calculate_realized_vs_predicted_average_returns(self):

        if not self.__regression_results_dict:
            raise ValueError("You have to calculate a regression first!")

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

        return self.__realized_average_returns_list, self.__predicted_average_returns_list

    def calculate_error_bars(self, confidence_level=0.95):

        if not self.__regression_results_dict:
            raise ValueError("You have to calculate a regression first!")

        self.__upper_error_bars_list = []
        self.__lower_error_bars_list = []
        for column_name, estimated_alpha in self.__estimated_alpha_dict.items():
            conf_int_df = self.__calculate_single_conf_int(
                self.__regression_results_dict[column_name],
                confidence_level)
            upper_errorbar = conf_int_df.loc['const'][
                                 'Upper bound'] - estimated_alpha
            lower_errorbar = estimated_alpha - conf_int_df.loc['const'][
                'Lower bound']
            self.__upper_error_bars_list.append(upper_errorbar)
            self.__lower_error_bars_list.append(lower_errorbar)

        return self.__lower_error_bars_list, self.__upper_error_bars_list

    ########################## PRIVATE ##########################

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


#########################################################################
#                       FactorModelVisualizer                           #
#########################################################################

class FactorModelVisualizer:

    def __init__(self, factor_models, labels=None, colors=None,
                 error_bars_colors=None):

        if isinstance(factor_models, FactorModel):
            self.__factor_models = [factor_models]
        else:
            self.__factor_models = factor_models

        self.__labels = labels if labels is not None else ["1", "2", "3", "4"]
        self.__colors = colors if colors is not None \
            else ['blue', 'red', 'green', 'yellow']
        self.__error_bars_colors = error_bars_colors \
            if error_bars_colors is not None else ['C0', 'C1', 'C2', 'C3']

        self.__fig = None
        self.__ax = None
        self.__fig_sml = None
        self.__ax_sml = None

    ##################### fig ###################    
    @property
    def fig(self):
        return self.__fig

    ##################### ax ###################    
    @property
    def ax(self):
        return self.__ax

    ##################### fig_sml ###################    
    @property
    def fig_sml(self):
        return self.__fig_sml

    ##################### ax_sml ###################    
    @property
    def ax_sml(self):
        return self.__ax_sml

    ########################## PUBLIC ########################## 

    def plot_realized_vs_predicted_average_return(self, min_return=0,
                                                  max_return=1.5):

        RETURN_STEP = 0.01

        # Set plot parameters
        self.__fig, self.__ax = plt.subplots(figsize=(16, 10))

        self.__ax.set_title("Realized vs. predicted average return",
                            fontsize=30)
        self.__ax.set_xlabel("Predicted average return", fontsize=30)
        self.__ax.set_ylabel("Realized average return", fontsize=30)
        self.__ax.set_xticks(np.arange(0, 2, step=0.2), fontsize=25)
        self.__ax.tick_params(axis='both', labelsize=25)

        # Plot predicted vs realized average returns
        predicted_average_return_line_array = np.arange(min_return, max_return,
                                                        RETURN_STEP)
        self.__ax.plot(predicted_average_return_line_array,
                       predicted_average_return_line_array, color='black',
                       linewidth=2)

        labels_iter = iter(self.__labels)
        colors_iter = iter(self.__colors)
        error_bars_colors_iter = iter(self.__error_bars_colors)
        for factor_model in self.__factor_models:
            label = next(labels_iter)
            color = next(colors_iter)
            error_bars_color = next(error_bars_colors_iter)

            self.__ax.errorbar(factor_model.predicted_average_returns,
                               factor_model.realized_average_returns,
                               linestyle="None",
                               marker='.',
                               markersize=15,
                               yerr=[factor_model.lower_error_bars,
                                     factor_model.upper_error_bars],
                               capsize=5,
                               color=error_bars_color,
                               markeredgecolor=color,
                               markerfacecolor=color,
                               linewidth=1,
                               label=label)

            # For loop to annotate all points
            for i in range(len(factor_model.Y.columns)):
                self.__ax.annotate(factor_model.Y.columns[i],
                                   (factor_model.predicted_average_returns[i],
                                    factor_model.realized_average_returns[
                                        i] + 0.02),
                                   fontsize=20)

        self.__ax.legend()

    def plot_sml(self, beta_min=0, beta_max=2):

        self.__check_if_one_factor()
        self.__check_factor_models_use_same_r_f()
        self.__check_factor_models_use_same_factors()

        beta_array, sml_array = self.__calculate_sml(
            self.__factor_models[0].r_f,
            self.__factor_models[0].X.drop('const', axis=1).squeeze(),
            beta_min, beta_max)

        # Set up plot parameters
        self.__fig_sml, self.__ax_sml = plt.subplots(figsize=(16, 10))

        self.__ax_sml.set_title("SML and average return against " + r"$\beta$",
                                fontsize=30)
        self.__ax_sml.set_xlabel(r"$\hat{\beta}$", fontsize=30)
        self.__ax_sml.set_ylabel("Realized average return", fontsize=30)
        self.__ax_sml.set_xticks(np.arange(0, 2, step=0.2), fontsize=25)
        self.__ax_sml.tick_params(axis='both', labelsize=25)

        # Plot the SML
        self.__ax_sml.plot(beta_array, sml_array, color='black', linewidth=2)

        labels_iter = iter(self.__labels)
        colors_iter = iter(self.__colors)
        error_bars_colors_iter = iter(self.__error_bars_colors)
        for factor_model in self.__factor_models:
            label = next(labels_iter)
            color = next(colors_iter)
            error_bars_color = next(error_bars_colors_iter)

            self.__ax_sml.errorbar(factor_model.estimated_betas.values(),
                                   factor_model.realized_average_returns,
                                   linestyle="None",
                                   marker='.',
                                   markersize=15,
                                   yerr=[factor_model.lower_error_bars,
                                         factor_model.upper_error_bars],
                                   capsize=5,
                                   color=error_bars_color,
                                   markeredgecolor=color,
                                   markerfacecolor=color,
                                   linewidth=1,
                                   label=label)

            # For loop to annotate all points
            for i in range(len(factor_model.Y.columns)):
                estimated_betas_list = \
                    list(factor_model.estimated_betas.values())[i]
                self.__ax_sml.annotate(factor_model.Y.columns[i],
                                       (estimated_betas_list,
                                        factor_model.realized_average_returns[
                                            i] + 0.02),
                                       fontsize=20)

        self.__ax_sml.legend()

    ########################## PRIVATE ##########################

    def __check_factor_models_use_same_factors(self):
        X = None
        for factor_model in self.__factor_models:
            if X is None:
                X = factor_model.X
            else:
                if not factor_model.X.equals(X):
                    raise ValueError(
                        "The factor models must use the same factors.")

    def __check_factor_models_use_same_r_f(self):
        r_f = None
        for factor_model in self.__factor_models:
            if r_f is None:
                r_f = factor_model.r_f
            else:
                if not factor_model.r_f.equals(r_f):
                    raise ValueError(
                        "The factor models must use the same r_f.")

    def __check_if_one_factor(self):
        for factor_model in self.__factor_models:
            if len(factor_model.X.columns) != 2:
                raise ValueError(
                    "One of the factor models uses more than one factor.")

    def __calculate_sml(self, r_f, mkt_rf_series, beta_min=0, beta_max=2):

        BETA_STEP = 0.01

        rf_mean = r_f.mean()
        mkt_rf_mean = mkt_rf_series.mean()

        beta_array = np.arange(beta_min, beta_max, BETA_STEP)
        sml_array = rf_mean + mkt_rf_mean * beta_array

        return beta_array, sml_array
