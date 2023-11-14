import pyvest

import matplotlib.pyplot as plt
import numpy as np


class FactorModelVisualizer:

    def __init__(self, factor_models, labels=None, colors=None,
                 error_bars_colors=None):

        if isinstance(factor_models, pyvest.FactorModel):
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
                                                  max_return=1.5,
                                                  legend='upper left'):

        RETURN_STEP = 0.01

        # Set plot parameters
        self.__fig, self.__ax = plt.subplots(figsize=(16, 10))

        self.__ax.set_title("Realized vs. predicted average return",
                            fontsize=30)
        self.__ax.set_xlabel("Predicted average return", fontsize=30)
        self.__ax.set_ylabel("Realized average return", fontsize=30)
        self.__ax.set_xticks(np.arange(0, 2, step=0.2))
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
        if type(legend) is str:
            self.__ax.legend(fontsize=15, loc=legend)

    def plot_sml(self, beta_min=0, beta_max=2, legend='upper left'):

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
        self.__ax_sml.set_xticks(np.arange(0, 2, step=0.2))
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
        if type(legend) is str:
            self.__ax_sml.legend(fontsize=15, loc=legend)

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
