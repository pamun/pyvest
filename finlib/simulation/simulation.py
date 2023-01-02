import numpy as np
import matplotlib.pyplot as plt


###############################################################################
######################################### SIMULATION ##########################
###############################################################################

class Simulation:

    def __init__(self, portfolio, initial_value, periodic_contribution,
                 nb_periods):
        self.__portfolio = portfolio
        self.__initial_value = initial_value
        self.__periodic_contribution = periodic_contribution
        self.__nb_periods = nb_periods

        self.__portfolio_value_paths_list = None
        self.__portfolio_value_mean_path = None
        self.__portfolio_value_std_path = None

    def calculate_portfolio_value_paths(self, nb_paths, random_seed=1):

        np.random.seed(
            seed=random_seed)  # Fix the seed of the random number generator.
        self.__portfolio_value_paths_list = []
        # Initiate empty list to contain the paths of the simulation

        for i in range(0, nb_paths):
            r_portfolio_array = np.random.normal(
                self.__portfolio.expected_return,
                self.__portfolio.standard_deviation,
                self.__nb_periods)  # Draw realizations of the market return
            r_portfolio_array = r_portfolio_array.clip(
                min=-1)  # We consider a loss greater than 100% as a 100% loss.

            portfolio_value_path = [
                self.__initial_value]  # Initiate sample path to "initial_value"
            current_value = self.__initial_value
            for annual_return in r_portfolio_array:
                current_value = current_value * (
                        1 + annual_return) + self.__periodic_contribution
                # Cumulate return and contribution
                portfolio_value_path.append(
                    current_value)  # Append current wealth to path

            self.__portfolio_value_paths_list.append(
                portfolio_value_path)  # Append current path to simulation

        return self.__portfolio_value_paths_list

    ########################## read-only attributes ###########################
    @property
    def portfolio_value_paths(self):
        return self.__portfolio_value_paths_list

    @property
    def portfolio_value_mean_path(self):
        if self.__portfolio_value_paths_list:
            return list(np.mean(self.__portfolio_value_paths_list, axis=0))
        else:
            return None

    @property
    def portfolio_value_std_path(self):
        if self.__portfolio_value_paths_list:
            return list(
                np.std(self.__portfolio_value_paths_list, axis=0, ddof=1))
        else:
            return None

    @property
    def portfolio_final_values(self):
        if self.__portfolio_value_paths_list:
            return [portfolio_value_array[-1] for portfolio_value_array in
                    self.__portfolio_value_paths_list]
        else:
            return None


###############################################################################
################################# SimulationVisualizer ########################
###############################################################################

class SimulationVisualizer:

    def __init__(self, simulations, labels=None,
                 colors=None):

        if colors is None:
            colors = ['blue', 'red', 'green', 'yellow']
        if labels is None:
            labels = ["1", "2", "3", "4"]
        if isinstance(simulations, Simulation):
            self.__simulations = [simulations]
        else:
            self.__simulations = simulations
        self.__labels = labels
        self.__colors = colors

        self.__fig = None
        self.__ax = None
        self.__fig_twd = None
        self.__ax_twd = None

    ################################# ATTRIBUTES ##############################

    @property
    def fig(self):
        return self.__fig

    @property
    def ax(self):
        return self.__ax

    @property
    def fig_twd(self):
        return self.__fig_twd

    @property
    def ax_twd(self):
        return self.__ax_twd

    def show_portfolios_value(self, nb_paths_max=100):

        self.__fig, self.__ax = plt.subplots(figsize=(16, 10))
        self.__ax.set_title('Portfolio value')
        self.__ax.set_xlabel('Number of years')
        self.__ax.set_ylabel('Portfolio value')

        colors_iter = iter(self.__colors)
        labels_iter = iter(self.__labels)
        for simulation in self.__simulations:
            color = next(colors_iter)
            label = next(labels_iter)
            nb_paths = 0
            for portfolio_value_path in simulation.portfolio_value_paths:
                self.__ax.plot(portfolio_value_path, color=color, alpha=0.2,
                               linewidth=1)
                nb_paths += 1
                if nb_paths >= nb_paths_max:
                    break;

        colors_iter = iter(self.__colors)
        labels_iter = iter(self.__labels)
        for simulation in self.__simulations:
            color = next(colors_iter)
            label = next(labels_iter)
            self.__ax.plot(simulation.portfolio_value_mean_path, color=color,
                           linewidth=3, label=label)

        self.__ax.legend()

    def show_terminal_wealth_distribution(self):

        self.__fig_twd, self.__ax_twd = plt.subplots(figsize=(16, 10))
        self.__ax_twd.set_title('Distribution of terminal wealth')
        self.__ax_twd.set_xlabel('Terminal wealth')
        self.__ax_twd.set_ylabel('Number of paths')

        colors_iter = iter(self.__colors)
        labels_iter = iter(self.__labels)
        for simulation in self.__simulations:
            color = next(colors_iter)
            label = next(labels_iter)
            self.__ax_twd.hist(simulation.portfolio_final_values, color=color,
                               alpha=0.5, bins=100)
            self.__ax_twd.axvline(np.mean(simulation.portfolio_final_values),
                                  color=color, label=label)

        self.__ax_twd.legend()
