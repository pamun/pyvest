import random


class Simulation:
    def __init__(self):
        pass

    def simulate(self, nb_periods):
        raise NotImplementedError("Simulation.simulate has not been "
                                  "implemented.")


class ProbabilityTableSimulation(Simulation):
    def __init__(self, probability_table):
        super().__init__()

        self.__probability_table = probability_table

    def simulate(self, nb_periods):
        probabilities = self.__probability_table.probabilities
        states = self.__probability_table.states

        random_states = random.choices(states, probabilities, k=nb_periods)

        random_returns = []
        for state in random_states:
            random_returns.append(
                self.__probability_table.get_return(state))

        # if self.__probability_table.tickers is None:
        #     random_returns = []
        #     for state in random_states:
        #         random_returns.append(
        #             self.__probability_table.get_return(state))
        # else:
        #     random_returns = {}
        #     for ticker in self.__probability_table.tickers:
        #         random_returns[ticker] = []
        #         for state in random_states:
        #             random_returns[ticker].append(
        #                 self.__probability_table.get_return(state, ticker))

        return random_states, random_returns
