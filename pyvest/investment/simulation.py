import random


class Simulation:
    def __init__(self):
        pass

    def simulate(self, nb_periods):
        raise NotImplementedError("Simulation.simulate has not been "
                                  "implemented.")


class ProbabilityTableSimulation(Simulation):
    def __init__(self, probability_by_state):
        super().__init__()

        self.__probability_by_state = probability_by_state

    def simulate(self, nb_periods):
        probabilities = self.__probability_by_state.probabilities
        states = self.__probability_by_state.states

        random_states = random.choices(states, probabilities, k=nb_periods)

        if self.__probability_by_state.tickers is None:
            random_returns = []
            for state in random_states:
                random_returns.append(
                    self.__probability_by_state.get_return(state))
        else:
            random_returns = {}
            for ticker in self.__probability_by_state.tickers:
                random_returns[ticker] = []
                for state in random_states:
                    random_returns[ticker].append(
                        self.__probability_by_state.get_return(state, ticker))

        return random_states, random_returns
