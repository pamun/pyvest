class ProbabilityTable:
    @property
    def states(self):
        raise NotImplementedError("ProbabilityTable.states has not "
                                  "been implemented.")

    @property
    def probabilities(self):
        raise NotImplementedError("ProbabilityTable.probabilities has not "
                                  "been implemented.")

    @property
    def probability_by_state(self):
        raise NotImplementedError("ProbabilityTable.probability_by_state has not "
                                  "been implemented.")

    def get_probability(self, state):
        raise NotImplementedError("ProbabilityTable.get_probability has not "
                                  "been implemented.")

    def get_return(self, state, ticker=None):
        raise NotImplementedError("ProbabilityTable.get_return has not "
                                  "been implemented.")


class SingleProbabilityTable(ProbabilityTable):
    def __init__(self, tickers=None, name=None):

        self.__tickers = [tickers] if isinstance(tickers, str) else tickers
        self.__name = name

        self.__probability_by_state = {}
        self.__states = []
        self.__probabilities = []

        self.__returns = []

        self.__returns_by_state = {}
        self.__returns_by_ticker_by_state = {}

    @property
    def tickers(self):
        return self.__tickers

    @property
    def name(self):
        return self.__name

    @property
    def states(self):
        return self.__states

    @property
    def probabilities(self):
        return self.__probabilities

    @property
    def probability_by_state(self):
        return self.__probability_by_state

    @property
    def returns(self):
        return self.__returns

    @property
    def returns_by_state(self):
        return self.__returns_by_state

    @property
    def returns_by_ticker_by_state(self):
        return self.__returns_by_ticker_by_state

    def add_states(self, states, probabilities):

        if sum(probabilities) != 1.0:
            raise ValueError("The probabilities should sum up to 1.0.")

        if len(probabilities) != len(states):
            raise ValueError("The number of probabilities must be equal to "
                             "the number of states.")

        for state, probability in zip(states, probabilities):
            self.__probability_by_state[state] = probability
            self.__states.append(state)
            self.__probabilities.append(probability)

    def add_returns(self, returns, ticker=None):

        if len(returns) != len(self.__states):
            raise ValueError("The number of returns must be equal to the "
                             "number of states.")

        for state, state_return in zip(self.__states, returns):
            if ticker is None:
                self.__returns.append(state_return)
                self.__returns_by_state[state] = state_return
            else:
                if state not in self.__returns_by_ticker_by_state:
                    self.__returns_by_ticker_by_state[state] = {}
                self.__returns_by_ticker_by_state[state][ticker] = state_return

    def get_probability(self, state):
        return self.__probability_by_state[state]

    def get_return(self, state, ticker=None):

        if ticker is None:
            result_return = self.__returns_by_state[state]
        else:
            result_return = self.__returns_by_ticker_by_state[state][ticker]

        return result_return


class JointProbabilityTable(ProbabilityTable):
    def __init__(self, probability_table_by_name):

        self.__probability_table_by_name = probability_table_by_name

        if self.__check_probability_table_compatibiity():
            first_prob_table = next(iter(probability_table_by_name.values()))
            self.__probability_by_state = first_prob_table.probability_by_state
            self.__states = first_prob_table.states
            self.__probabilities = first_prob_table.probabilities
        else:
            raise ValueError("The probability tables must have all the same "
                             "states and the same probabilities")

        self.__returns_by_state_by_table = {}
        self.__returns_by_ticker_by_state_by_table = {}

        for name, probability_table in probability_table_by_name.items():
            self.__returns_by_state_by_table[name] \
                = probability_table.returns_by_state
            self.__returns_by_ticker_by_state_by_table[name] \
                = probability_table.returns_by_ticker_by_state

    @property
    def states(self):
        return self.__states

    @property
    def probabilities(self):
        return self.__probabilities

    @property
    def probability_by_state(self):
        return self.__probability_by_state

    def get_probability(self, state):
        return self.__probability_by_state[state]

    def get_return(self, state, ticker=None):

        result_return = {}

        for table_name, probability_table \
                in self.__probability_table_by_name.items():
            if ticker is None:
                result_return[table_name] = \
                    self.__returns_by_state_by_table[table_name][state]
            else:
                result_return[table_name] = \
                    self.__returns_by_ticker_by_state_by_table[table_name][
                        state][ticker]

        return result_return

    def __check_probability_table_compatibiity(self):

        # TODO: To be implemented

        return True
