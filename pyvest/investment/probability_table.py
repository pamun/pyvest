class ProbabilityTable:
    def __init__(self, tickers=None):
        self.__tickers = [tickers] if isinstance(tickers, str) else tickers

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
    def states(self):
        return self.__states

    @property
    def probabilities(self):
        return self.__probabilities

    @property
    def returns(self):
        return self.__returns

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
