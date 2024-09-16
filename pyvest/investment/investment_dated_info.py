class InvestmentDatedInfo:
    def __init__(self, date, ticker=None, total_return=None, weight=None,
                 value=None):

        self.__date = date
        self.__ticker = ticker
        self.__total_return = total_return
        self.__weight = weight
        self.__value = value

    def __repr__(self):
        return self.__generate_output()

    def __str__(self):
        return self.__generate_output()

    @property
    def date(self):
        return self.__date

    @property
    def ticker(self):
        return self.__ticker

    @property
    def total_return(self):
        return self.__total_return

    @property
    def weight(self):
        return self.__weight

    @property
    def value(self):
        return self.__value

    def __generate_output(self):
        output = ""

        if self.__date is not None:
            output += "Date: " + str(self.__date)

        if self.__total_return is not None:
            if len(output) > 0:
                output += "\n"
            output += "Total Return: " \
                      + str(self.__total_return)

        if self.__weight is not None:
            if len(output) > 0:
                output += "\n"
            output += "Weight: " \
                      + str(self.__weight)

        if self.__value is not None:
            if len(output) > 0:
                output += "\n"
            output += "Value: " \
                      + str(self.__value)

        return output
