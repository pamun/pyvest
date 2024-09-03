class InvestmentReturn:
    def __init__(self, date, total_return):

        self.__date = date
        self.__total_return = total_return

    def __repr__(self):
        return self.__generate_output()

    def __str__(self):
        return self.__generate_output()

    @property
    def date(self):
        return self.__date

    @property
    def total_return(self):
        return self.__total_return

    def __generate_output(self):
        output = ""

        if self.__date is not None:
            output += "Date: " + str(self.__date)

        if self.__total_return is not None:
            if len(output) > 0:
                output += "\n"
            output += "Total Return: " \
                      + str(self.__total_return)

        return output