# PyVest

PyVest is a Python library that provides tools for investment.

## Risk-return trade-off graph

PyVest can be used to easily create graphs of risk-return trade-off. 
Given a set of risky and/or non-risky assets, the following objects 
can be represented on a two-dimensional graph of the expected return 
vs the standard deviation:

- Feasible portfolios
- Minimum variance portfolio (MVP)
- Efficient frontier
- Tangency portfolio
- Capital allocation line (CAL)


### Example 1: No risk-free asset

Import the class InvestmentUniverse:

    from pyvest import InvestmentUniverse

Define the names of the assets:

    assets = ['KO', 'MSFT']

Define the expected returns corresponding to each asset:

    mu = [8, 14]

Define the variance-covariance matrix of the assets:

    cov = [[3**2, 0],
           [0, 6**2]]

Construct the InvestmentUniverse corresponding to those assets:

    investment_universe = InvestmentUniverse(assets, mu, cov)

Calculate the feasible portfolios:

    investment_universe.calculate_feasible_portfolios()

Calculate the MVP:

    investment_universe.calculate_mvp()

Calculate the efficient frontier:

    investment_universe.calculate_efficient_frontier()

Plot the risk-return trade-off graph of the investment universe:

    investment_universe.plot()

### Example 2: With a risk-free asset

The risky assets are defined as above:

    from pyvest import InvestmentUniverse

    assets = ['KO', 'MSFT']
    mu = [8, 14]
    cov = [[3**2, 0],
           [0, 6**2]]

A risk-free asset of 2% is added to the investment universe:

    investment_universe_with_rf = InvestmentUniverse(assets, mu, cov, r_f=2)

The feasible portfolios, the MVP and the efficient frontier are 
calculated as above:

    investment_universe_with_rf.calculate_feasible_portfolios()
    investment_universe_with_rf.calculate_mvp()
    investment_universe_with_rf.calculate_efficient_frontier()

Calculate the tangency portfolio:

    investment_universe_with_rf.calculate_tangency_portfolio()

Calculate the CAL:

    investment_universe_with_rf.calculate_cal()

Plot the risk-return trade-off graph of the investment universe with a 
risk-free asset:

    investment_universe_with_rf.plot()