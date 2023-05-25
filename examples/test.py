import pyvest.general.general as finance
from pyvest.investment_universe.investment_universe import InvestmentUniverse

tickers_list = ["META", "AAPL", "AMZN"]
start_date = "2015-01-01"
end_date = "2021-12-01"


# Download data

monthly_returns_df = finance.get_monthly_returns_df(tickers_list,
                                                    start_date=start_date,
                                                    end_date=end_date)

# Portfolio

r_bar = monthly_returns_df.mean()
sigma = monthly_returns_df.cov()

print("r_bar={}, sigma={}".format(r_bar, sigma))

investment_universe = InvestmentUniverse(tickers_list, r_bar, sigma, r_f=0.005,
                                         min_weight=-0.5,
                                         optimization_tolerance=1e-8)

investment_universe.calculate_portfolio([0.4, 0.1, 0.5], name="my_portfolio")

print(investment_universe.other_portfolios["my_portfolio"])

print(investment_universe.calculate_portfolio([0.2, 0.3, 0.5]))


investment_universe.calculate_mvp()

print(investment_universe.mvp)
