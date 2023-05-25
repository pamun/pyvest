import pyvest.general.general as finance
from pyvest.investment_universe.investment_universe import InvestmentUniverse, \
    InvestmentUniverseVisualizer

import matplotlib.pyplot as plt

tickers_list = ["META", "AAPL", "AMZN"]
tickers2_list = ['MSFT', 'MCD', 'IBM']
start_date = "2015-01-01"
end_date = "2021-12-01"


# Download data

monthly_returns_df = finance.get_monthly_returns_df(tickers_list,
                                                    start_date=start_date,
                                                    end_date=end_date)

print(monthly_returns_df.head())

monthly_returns2_df = finance.get_monthly_returns_df(tickers2_list,
                                                     start_date=start_date,
                                                     end_date=end_date)

print(monthly_returns2_df.head())

# Portfolio

r_bar = monthly_returns_df.mean()
sigma = monthly_returns_df.cov()

print("r_bar={}, sigma={}".format(r_bar, sigma))

r_bar2 = monthly_returns2_df.mean()
sigma2 = monthly_returns2_df.cov()

print("r_bar2={}, sigma2={}".format(r_bar2, sigma2))

# InvestmentUniverse

investment_universe = InvestmentUniverse(tickers_list, r_bar, sigma, r_f=0.005,
                                         min_weight=-0.5,
                                         optimization_tolerance=1e-8)
investment_universe.calculate_feasible_portfolios(nb_portfolios=20000)
print("add_feasible_portfolios")
investment_universe.calculate_mvp()
print("add_mvp")
investment_universe.calculate_efficient_frontier()
print("add_efficient_frontier")
investment_universe.calculate_tangency_portfolio()
print("add_tangency_portfolio")
investment_universe.calculate_cal()
print("add_cal")

investment_universe2 = InvestmentUniverse(tickers2_list, r_bar2, sigma2,
                                          r_f=0.5, min_weight=-0.5,
                                          optimization_tolerance=1e-8)
investment_universe2.calculate_feasible_portfolios(nb_portfolios=50000)
print("add_feasible_portfolios")
investment_universe2.calculate_mvp()
print("add_mvp")
investment_universe2.calculate_efficient_frontier()
print("add_efficient_frontier")
investment_universe2.calculate_tangency_portfolio()
print("add_tangency_portfolio")
investment_universe2.calculate_cal()
print("add_cal")

# InvestmentUniverseVisualizer

my_investment_universe_visualizer = InvestmentUniverseVisualizer(
    [investment_universe2])

my_investment_universe_visualizer.feasible_portfolios_visible = True
my_investment_universe_visualizer.mvp_visible = True
my_investment_universe_visualizer.efficient_frontier_visible = True
my_investment_universe_visualizer.tangency_portfolio_visible = True
my_investment_universe_visualizer.cal_visible = True
my_investment_universe_visualizer.r_f_visible = True
my_investment_universe_visualizer.other_portfolios_visible = True

print("plot")

my_investment_universe_visualizer.plot()

plt.show()

# my_investment_universe_visualizer.fig


investment_universe.calculate_portfolio([0.2, 0.3, 0.5])

