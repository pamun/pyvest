from setuptools import setup

setup(
    package_dir={'pyvest': 'pyvest'},
    packages=["pyvest", "pyvest.data_reader", "pyvest.factor_model",
              "pyvest.general", "pyvest.investment",
              "pyvest.investment_universe", "pyvest.simulation"]
)
