from setuptools import setup

setup(
    package_dir={'pyvest': 'pyvest'},
    packages=["pyvest", "pyvest.factor_model", "pyvest.general",
               "pyvest.investment_universe", "pyvest.simulation"]
)
