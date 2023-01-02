from setuptools import setup, find_packages

setup(
    name='finlib',
    version='0.0.1',
    authors=['Patrick Munroe', 'Sébastien Plante'],
    maintainer='Patrick Munroe',
    packages=['finlib', 'finlib.factor_models', 'finlib.general',
              'finlib.portfolio_theory', 'finlib.simulation'],
    package_dir={'finlib': 'finlib'}
)
