from setuptools import setup, find_packages

setup(
    name='finlib',
    version='0.0.1',
    authors=['Patrick Munroe', 'Sébastien Plante'],
    maintainer='Patrick Munroe',
    packages=['finlib'],
    package_dir={'finlib': 'finlib'},
    install_requires=['matplotlib', 'scipy', 'numpy', 'pandas', 'yfinance',
                      'statsmodels']
)
