from setuptools import setup

setup(
    name='pyvest',
    version='0.0.1',
    authors=['Patrick Munroe', 'Sébastien Plante'],
    maintainer='Patrick Munroe',
    packages=['pyvest'],
    package_dir={'pyvest': 'pyvest'},
    install_requires=['matplotlib', 'scipy', 'numpy', 'pandas', 'yfinance',
                      'statsmodels']
)
