from setuptools import setup, find_packages

from dimod import __version__

setup(
    name='dimod',
    version=__version__,
    packages=find_packages(),
    install_requires=['decorator']
)
