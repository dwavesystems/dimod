from setuptools import setup, find_packages

from dimod import __version__, __author__, __description__

setup(
    name='dimod',
    version=__version__,
    author=__author__,
    description=__description__,
    url='https://github.com/dwavesystems/dimod',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=['decorator']
)
