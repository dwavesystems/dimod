from setuptools import setup

from dimod import __version__

setup(
    name='dimod',
    version=__version__,
    packages=['dimod'],
    install_requires=['decorator']
)