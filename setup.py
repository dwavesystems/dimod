from setuptools import setup

from dwave_embedding_utilities import __version__, __author__, __description__, __authoremail__

tests_require = ['dimod',
                 'dwave_networkx',
                 'networkx']

setup(
    name='dwave_embedding_utilities',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    url='https://github.com/dwavesystems/dwave_embedding_utilities',
    license='Apache 2.0',
    tests_require=tests_require,
    py_modules=['dwave_embedding_utilities']
)
