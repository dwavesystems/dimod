from setuptools import setup, find_packages

from dimod import __version__, __author__, __description__, __authoremail__

install_requires = ['decorator>=4.1.0']
extras_require = {'all': ['numpy']}

packages = ['dimod',
            'dimod.responses',
            'dimod.composites',
            'dimod.samplers']

setup(
    name='dimod',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    url='https://github.com/dwavesystems/dimod',
    download_url='https://github.com/dwavesys/dimod/archive/0.1.1.tar.gz',
    license='Apache 2.0',
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
)
