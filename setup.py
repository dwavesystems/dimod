from __future__ import absolute_import

import sys
from setuptools import setup

_PY2 = sys.version_info.major == 2

# add __version__, __author__, __authoremail__, __description__ to this namespace
# equivalent to:
if _PY2:
    execfile("./dimod/package_info.py")
else:
    exec(open("./dimod/package_info.py").read())

install_requires = ['enum34>=1.1.6']

extras_require = {'docs': ['sphinx', 'sphinx_rtd_theme', 'recommonmark'],
                  'all': ['numpy>=1.14.0', 'pandas>=0.22.0', 'networkx>=2.0']}

packages = ['dimod',
            'dimod.response',
            'dimod.composites',
            'dimod.samplers',
            'dimod.binary_quadratic_model']

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
    extras_require=extras_require
)
