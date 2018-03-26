from __future__ import absolute_import

import sys
import os

from setuptools import setup

# add __version__, __author__, __authoremail__, __description__ to this namespace
_PY2 = sys.version_info.major == 2
my_loc = os.path.dirname(os.path.abspath(__file__))
os.chdir(my_loc)
if _PY2:
    execfile(os.path.join(".", "dimod", "package_info.py"))
else:
    exec(open(os.path.join(".", "dimod", "package_info.py")).read())

install_requires = ['enum34>=1.1.6,<2.0.0',
                    'numpy>=1.11.3,<2.0.0',
                    'six>=1.10.0,<2.0.0',
                    'jsonschema>=2.6.0,<3.0.0']

extras_require = {'all': ['networkx>=2.0,<3.0',
                          'pandas>=0.22.0,<0.23.0'],
                  ':python_version == "2.7"': ['futures']}

packages = ['dimod',
            'dimod.core',
            'dimod.io',
            'dimod.reference',
            'dimod.reference.composites',
            'dimod.reference.samplers',
            'dimod.testing']

setup(
    name='dimod',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    long_description=open('README.rst').read(),
    url='https://github.com/dwavesystems/dimod',
    download_url='https://github.com/dwavesystems/dimod/releases',
    license='Apache 2.0',
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True
)
