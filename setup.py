# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================

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

install_requires = ['numpy>=1.14.0,<2.0.0',
                    'six>=1.10.0,<2.0.0',
                    'jsonschema>=2.6.0,<3.0.0']

extras_require = {'all': ['networkx>=2.0,<3.0',
                          'pandas>=0.22.0,<0.23.0',
                          'pymongo>=3.7.0,<3.8.0'],
                  ':python_version == "2.7"': ['futures'],
                  ':python_version <= "3.3"': ['enum34>=1.1.6,<2.0.0']}

packages = ['dimod',
            'dimod.core',
            'dimod.embedding',
            'dimod.io',
            'dimod.reference',
            'dimod.reference.composites',
            'dimod.reference.samplers',
            'dimod.testing']

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    ]

python_requires = '>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*'

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
    include_package_data=True,
    classifiers=classifiers,
    python_requires=python_requires
)
