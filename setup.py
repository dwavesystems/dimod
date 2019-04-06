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
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError

# add __version__, __author__, __authoremail__, __description__ to this namespace
_PY2 = sys.version_info.major == 2
my_loc = os.path.dirname(os.path.abspath(__file__))
os.chdir(my_loc)
if _PY2:
    execfile(os.path.join(".", "dimod", "package_info.py"))
else:
    exec(open(os.path.join(".", "dimod", "package_info.py")).read())

install_requires = ['numpy>=1.15.0,<2.0.0',
                    'six>=1.10.0,<2.0.0',
                    'jsonschema>=2.6.0,<3.0.0']

extras_require = {'all': ['networkx>=2.0,<3.0',
                          'pandas>=0.22.0,<0.23.0',
                          'pymongo>=3.7.0,<3.8.0'],
                  ':python_version == "2.7"': ['futures'],
                  ':python_version <= "3.3"': ['enum34>=1.1.6,<2.0.0']}

packages = ['dimod',
            'dimod.core',
            'dimod.generators',
            'dimod.higherorder',
            'dimod.reference',
            'dimod.reference.composites',
            'dimod.reference.samplers',
            'dimod.roof_duality',
            'dimod.serialization',
            'dimod.testing',
            'dimod.views',
            ]

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

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

ext = '.pyx' if USE_CYTHON else '.cpp'

extensions = [Extension("dimod.roof_duality._fix_variables",
                        ['dimod/roof_duality/_fix_variables'+ext,
                         'dimod/roof_duality/src/fix_variables.cpp'],
                        include_dirs=['dimod/roof_duality/src/'])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

###############################################################################
# The following code to fail gracefully when when c++ extensions cannot be
# built is adapted from the simplejson library under the MIT License
#
# MIT License
# ===========
#
# Copyright (c) 2006 Bob Ippolito
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


class BuildFailed(Exception):
    pass


class ve_build_ext(build_ext):
    # This class allows C extension building to fail.

    def run(self):
        try:
            build_ext.run(self)
        except DistutilsPlatformError:
            raise BuildFailed()

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError, DistutilsPlatformError):
            raise BuildFailed()


def run_setup(cpp):
    if cpp:
        kw = dict(cmdclass=dict(build_ext=ve_build_ext), ext_modules=extensions)
    else:
        kw = {}

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
        zip_safe=False,
        python_requires=python_requires,
        **kw
    )


try:
    run_setup(cpp=True)
except BuildFailed:
    print("building c++ extensions failed, trying to build without")
    run_setup(cpp=False)
