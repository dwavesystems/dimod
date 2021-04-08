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

import os

from setuptools import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext as _build_ext


# if the numpy ranges change here, don't forget to update the circle-ci job
install_requires = ['numpy>=1.17.3,<2.0.0']

setup_requires = ['numpy>=1.17.3,<2.0.0']

extras_require = {'all': ['networkx>=2.0,<3.0',
                          'pandas>=0.22.0,<0.23.0',
                          'pymongo>=3.7.0,<3.8.0'],
                  }

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

ext = '.pyx' if USE_CYTHON else '.cpp'

extra_compile_args = {
    'msvc': ['/EHsc'],
    'unix': ['-std=c++11'],
}

extra_link_args = {
    'msvc': [],
    'unix': ['-std=c++11'],
}


class build_ext(_build_ext):
    def run(self):
        # add numpy headers
        import numpy
        self.include_dirs.append(numpy.get_include())

        # add dimod headers
        include = os.path.join(os.path.dirname(__file__), 'dimod', 'include')
        self.include_dirs.append(include)

        super().run()

    def build_extensions(self):
        compiler = self.compiler.compiler_type

        compile_args = extra_compile_args[compiler]
        for ext in self.extensions:
            ext.extra_compile_args.extend(compile_args)

        link_args = extra_link_args[compiler]
        for ext in self.extensions:
            ext.extra_compile_args.extend(link_args)

        super().build_extensions()

    def initialize_options(self):
        super().initialize_options()
        self.build_tests = None


extensions = [Extension("dimod.roof_duality._fix_variables",
                        ['dimod/roof_duality/_fix_variables'+ext,
                         'dimod/roof_duality/src/fix_variables.cpp'],
                        include_dirs=['dimod/roof_duality/src/']),
              Extension("dimod.bqm.adjvectorbqm",
                        ['dimod/bqm/adjvectorbqm'+ext]),
              Extension("dimod.bqm.utils",
                        ['dimod/bqm/utils'+ext]),
              Extension("dimod.bqm.common",
                        ['dimod/bqm/common'+ext]),
              Extension("dimod.discrete.cydiscrete_quadratic_model",
                        ["dimod/discrete/cydiscrete_quadratic_model"+ext]),
              Extension("dimod.cyvariables", ["dimod/cyvariables"+ext]),
              ]


if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions,
                           # annotate=True,
                           )

setup(
    name='dimod',
    install_requires=install_requires,
    extras_require=extras_require,
    setup_requires=setup_requires,
    cmdclass=dict(build_ext=build_ext),
    ext_modules=extensions,
)
