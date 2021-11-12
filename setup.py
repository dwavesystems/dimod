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

import numpy

from Cython.Build import cythonize
from distutils.extension import Extension
from distutils.command.build_ext import build_ext as _build_ext

extra_compile_args = {
    'msvc': ['/EHsc'],
    'unix': ['-std=c++11'],
}

extra_link_args = {
    'msvc': [],
    'unix': ['-std=c++11'],
}


class build_ext(_build_ext):
    def build_extensions(self):
        compiler = self.compiler.compiler_type

        compile_args = extra_compile_args[compiler]
        for ext in self.extensions:
            ext.extra_compile_args.extend(compile_args)

        link_args = extra_link_args[compiler]
        for ext in self.extensions:
            ext.extra_compile_args.extend(link_args)

        super().build_extensions()

    def finalize_options(self):
        # allow us to set/override the `-j` user option with the
        # DIMOD_NUM_BUILD_JOBS env (inspired by NumPy's NPY_NUM_BUILD_JOBS).
        # This is useful for building in CI via pip and other places where
        # messing with PIP_GLOBAL_OPTION creates undesired side-effects.
        # Note that this is different than building a single extension from
        # multiple compiled object files. For that we could use NumPy's
        # `numpy.distutils.ccompiler.CCompiler_compile` since we use
        # NumPy at compile time.
        parallel = os.getenv('DIMOD_NUM_BUILD_JOBS')
        if parallel is not None:
            self.parallel = parallel
        super().finalize_options()


setup(
    name='dimod',
    cmdclass=dict(build_ext=build_ext),
    ext_modules=cythonize(
        ['dimod/binary/cybqm/*.pyx',
         'dimod/bqm/adjvectorbqm.pyx',
         'dimod/bqm/utils.pyx',
         'dimod/bqm/common.pyx',
         'dimod/cyutilities.pyx',
         'dimod/cyvariables.pyx',
         'dimod/discrete/cydiscrete_quadratic_model.pyx',
         'dimod/quadratic/cyqm/*.pyx',
         ],
        annotate=bool(os.getenv('CYTHON_ANNOTATE', False)),
        nthreads=int(os.getenv('CYTHON_NTHREADS', 0)),
        ),
    include_dirs=[
        numpy.get_include(),
        'dimod/include/',
        ],
    install_requires=[
        # ignoring 1.21.0 and 1.21.1 due to https://github.com/dwavesystems/dimod/issues/901
        'numpy>=1.17.3,<2.0.0,!=1.21.0,!=1.21.1', # keep synced with circle-ci, pyproject.toml
        'dwave-preprocessing>=0.3,<0.4',
        'pyparsing>=2.4.7, <3.0.0',
        ],
)
