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

from dimod.core import *
import dimod.core

from dimod.reference import *
import dimod.reference

try:
    import dimod.roof_duality._fix_variables as _
except ImportError:
    pass
else:
    # we only import fix_variables function into the top level namespace if the c++ extension
    # is built
    from dimod.roof_duality import fix_variables

import dimod.testing

from dimod.binary_quadratic_model import BinaryQuadraticModel, BQM
import dimod.binary_quadratic_model

import dimod.decorators

import dimod.generators

from dimod.exceptions import *
import dimod.exceptions

from dimod.higherorder import make_quadratic, poly_energy, poly_energies, BinaryPolynomial
import dimod.higherorder

from dimod.package_info import __version__, __author__, __authoremail__, __description__

from dimod.sampleset import as_samples, concatenate, SampleSet

from dimod.serialization.format import set_printoptions

from dimod.response import *
import dimod.response

from dimod.utilities import *
import dimod.utilities

from dimod.vartypes import as_vartype, Vartype, SPIN, BINARY
