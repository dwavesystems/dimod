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

# version is used by serialization below so we need it before everything
__version__ = '0.10.0.dev6'

from dimod.bqm import *

from dimod.constrained import *
import dimod.constrained

from dimod.core import *
import dimod.core

from dimod.reference import *
import dimod.reference

from dimod.roof_duality import fix_variables

from dimod.binary import *
import dimod.binary

from dimod.discrete import *

import dimod.testing

from dimod.converters import *

import dimod.decorators

import dimod.generators

from dimod.exceptions import *
import dimod.exceptions

from dimod.higherorder import make_quadratic, poly_energy, poly_energies, BinaryPolynomial
import dimod.higherorder

from dimod.package_info import __version__, __author__, __authoremail__, __description__

from dimod.traversal import *

from dimod.sampleset import *

from dimod.serialization.format import set_printoptions

from dimod.utilities import *
import dimod.utilities

from dimod.vartypes import *
