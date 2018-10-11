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
    from dimod.roof_duality import *

import dimod.testing

from dimod.binary_quadratic_model import *
import dimod.binary_quadratic_model

import dimod.decorators

# for now, let's just import the most important embedding functions
from dimod.embedding import embed_bqm, embed_ising, embed_qubo, chain_break_frequency, unembed_response
import dimod.embedding

import dimod.generators

from dimod.exceptions import *
import dimod.exceptions

from dimod.higherorder import *
import dimod.higherorder

from dimod.package_info import __version__, __author__, __authoremail__, __description__

from dimod.sampleset import *

from dimod.response import *
import dimod.response

from dimod.utilities import *
import dimod.utilities

from dimod.vartypes import *
import dimod.vartypes
