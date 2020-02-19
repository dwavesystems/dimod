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
# =============================================================================
"""
JSON-encoding of dimod objects.

Examples:

    >>> import json
    >>> from dimod.serialization.json import DimodEncoder, DimodDecoder
    ...
    >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {('a', 'b'): -1})
    >>> s = json.dumps(bqm, cls=DimodEncoder)
    >>> new = json.loads(s, cls=DimodDecoder)
    >>> bqm == new
    True

    >>> import json
    >>> from dimod.serialization.json import DimodEncoder, DimodDecoder
    ...
    >>> sampleset = dimod.SampleSet.from_samples({'a': -1, 'b': 1}, dimod.SPIN, energy=5)
    >>> s = json.dumps(sampleset, cls=DimodEncoder)
    >>> new = json.loads(s, cls=DimodDecoder)
    >>> sampleset == new
    True

    >>> import json
    >>> from dimod.serialization.json import DimodEncoder, DimodDecoder
    ...
    >>> # now inside a list
    >>> s = json.dumps([sampleset, bqm], cls=DimodEncoder)
    >>> new = json.loads(s, cls=DimodDecoder)
    >>> new == [sampleset, bqm]
    True

"""

from __future__ import absolute_import

import json
import base64
import operator

from functools import reduce

import numpy as np

from six import iteritems

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.package_info import __version__
from dimod.sampleset import SampleSet
from dimod.vartypes import Vartype

__all__ = 'DimodEncoder', 'DimodDecoder', 'dimod_object_hook'


class DimodEncoder(json.JSONEncoder):
    """Subclass the JSONEncoder for dimod objects.
    """
    def default(self, obj):
        if isinstance(obj, (SampleSet, BinaryQuadraticModel)):
            return obj.to_serializable()

        return json.JSONEncoder.default(self, obj)


def _is_sampleset_v2(obj):
    if obj.get("type", "") != "SampleSet":
        return False
    # we could do more checking but probably this is sufficient
    return True


def _is_bqm(obj):
    # we could do more checking but probably this is sufficient
    return obj.get("type", "") == "BinaryQuadraticModel"


def dimod_object_hook(obj):
    """JSON-decoding for dimod objects.

    See Also:
        :class:`json.JSONDecoder` for using custom decoders.

    """
    if _is_sampleset_v2(obj):
        # in the future we could handle subtypes but right now we just have the
        # one
        return SampleSet.from_serializable(obj)
    elif _is_bqm(obj):
        # in the future we could handle subtypes but right now we just have the
        # one
        return BinaryQuadraticModel.from_serializable(obj)
    return obj


class DimodDecoder(json.JSONDecoder):
    """Subclass the JSONDecoder for dimod objects.

    Uses :func:`.dimod_object_hook`.
    """
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=dimod_object_hook, *args, **kwargs)
