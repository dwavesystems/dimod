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

import jsonschema
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
    if obj.get("basetype", "") != "SampleSet":
        return False
    # we could do more checking but probably this is sufficient
    return True


def _is_bqm_v2(obj):
    if obj.get("basetype", "") != "BinaryQuadraticModel":
        return False
    # we could do more checking but probably this is sufficient
    return True


def dimod_object_hook(obj):
    """JSON-decoding for dimod objects.

    See Also:
        :class:`json.JSONDecoder` for using custom decoders.

    """
    if _is_sampleset_v2(obj):
        # in the future we could handle subtypes but right now we just have the
        # one
        return SampleSet.from_serializable(obj)
    elif _is_bqm_v2(obj):
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


###############################################################################
# Deprecated
# legacy decoding formats
###############################################################################

json_schema_version = "1.0.0"

bqm_json_schema_v1 = {
  "$schema": "http://json-schema.org/draft-04/schema#",
  "title": "binary quadratic model schema",
  "type": "object",
  "required": ["linear_terms",
               "info",
               "offset",
               "quadratic_terms",
               "variable_labels",
               "variable_type",
               "version"],
  "properties": {
    "info": {
      "type": "object"
    },
    "variable_labels": {
      "type": "array",
      "items": {
        "type": ["integer", "string", "array"],
        "minimum": 0
      }
    },
    "variable_type": {
      "type":"string",
      "enum":["SPIN", "BINARY"]
    },
    "offset" : {
      "type": "number"
    },
    "linear_terms": {
      "type": "array",
      "items": {
        "type": "object",
        "required":["label", "bias"],
        "properties": {
          "label" : {
            "type": ["integer", "string", "array"],
            "minimum": 0
          },
          "bias":{
             "type": "number"
          }
        }
      }
    },
    "quadratic_terms": {
      "type": "array",
      "items": {
        "type": "object",
        "required":["label_head", "label_tail", "bias"],
        "properties": {
          "label_head" : {
            "type": ["integer", "string", "array"],
            "minimum": 0
          },
          "label_tail" : {
            "type": ["integer", "string", "array"],
            "minimum": 0
          },
          "bias":{
             "type": "number"
          }
        }
      }
    },
    "version": {
      "type": "object",
      "required": ["bqm_schema", "dimod"],
      "properties": {
        "bqm_schema": {
          "enum":["1.0.0"]
        }
      }
    }
  }
}


sampleset_json_schema_v1 = {
  "$schema": "http://json-schema.org/draft-04/schema#",
  "title": "sample set schema",
  "type": "object",
  "required": ["record",
               "info",
               "variable_labels",
               "variable_type",
               "version"],
  "properties": {
    "record": {
        "type": "object",
        "required": ["sample",
                     "energy",
                     "num_occurrences"]
    },
    "info": {
      "type": "object"
    },
    "variable_labels": {
      "type": "array",
      "items": {
        "type": ["integer", "string", "array"],
        "minimum": 0
      }
    },
    "variable_type": {
      "type":"string",
      "enum":["SPIN", "BINARY"]
    },
    "version": {
      "type": "object",
      "required": ["sampleset_schema", "dimod"],
      "properties": {
        "sampleset_schema": {
          "enum":["1.0.0"]
        }
      }
    }
  }
}


def _decode_label(label):
    """Convert a list label into a tuple. Works recursively on nested lists."""
    if isinstance(label, list):
        return tuple(_decode_label(v) for v in label)
    return label


def _encode_label(label):
    """Convert a tuple label into a list. Works recursively on nested tuples."""
    if isinstance(label, tuple):
        return [_encode_label(v) for v in label]
    return label


def _pack_record(record):
    doc = {}
    for field in record.dtype.fields:
        dat = record[field]
        if field == 'sample':
            binary = np.packbits(dat > 0).tobytes()
        else:
            binary = dat.tobytes()

        doc[field] = {'data': (base64.b64encode(binary)).decode("UTF-8"),
                      'shape': dat.shape,
                      'dtype': str(dat.dtype)}
    return doc


def _prod(iterable):
    return reduce(operator.mul, iterable, 1)


def _unpack_record(obj, vartype):
    fields = {}
    datatypes = []

    for field, data in obj.items():

        shape = tuple(data['shape'])
        dtype = data['dtype']

        if field == 'sample':
            raw = np.unpackbits(np.frombuffer(base64.b64decode(data['data']), dtype=np.uint8))
            arr = raw[:_prod(shape)].astype(dtype).reshape(shape)

            if vartype is Vartype.SPIN:
                arr = 2 * arr - 1
        else:
            raw = np.frombuffer(base64.b64decode(data['data']), dtype=dtype)
            arr = raw[:_prod(shape)].reshape(shape)

        fields[field] = arr
        datatypes.append((field, dtype, shape[1:]))

    record = np.rec.array(np.zeros(shape[0], dtype=datatypes))

    for field, arr in fields.items():
        record[field] = arr

    return record


def bqm_decode_hook(dct, cls=None):
    # deprecated

    if cls is None:
        cls = BinaryQuadraticModel

    if jsonschema.Draft4Validator(bqm_json_schema_v1).is_valid(dct):
        # BinaryQuadraticModel

        linear = {_decode_label(obj['label']): obj['bias'] for obj in dct['linear_terms']}
        quadratic = {(_decode_label(obj['label_head']), _decode_label(obj['label_tail'])): obj['bias']
                     for obj in dct['quadratic_terms']}
        offset = dct['offset']
        vartype = Vartype[dct['variable_type']]

        return cls(linear, quadratic, offset, vartype, **dct['info'])

    return dct


def sampleset_decode_hook(dct, cls=None):
    # deprecated

    if cls is None:
        cls = SampleSet

    if jsonschema.Draft4Validator(sampleset_json_schema_v1).is_valid(dct):
        # SampleSet

        vartype = Vartype[dct['variable_type']]
        record = _unpack_record(dct['record'], vartype)
        return cls(record, dct['variable_labels'], dct['info'], vartype)

    return dct
