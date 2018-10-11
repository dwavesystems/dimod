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

import json
import base64
import operator

from functools import reduce
from pkg_resources import resource_filename

import jsonschema
import numpy as np

from six import iteritems

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.package_info import __version__
from dimod.sampleset import SampleSet
from dimod.vartypes import Vartype

json_schema_version = "1.0.0"

with open(resource_filename(__name__, 'bqm_json_schema.json'), 'r') as schema_file:
    bqm_json_schema = json.load(schema_file)

with open(resource_filename(__name__, 'sampleset_json_schema.json'), 'r') as schema_file:
    sampleset_json_schema = json.load(schema_file)


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
    """Decode hook as can be used with json.loads."""

    if cls is None:
        cls = BinaryQuadraticModel

    if jsonschema.Draft4Validator(bqm_json_schema).is_valid(dct):
        # BinaryQuadraticModel

        linear = {_decode_label(obj['label']): obj['bias'] for obj in dct['linear_terms']}
        quadratic = {(_decode_label(obj['label_head']), _decode_label(obj['label_tail'])): obj['bias']
                     for obj in dct['quadratic_terms']}
        offset = dct['offset']
        vartype = Vartype[dct['variable_type']]

        return cls(linear, quadratic, offset, vartype, **dct['info'])

    return dct


def sampleset_decode_hook(dct, cls=None):
    """Decode hook as can be used with json.loads."""

    if cls is None:
        cls = SampleSet

    if jsonschema.Draft4Validator(sampleset_json_schema).is_valid(dct):
        # SampleSet

        vartype = Vartype[dct['variable_type']]
        record = _unpack_record(dct['record'], vartype)
        return cls(record, dct['variable_labels'], dct['info'], vartype)

    return dct


class DimodEncoder(json.JSONEncoder):
    """Subclass the JSONEncoder for dimod objects."""
    def default(self, obj):

        if isinstance(obj, BinaryQuadraticModel):

            if obj.vartype is Vartype.SPIN:
                vartype_string = 'SPIN'
            elif obj.vartype is Vartype.BINARY:
                vartype_string = 'BINARY'
            else:
                raise RuntimeError("unknown vartype")

            json_dict = {"linear_terms": list(self._linear_biases(obj.linear)),
                         "quadratic_terms": list(self._quadratic_biases(obj.quadratic)),
                         "offset": obj.offset,
                         "variable_type": vartype_string,
                         "version": {"dimod": __version__, "bqm_schema": json_schema_version},
                         "variable_labels": list(self._variable_labels(obj.linear)),
                         "info": obj.info}
            return json_dict

        elif isinstance(obj, SampleSet):

            if obj.vartype is Vartype.SPIN:
                vartype_string = 'SPIN'
            elif obj.vartype is Vartype.BINARY:
                vartype_string = 'BINARY'
            else:
                raise RuntimeError("unknown vartype")

            return {"record": _pack_record(obj.record),
                    "variable_type": vartype_string,
                    "info": obj.info,
                    "version": {"dimod": __version__, "sampleset_schema": json_schema_version},
                    "variable_labels": list(obj.variables)}

        return json.JSONEncoder.default(self, obj)

    @staticmethod
    def _linear_biases(linear):
        for v, bias in iteritems(linear):
            if isinstance(v, tuple):
                v = json.loads(json.dumps(v))  # handles nested tuples
            yield {"bias": bias, "label": v}

    @staticmethod
    def _quadratic_biases(quadratic):
        for (u, v), bias in iteritems(quadratic):
            if isinstance(u, tuple):
                u = _encode_label(u)  # handles nested tuples
            if isinstance(v, tuple):
                v = _encode_label(v)  # handles nested tuples
            yield {"bias": bias, "label_head": u, "label_tail": v}

    @staticmethod
    def _variable_labels(linear):
        for u in linear:
            if isinstance(u, tuple):
                u = _encode_label(u)  # handles nested tuples
            yield u
