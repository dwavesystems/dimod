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
from pkg_resources import resource_filename

import jsonschema

from six import iteritems

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.package_info import __version__
from dimod.response import Response
from dimod.vartypes import Vartype

bqm_json_schema_version = "1.0.0"

with open(resource_filename(__name__, 'bqm_json_schema.json'), 'r') as schema_file:
    bqm_json_schema = json.load(schema_file)


def _decode_label(label):
    if isinstance(label, list):
        return tuple(_decode_label(v) for v in label)
    return label


def _encode_label(label):
    if isinstance(label, tuple):
        return [_encode_label(v) for v in label]
    return label


def bqm_decode_hook(dct, cls=None):

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


class DimodStreamEncoder(json.JSONEncoder):
    """Subclass the JSONEncoder for dimod objects.

    By using the stream objects, we don't need to create the entire serializable object in memory
    """
    def default(self, obj):

        if isinstance(obj, BinaryQuadraticModel):

            if obj.vartype is Vartype.SPIN:
                vartype_string = 'SPIN'
            elif obj.vartype is Vartype.BINARY:
                vartype_string = 'BINARY'
            else:
                raise RuntimeError("unknown vartype")

            json_dict = {"linear_terms": self._linear_biases(obj.linear),
                         "quadratic_terms": self._quadratic_biases(obj.quadratic),
                         "offset": obj.offset,
                         "variable_type": vartype_string,
                         "version": {"dimod": __version__, "bqm_schema": bqm_json_schema_version},
                         "variable_labels": self._variable_labels(obj.linear),
                         "info": obj.info}
            return json_dict

        elif isinstance(obj, Response):
            # we will eventually want to implement this
            raise NotImplementedError

        return json.JSONEncoder.default(self, obj)

    @staticmethod
    def _linear_biases(linear):
        return _JSONLinearBiasStream.from_dict(linear)

    @staticmethod
    def _quadratic_biases(quadratic):
        return _JSONQuadraticBiasStream.from_dict(quadratic)

    @staticmethod
    def _variable_labels(linear):
        return _JSONLabelsStream.from_dict(linear)


class DimodEncoder(DimodStreamEncoder):
    """Subclass the JSONEncoder for dimod objects.

    In this case we dump the steam contents into arrays so that intermediate representations function
    correctly.
    """
    @staticmethod
    def _linear_biases(linear):
        return list(_JSONLinearBiasStream.from_dict(linear))

    @staticmethod
    def _quadratic_biases(quadratic):
        return list(_JSONQuadraticBiasStream.from_dict(quadratic))

    @staticmethod
    def _variable_labels(linear):
        return list(_JSONLabelsStream.from_dict(linear))


class _JSONLinearBiasStream(list):
    """Allows json to encode linear biases without needing to create a large number of dicts in
    memory.
    """
    @classmethod
    def from_dict(cls, linear):
        jlbs = cls()
        jlbs.linear = linear
        return jlbs

    def __iter__(self):
        for v, bias in iteritems(self.linear):
            if isinstance(v, tuple):
                v = json.loads(json.dumps(v))  # handles nested tuples
            yield {"bias": bias, "label": v}

    def __len__(self):
        return len(self.linear)

    def __bool__(self):
        return bool(self.linear)


class _JSONQuadraticBiasStream(list):
    """Allows json to encode quadratic biases without needing to create a large number of dicts in
    memory.
    """
    @classmethod
    def from_dict(cls, quadratic):
        jqbs = cls()
        jqbs.quadratic = quadratic
        return jqbs

    def __iter__(self):
        for (u, v), bias in iteritems(self.quadratic):
            if isinstance(u, tuple):
                u = _encode_label(u)  # handles nested tuples
            if isinstance(v, tuple):
                v = _encode_label(v)  # handles nested tuples
            yield {"bias": bias, "label_head": u, "label_tail": v}

    def __len__(self):
        return len(self.quadratic)

    def __bool__(self):
        return bool(self.quadratic)


class _JSONLabelsStream(list):
    """Allows json to encode variable labels without needing to create a large list in memory.
    """
    @classmethod
    def from_dict(cls, linear):
        jlbs = cls()
        jlbs.linear = linear
        return jlbs

    def __iter__(self):
        for u in self.linear:
            if isinstance(u, tuple):
                u = _encode_label(u)  # handles nested tuples
            yield u

    def __len__(self):
        return len(self.linear)

    def __bool__(self):
        return bool(self.linear)
