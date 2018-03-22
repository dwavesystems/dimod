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


def bqm_decode_hook(dct):
    if 'bias' in dct:
        bias = dct['bias']
        if 'label' in dct:
            u = dct['label']
            if isinstance(u, list):
                u = tuple(u)
            return (u, bias)
        else:
            u = dct['label_head']
            v = dct['label_tail']
            if isinstance(u, list):
                u = tuple(u)
            if isinstance(v, list):
                v = tuple(v)
            return ((u, v), bias)

    elif 'linear_terms' in dct and 'quadratic_terms' in dct:
        return BinaryQuadraticModel(dict(dct['linear_terms']),
                                    dict(dct['quadratic_terms']),
                                    dct['offset'],
                                    Vartype[dct['variable_type']],
                                    **dct['info'])

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

            # by using the stream objects, we don't need to create the entire json_dict in memory
            json_dict = {"linear_terms": _JSONLinearBiasStream.from_dict(obj.linear),
                         "quadratic_terms": _JSONQuadraticBiasStream.from_dict(obj.quadratic),
                         "offset": obj.offset,
                         "variable_type": vartype_string,
                         "version": {"dimod": __version__, "bqm_schema": bqm_json_schema_version},
                         "variable_labels": _JSONLabelsStream.from_dict(obj.linear),
                         "info": obj.info}
            return json_dict

        if isinstance(obj, Response):
            # we will eventually want to implement this
            raise NotImplementedError

        return json.JSONEncoder.default(self, obj)


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
            yield {"bias": bias, "label": v}

    def __len__(self):
        return len(self.linear)


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
            yield {"bias": bias, "label_head": v, "label_tail": u}

    def __len__(self):
        return len(self.quadratic)


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
            yield u

    def __len__(self):
        return len(self.linear)
