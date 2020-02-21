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
"""

The binary quadratic model (BQM) class contains
Ising and quadratic unconstrained binary optimization (QUBO) models
used by samplers such as the D-Wave system.

The :term:`Ising` model is an objective function of :math:`N` variables
:math:`s=[s_1,...,s_N]` corresponding to physical Ising spins, where :math:`h_i`
are the biases and :math:`J_{i,j}` the couplings (interactions) between spins.

.. math::

    \\text{Ising:} \\qquad  E(\\bf{s}|\\bf{h},\\bf{J})
    = \\left\\{ \\sum_{i=1}^N h_i s_i + \\sum_{i<j}^N J_{i,j} s_i s_j  \\right\\}
    \\qquad\\qquad s_i\\in\\{-1,+1\\}


The :term:`QUBO` model is an objective function of :math:`N` binary variables represented
as an upper-diagonal matrix :math:`Q`, where diagonal terms are the linear coefficients
and the nonzero off-diagonal terms the quadratic coefficients.

.. math::

    \\text{QUBO:} \\qquad E(\\bf{x}| \\bf{Q})  =  \\sum_{i\\le j}^N x_i Q_{i,j} x_j
    \\qquad\\qquad x_i\\in \\{0,1\\}

The :class:`.BinaryQuadraticModel` class can contain both these models and its methods provide
convenient utilities for working with, and interworking between, the two representations
of a problem.

"""
import inspect
import warnings

from collections.abc import Sized, Iterable, Container

import numpy as np

from dimod.serialization.utils import serialize_ndarrays, deserialize_ndarrays
from dimod.variables import iter_serialize_variables

from dimod.bqm.adjdictbqm import AdjDictBQM

__all__ = ['BinaryQuadraticModel', 'BQM']


class BinaryQuadraticModel(AdjDictBQM, Sized, Iterable, Container):
    """Encodes a binary quadratic model.

    Binary quadratic model is the superclass that contains the `Ising model`_ and the QUBO_.

    .. _Ising model: https://en.wikipedia.org/wiki/Ising_model
    .. _QUBO: https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization

    Args:
        linear (dict[variable, bias]):
            Linear biases as a dict, where keys are the variables of
            the binary quadratic model and values the linear biases associated
            with these variables.
            A variable can be any python object that is valid as a dictionary key.
            Biases are generally numbers but this is not explicitly checked.

        quadratic (dict[(variable, variable), bias]):
            Quadratic biases as a dict, where keys are
            2-tuples of variables and values the quadratic biases associated
            with the pair of variables (the interaction).
            A variable can be any python object that is valid as a dictionary key.
            Biases are generally numbers but this is not explicitly checked.
            Interactions that are not unique are added.

        offset (number):
            Constant energy offset associated with the binary quadratic model.
            Any input type is allowed, but many applications assume that offset is a number.
            See :meth:`.BinaryQuadraticModel.energy`.

        vartype (:class:`.Vartype`/str/set):
            Variable type for the binary quadratic model. Accepted input values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

        **kwargs:
            Any additional keyword parameters and their values are stored in
            :attr:`.BinaryQuadraticModel.info`.

    Notes:
        The :class:`.BinaryQuadraticModel` class does not enforce types on biases
        and offsets, but most applications that use this class assume that they are numeric.

    Examples:
        This example creates a binary quadratic model with three spin variables.

        >>> bqm = dimod.BinaryQuadraticModel({0: 1, 1: -1, 2: .5},
        ...                                  {(0, 1): .5, (1, 2): 1.5},
        ...                                  1.4,
        ...                                  dimod.Vartype.SPIN)

        This example creates a binary quadratic model with non-numeric variables
        (variables can be any hashable object).

        >>> bqm = dimod.BQM({'a': 0.0, 'b': -1.0, 'c': 0.5},
        ...                                  {('a', 'b'): -1.0, ('b', 'c'): 1.5},
        ...                                  1.4,
        ...                                  dimod.SPIN)
        >>> len(bqm)
        3
        >>> 'b' in bqm
        True

    Attributes:
        linear (dict[variable, bias]):
            Linear biases as a dict, where keys are the variables of
            the binary quadratic model and values the linear biases associated
            with these variables.

        quadratic (dict[(variable, variable), bias]):
            Quadratic biases as a dict, where keys are 2-tuples of variables, which
            represent an interaction between the two variables, and values
            are the quadratic biases associated with the interactions.

        offset (number):
            The energy offset associated with the model. Same type as given
            on instantiation.

        vartype (:class:`.Vartype`):
            The model's type. One of :class:`.Vartype.SPIN` or :class:`.Vartype.BINARY`.

        variables (keysview):
            The variables in the binary quadratic model as a dictionary keys
            view object.

        adj (dict):
            The model's interactions as nested dicts.
            In graphic representation, where variables are nodes and interactions
            are edges or adjacencies, keys of the outer dict (`adj`) are all
            the model's nodes (e.g. `v`) and values are the inner dicts. For the
            inner dict associated with outer-key/node 'v', keys are all the nodes
            adjacent to `v` (e.g. `u`) and values are quadratic biases associated
            with the pair of inner and outer keys (`u, v`).

        info (dict):
            A place to store miscellaneous data about the binary quadratic model
            as a whole.

        SPIN (:class:`.Vartype`): An alias of :class:`.Vartype.SPIN` for easier access.

        BINARY (:class:`.Vartype`): An alias of :class:`.Vartype.BINARY` for easier access.

    Examples:
       This example creates an instance of the :class:`.BinaryQuadraticModel`
       class for the K4 complete graph, where the nodes have biases
       set equal to their sequential labels and interactions are the
       concatenations of the node pairs (e.g., 23 for u,v = 2,3).

       >>> import dimod
       ...
       >>> linear = {1: 1, 2: 2, 3: 3, 4: 4}
       >>> quadratic = {(1, 2): 12, (1, 3): 13, (1, 4): 14,
       ...              (2, 3): 23, (2, 4): 24,
       ...              (3, 4): 34}
       >>> offset = 0.0
       >>> vartype = dimod.BINARY
       >>> bqm_k4 = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
       >>> bqm_k4.info = {'Complete K4 binary quadratic model.'}
       >>> bqm_k4.info.issubset({'Complete K3 binary quadratic model.',
       ...                       'Complete K4 binary quadratic model.',
       ...                       'Complete K5 binary quadratic model.'})
       True
       >>> bqm_k4.adj.viewitems()   # Show all adjacencies  # doctest: +SKIP
       [(1, {2: 12, 3: 13, 4: 14}),
        (2, {1: 12, 3: 23, 4: 24}),
        (3, {1: 13, 2: 23, 4: 34}),
        (4, {1: 14, 2: 24, 3: 34})]
       >>> bqm_k4.adj[2]            # Show adjacencies for node 2  # doctest: +SKIP
       {1: 12, 3: 23, 4: 24}
       >>> bqm_k4.adj[2][3]         # Show the quadratic bias for nodes 2,3 # doctest: +SKIP
       23

    """

    def __init__(self, *args, **kwargs):

        # handle .info which is being deprecated

        super_params = inspect.signature(super().__init__).parameters

        newkwargs = {}
        self.info = info = {}
        for kw, val in kwargs.items():
            if kw not in super_params:
                info[kw] = val
            else:
                newkwargs[kw] = val

        if info:
            msg = ("BinaryQuadraticModel.info is deprecated and will be removed")
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

        super().__init__(*args, **newkwargs)

    def __contains__(self, v):
        msg = ("treating binary quadratic models as containers is deprecated, "
               "please use `v in bqm.variables` or `bqm.has_variable(v)` "
               "instead")
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return v in self.adj

    def __iter__(self):
        msg = ("iterating over binary quadratic models is deprecated, "
               "please use `for v in bqm.variables` or `bqm.iter_variables()` "
               "instead")
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return iter(self.adj)

    def copy(self):
        """Create a copy of a binary quadratic model.

        Returns:
            :class:`.BinaryQuadraticModel`

        Examples:

            >>> bqm = dimod.BinaryQuadraticModel({1: 1, 2: 2}, {(1, 2): 0.5}, 0.5, dimod.SPIN)
            >>> bqm2 = bqm.copy()

        """
        new = super().copy()
        new.info.update(self.info)
        return new

    @classmethod
    def empty(cls, vartype):
        # for backwards compatibility reasons, some subclasses out there relied
        # on there being 4 arguments, so we recreate that here temporarily
        return cls({}, {}, 0.0, vartype)

    def to_serializable(self, use_bytes=False, bias_dtype=np.float32,
                        bytes_type=bytes):
        """Convert the binary quadratic model to a serializable object.

        Args:
            use_bytes (bool, optional, default=False):
                If True, a compact representation representing the biases as
                bytes is used. Uses :func:`~numpy.ndarray.tobytes`.

            bias_dtype (data-type, optional, default=numpy.float32):
                If `use_bytes` is True, this :class:`~numpy.dtype` will be used
                to represent the bias values in the serialized format.

            bytes_type (class, optional, default=bytes):
                This class will be used to wrap the bytes objects in the
                serialization if `use_bytes` is true.

        Returns:
            dict: An object that can be serialized.

        Examples:

            Encode using JSON

            >>> import json
            ...
            >>> bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 1.0}, {('a', 'b'): -1.0}, 0.0, dimod.SPIN)
            >>> s = json.dumps(bqm.to_serializable())

            Encode using BSON_

            >>> import bson
            ...
            >>> bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 1.0}, {('a', 'b'): -1.0}, 0.0, dimod.SPIN)
            >>> doc = bqm.to_serializable(use_bytes=True)
            >>> b = bson.BSON.encode(doc)  # doctest: +SKIP

        See also:
            :meth:`~.BinaryQuadraticModel.from_serializable`

            :func:`json.dumps`, :func:`json.dump` JSON encoding functions

            :meth:`bson.BSON.encode` BSON encoding method

        .. _BSON: http://bsonspec.org/

        """
        from dimod.package_info import __version__

        schema_version = "3.0.0"

        variables = list(iter_serialize_variables(self.variables))

        try:
            variables.sort()
        except TypeError:
            # cannot unlike types in py3
            pass

        num_variables = len(variables)

        # when doing byte encoding we can use less space depending on the
        # total number of variables
        index_dtype = np.uint16 if num_variables <= 2**16 else np.uint32

        ldata, (irow, icol, qdata), offset = self.to_numpy_vectors(
            dtype=bias_dtype,
            index_dtype=index_dtype,
            sort_indices=True,  # to make it deterministic for the order
            variable_order=variables)

        num_interactions = len(irow)

        doc = {
            # metadata
            "type": type(self).__name__,
            "version": {"bqm_schema": schema_version},
            "use_bytes": bool(use_bytes),
            "index_type": np.dtype(index_dtype).name,
            "bias_type": np.dtype(bias_dtype).name,

            # bqm
            "num_variables": num_variables,
            "num_interactions": num_interactions,
            "variable_labels": variables,
            "variable_type": self.vartype.name,
            "offset": float(offset),
            "info": serialize_ndarrays(self.info, use_bytes=use_bytes,
                                          bytes_type=bytes_type),
            }

        if use_bytes:
            # these are vectors so don't need to specify byte-order
            doc.update({'linear_biases': bytes_type(ldata.tobytes()),
                        'quadratic_biases': bytes_type(qdata.tobytes()),
                        'quadratic_head': bytes_type(irow.tobytes()),
                        'quadratic_tail': bytes_type(icol.tobytes())})
        else:
            doc.update({'linear_biases': ldata.tolist(),
                        'quadratic_biases': qdata.tolist(),
                        'quadratic_head': irow.tolist(),
                        'quadratic_tail': icol.tolist()})

        return doc

    def _asdict(self):
        # support simplejson encoding
        return self.to_serializable()

    @classmethod
    def from_serializable(cls, obj):
        """Deserialize a binary quadratic model.

        Args:
            obj (dict):
                A binary quadratic model serialized by :meth:`~.BinaryQuadraticModel.to_serializable`.

        Returns:
            :obj:`.BinaryQuadraticModel`

        Examples:

            Encode and decode using JSON

            >>> import dimod
            >>> import json
            ...
            >>> bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 1.0}, {('a', 'b'): -1.0}, 0.0, dimod.SPIN)
            >>> s = json.dumps(bqm.to_serializable())
            >>> new_bqm = dimod.BinaryQuadraticModel.from_serializable(json.loads(s))

        See also:
            :meth:`~.BinaryQuadraticModel.to_serializable`

            :func:`json.loads`, :func:`json.load` JSON deserialization functions

        """
        version = obj.get("version", {"bqm_schema": "1.0.0"})["bqm_schema"]
        if version < "2.0.0":
            raise ValueError("No longer supported serialization format")
        elif version < "3.0.0" and obj.get("use_bytes", False):
            # from 2.0.0 to 3.0.0 the formatting of the bytes changed
            raise ValueError("No longer supported serialization format")

        variables = [tuple(v) if isinstance(v, list) else v
                     for v in obj["variable_labels"]]

        if obj["use_bytes"]:
            bias_dtype = np.dtype(obj['bias_type'])
            index_dtype = np.dtype(obj['index_type'])

            ldata = np.frombuffer(obj['linear_biases'], dtype=bias_dtype)
            qdata = np.frombuffer(obj['quadratic_biases'], dtype=bias_dtype)
            irow = np.frombuffer(obj['quadratic_head'], dtype=index_dtype)
            icol = np.frombuffer(obj['quadratic_tail'], dtype=index_dtype)
        else:
            ldata = obj["linear_biases"]
            qdata = obj["quadratic_biases"]
            irow = obj["quadratic_head"]
            icol = obj["quadratic_tail"]

        offset = obj["offset"]
        vartype = obj["variable_type"]

        bqm = cls.from_numpy_vectors(ldata,
                                     (irow, icol, qdata),
                                     offset,
                                     str(vartype),  # handle unicode for py2
                                     variable_order=variables)

        bqm.info.update(deserialize_ndarrays(obj['info']))
        return bqm


BQM = BinaryQuadraticModel
"""Alias for :obj:`.BinaryQuadraticModel`"""
