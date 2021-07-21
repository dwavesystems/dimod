# Copyright 2021 D-Wave Systems Inc.
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

import struct
import tempfile

from collections.abc import Callable
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, Iterator, Iterable, Mapping, Optional, Sequence, Tuple, Union
from typing import BinaryIO, ByteString
from typing import TYPE_CHECKING

import numpy as np

try:
    from numpy.typing import ArrayLike, DTypeLike
except ImportError:
    ArrayLike = Any
    DTypeLike = Any

from dimod.decorators import forwarding_method
from dimod.quadratic.cyqm import cyQM_float32, cyQM_float64
from dimod.serialization.fileview import SpooledTemporaryFile, _BytesIO
from dimod.serialization.fileview import VariablesSection, Section
from dimod.serialization.fileview import load, read_header, write_header
from dimod.sym import Eq, Ge, Le, Comparison
from dimod.typing import Variable, Bias, VartypeLike
from dimod.variables import Variables
from dimod.vartypes import Vartype, as_vartype
from dimod.views.quadratic import QuadraticViewsMixin

if TYPE_CHECKING:
    # avoid circular imports
    from dimod import BinaryQuadraticModel


__all__ = ['QuadraticModel', 'QM', 'Integer']


QM_MAGIC_PREFIX = b'DIMODQM'


Vartypes = Union[Mapping[Variable, Vartype], Iterable[Tuple[Variable, VartypeLike]]]


class LinearSection(Section):
    """Serializes the linear biases of a quadratic model."""
    magic = b'LINB'

    def __init__(self, qm: 'QuadraticModel'):
        self.quadratic_model = qm

    def dump_data(self):
        return memoryview(self.quadratic_model.data._ilinear()).cast('B')

    @classmethod
    def loads_data(self, data, *, dtype, num_variables):
        arr = np.frombuffer(data[:num_variables*np.dtype(dtype).itemsize], dtype=dtype)
        return arr


class NeighborhoodSection(Section):
    magic = b'NEIG'

    def __init__(self, qm: 'QuadraticModel'):
        self.quadratic_model = qm

    def dump_data(self, *, vi: int):
        arr = self.quadratic_model.data._ilower_triangle(vi)
        return (struct.pack('<q', arr.shape[0]) + memoryview(arr).cast('B'))

    @classmethod
    def loads_data(self, data):
        return struct.unpack('<q', data[:8])[0], data[8:]


class OffsetSection(Section):
    """Serializes the offset of a quadratic model."""
    magic = b'OFFS'

    def __init__(self, qm: 'QuadraticModel'):
        self.quadratic_model = qm

    def dump_data(self):
        return memoryview(self.quadratic_model.offset).cast('B')

    @classmethod
    def loads_data(self, data, *, dtype):
        arr = np.frombuffer(data[:np.dtype(dtype).itemsize], dtype=dtype)
        return arr[0]


class VartypesSection(Section):
    """Serializes the vartypes of a quadratic model."""
    magic = b'VTYP'

    def __init__(self, qm: 'QuadraticModel'):
        self.quadratic_model = qm

    def dump_data(self):
        return self.quadratic_model.data._ivartypes()

    @classmethod
    def loads_data(self, data):
        return data


class QuadraticModel(QuadraticViewsMixin):
    _DATA_CLASSES = {
        np.dtype(np.float32): cyQM_float32,
        np.dtype(np.float64): cyQM_float64,
    }

    DEFAULT_DTYPE = np.float64
    """The default dtype used to construct the class."""

    def __init__(self,
                 linear: Optional[Mapping[Variable, Bias]] = None,
                 quadratic: Optional[Mapping[Tuple[Variable, Variable], Bias]] = None,
                 offset: Bias = 0,
                 vartypes: Optional[Vartypes] = None,
                 *,
                 dtype: Optional[DTypeLike] = None):
        dtype = np.dtype(self.DEFAULT_DTYPE) if dtype is None else np.dtype(dtype)
        self.data = self._DATA_CLASSES[np.dtype(dtype)]()

        if vartypes is not None:
            if isinstance(vartypes, Mapping):
                vartypes = vartypes.items()
            for v, vartype in vartypes:
                self.add_variable(vartype, v)
                self.set_linear(v, 0)

        # todo: in the future we can support more types for construction, but
        # let's keep it simple for now
        if linear is not None:
            for v, bias in linear.items():
                self.add_linear(v, bias)
        if quadratic is not None:
            for (u, v), bias in quadratic.items():
                self.add_quadratic(u, v, bias)
        self.offset += offset

    def __deepcopy__(self, memo: Dict[int, Any]) -> 'QuadraticModel':
        new = type(self).__new__(type(self))
        new.data = deepcopy(self.data, memo)
        memo[id(self)] = new
        return new

    def __repr__(self):
        vartypes = {v: self.vartype(v).name for v in self.variables}
        return (f"{type(self).__name__}({self.linear}, {self.quadratic}, "
                f"{self.offset}, {vartypes}, dtype={self.dtype.name!r})")

    def __add__(self, other: Union['QuadraticModel', Bias]) -> 'QuadraticModel':
        # in python 3.8+ we could do this is functools.singledispatchmethod
        if isinstance(other, QuadraticModel):
            new = self.copy()
            new.update(other)
            return new
        if isinstance(other, Number):
            new = self.copy()
            new.offset += other
            return new
        return NotImplemented

    def __iadd__(self, other: Union['QuadraticModel', Bias]) -> 'QuadraticModel':
        # in python 3.8+ we could do this is functools.singledispatchmethod
        if isinstance(other, QuadraticModel):
            self.update(other)
            return self
        if isinstance(other, Number):
            self.offset += other
            return self
        return NotImplemented

    def __radd__(self, other: Bias) -> 'QuadraticModel':
        # should only miss on number
        if isinstance(other, Number):
            new = self.copy()
            new.offset += other
            return new
        return NotImplemented

    def __mul__(self, other: Union['QuadraticModel', Bias]) -> 'QuadraticModel':
        if isinstance(other, QuadraticModel):
            if not (self.is_linear() and other.is_linear()):
                raise TypeError(
                    "cannot multiply QMs with interactions")

            # todo: performance

            new = type(self)(dtype=self.dtype)

            for v in self.variables:
                new.add_variable(self.vartype(v), v)
            for v in other.variables:
                new.add_variable(other.vartype(v), v)

            self_offset = self.offset
            other_offset = other.offset

            for u, ubias in self.linear.items():
                for v, vbias in other.linear.items():
                    if u == v:
                        u_vartype = self.vartype(u)
                        if u_vartype is Vartype.BINARY:
                            new.add_linear(u, ubias*vbias)
                        elif u_vartype is Vartype.SPIN:
                            new.offset += ubias * vbias
                        elif u_vartype is Vartype.INTEGER:
                            new.add_quadratic(u, v, ubias*vbias)
                        else:
                            raise RuntimeError("unexpected vartype")
                    else:
                        new.add_quadratic(u, v, ubias * vbias)

                new.add_linear(u, ubias * other_offset)

            for v, bias in other.linear.items():
                new.add_linear(v, bias*self_offset)

            return new
        if isinstance(other, Number):
            new = self.copy()
            new.scale(other)
            return new
        return NotImplemented

    def __imul__(self, other: Bias) -> 'QuadraticModel':
        # in-place multiplication is only defined for numbers
        if isinstance(other, Number):
            raise NotImplementedError
        return NotImplemented

    def __rmul__(self, other: Bias) -> 'QuadraticModel':
        # should only miss on number
        if isinstance(other, Number):
            return self * other  # communative
        return NotImplemented

    def __neg__(self: 'QuadraticModel') -> 'QuadraticModel':
        new = self.copy()
        new.scale(-1)
        return new

    def __sub__(self, other: Union['QuadraticModel', Bias]) -> 'QuadraticModel':
        if isinstance(other, QuadraticModel):
            new = self.copy()
            new.scale(-1)
            new.update(other)
            new.scale(-1)
            return new
        if isinstance(other, Number):
            new = self.copy()
            new.offset -= other
            return new
        return NotImplemented

    def __isub__(self, other: Union['QuadraticModel', Bias]) -> 'QuadraticModel':
        if isinstance(other, QuadraticModel):
            self.scale(-1)
            self.update(other)
            self.scale(-1)
            return self
        if isinstance(other, Number):
            self.offset -= other
            return self
        return NotImplemented

    def __rsub__(self, other: Bias) -> 'QuadraticModel':
        # should only miss on a number
        if isinstance(other, Number):
            new = self.copy()
            new.scale(-1)
            new += other
            return new
        return NotImplemented

    def __eq__(self, other: Number) -> Comparison:
        if isinstance(other, Number):
            return Eq(self, other)
        return NotImplemented

    def __ge__(self, other: Bias) -> Comparison:
        if isinstance(other, Number):
            return Ge(self, other)
        return NotImplemented

    def __le__(self, other: Bias) -> Comparison:
        if isinstance(other, Number):
            return Le(self, other)
        return NotImplemented

    @property
    def dtype(self) -> np.dtype:
        """Data-type of the model's biases."""
        return self.data.dtype

    @property
    def num_interactions(self) -> int:
        """Number of interactions in the model.

        The complexity is linear in the number of variables.
        """
        return self.data.num_interactions()

    @property
    def num_variables(self) -> int:
        """Number of variables in the model."""
        return self.data.num_variables()

    @property
    def offset(self) -> np.number:
        """Constant energy offset associated with the model."""
        return self.data.offset

    @offset.setter
    def offset(self, offset):
        self.data.offset = offset

    @property
    def shape(self) -> Tuple[int, int]:
        """A 2-tuple of :attr:`num_variables` and :attr:`num_interactions`."""
        return self.num_variables, self.num_interactions

    @property
    def variables(self) -> Variables:
        """The variables of the quadratic model"""
        return self.data.variables

    @forwarding_method
    def add_linear(self, v: Variable, bias: Bias):
        """Add a quadratic term."""
        return self.data.add_linear

    @forwarding_method
    def add_quadratic(self, u: Variable, v: Variable, bias: Bias):
        return self.data.add_quadratic

    @forwarding_method
    def add_variable(self, vartype: VartypeLike,
                     v: Optional[Variable] = None,
                     *, lower_bound: int = 0, upper_bound: Optional[int] = None) -> Variable:
        """Add a variable to the quadratic model.

        Args:
            vartype:
                Variable type. One of:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``
                * :class:`.Vartype.INTEGER`, ``'INTEGER'``

            label:
                A label for the variable. Defaults to the length of the
                quadratic model, if that label is available. Otherwise defaults
                to the lowest available positive integer label.

            lower_bound:
                A lower bound on the variable. Ignored when the variable is
                not :class:`Vartype.INTEGER`.

            upper_bound:
                An upper bound on the variable. Ignored when the variable is
                not :class:`Vartype.INTEGER`.

        Returns:
            The variable label.

        """
        return self.data.add_variable

    def add_variables_from(self, vartype: VartypeLike, variables: Iterable[Variable]):
        vartype = as_vartype(vartype, extended=True)
        for v in variables:
            self.add_variable(vartype, v)

    def change_vartype(self, vartype: VartypeLike, v: Variable) -> "QuadraticModel":
        """Change the variable type of the given variable, updating the biases.

        Example:
            >>> qm = dimod.QuadraticModel()
            >>> a = qm.add_variable('SPIN', 'a')
            >>> qm.set_linear(a, 1.5)
            >>> qm.energy({a: +1})
            1.5
            >>> qm.energy({a: -1})
            -1.5
            >>> qm.change_vartype('BINARY', a)
            >>> qm.energy({a: 1})
            1.5
            >>> qm.energy({a: 0})
            -1.5

        """
        self.data.change_vartype(vartype, v)
        return self

    def copy(self):
        """Return a copy."""
        return deepcopy(self)

    @forwarding_method
    def degree(self, v: Variable) -> int:
        """Return the degree of variable `v`.

        The degree is the number of interactions that contain `v`.
        """
        return self.data.degree

    def energies(self, samples_like, dtype: Optional[DTypeLike] = None) -> np.ndarray:
        return self.data.energies(samples_like, dtype=dtype)

    def energy(self, sample, dtype=None) -> Bias:
        """Determine the energy of the given sample.

        Args:
            samples_like (samples_like):
                Raw sample. `samples_like` is an extension of
                NumPy's array_like structure. See :func:`.as_samples`.

            dtype (data-type, optional, default=None):
                Desired NumPy data type for the energy. Matches
                :attr:`.dtype` by default.

        Returns:
            The energy.

        """
        energy, = self.energies(sample, dtype=dtype)
        return energy

    @classmethod
    def from_bqm(cls, bqm: 'BinaryQuadraticModel') -> 'QuadraticModel':
        obj = cls.__new__(cls)

        try:
            obj.data = obj._DATA_CLASSES[np.dtype(bqm.dtype)].from_cybqm(bqm.data)
        except (TypeError, KeyError):
            # not a cybqm or unsupported dtype
            obj = cls()
        else:
            return obj

        # fallback to python
        for v in bqm.variables:
            obj.set_linear(obj.add_variable(bqm.vartype, v), bqm.get_linear(v))

        for u, v, bias in bqm.iter_quadratic():
            obj.set_quadratic(u, v, bias)

        obj.offset = bqm.offset

        return obj

    @classmethod
    def from_file(cls, fp: Union[BinaryIO, ByteString]):
        """Construct a QM from a file-like object.

        The inverse of :meth:`~QuadraticModel.to_file`.
        """
        if isinstance(fp, ByteString):
            file_like: BinaryIO = _BytesIO(fp)  # type: ignore[assignment]
        else:
            file_like = fp

        header_info = read_header(file_like, QM_MAGIC_PREFIX)

        num_variables, num_interactions = header_info.data['shape']
        dtype = np.dtype(header_info.data['dtype'])
        itype = np.dtype(header_info.data['itype'])

        if header_info.version > (2, 0):
            raise ValueError("cannot load a QM serialized with version "
                             f"{header_info.version!r}, "
                             "try upgrading your dimod version")

        obj = cls(dtype=dtype)

        # the vartypes
        obj.data._ivartypes_load(VartypesSection.load(file_like), num_variables)

        # offset
        obj.offset += OffsetSection.load(file_like, dtype=dtype)

        # linear
        obj.data.add_linear_from_array(
            LinearSection.load(file_like, dtype=dtype, num_variables=num_variables))

        # quadratic
        for vi in range(num_variables):
            obj.data._ilower_triangle_load(vi, *NeighborhoodSection.load(file_like))

        # labels (if applicable)
        if header_info.data['variables']:
            obj.relabel_variables(dict(enumerate(VariablesSection.load(file_like))))

        return obj

    @forwarding_method
    def get_linear(self, v: Variable) -> Bias:
        """Get the linear bias of `v`."""
        return self.data.get_linear

    @forwarding_method
    def get_quadratic(self, u: Variable, v: Variable,
                      default: Optional[Bias] = None) -> Bias:
        return self.data.get_quadratic

    def is_equal(self, other):
        """Return True if the given model has the same variables, vartypes and biases."""
        if isinstance(other, Number):
            return not self.num_variables and self.offset == other
        # todo: performance
        try:
            return (self.shape == other.shape  # redundant, fast to check
                    and self.offset == other.offset
                    and self.linear == other.linear
                    and all(self.vartype(v) == other.vartype(v) for v in self.variables)
                    and self.adj == other.adj)
        except AttributeError:
            return False

    def is_linear(self) -> bool:
        """Return True if the model has no quadratic interactions."""
        return self.data.is_linear()

    @forwarding_method
    def iter_neighborhood(self, v: Variable) -> Iterator[Tuple[Variable, Bias]]:
        """Iterate over the neighbors and quadratic biases of a variable."""
        return self.data.iter_neighborhood

    @forwarding_method
    def iter_quadratic(self) -> Iterator[Tuple[Variable, Variable, Bias]]:
        return self.data.iter_quadratic

    @forwarding_method
    def lower_bound(self, v: Variable) -> Bias:
        """Return the lower bound on variable `v`."""
        return self.data.lower_bound

    @forwarding_method
    def reduce_linear(self, function: Callable,
                      initializer: Optional[Bias] = None) -> Any:
        """Apply function of two arguments cumulatively to the linear biases.
        """
        return self.data.reduce_linear

    @forwarding_method
    def reduce_neighborhood(self, v: Variable, function: Callable,
                            initializer: Optional[Bias] = None) -> Any:
        """Apply function of two arguments cumulatively to the quadratic biases
        associated with a single variable.
        """
        return self.data.reduce_neighborhood

    @forwarding_method
    def reduce_quadratic(self, function: Callable,
                         initializer: Optional[Bias] = None) -> Any:
        """Apply function of two arguments cumulatively to the quadratic
        biases.
        """
        return self.data.reduce_quadratic

    def relabel_variables(self, mapping: Mapping[Variable, Variable],
                          inplace: bool = True) -> 'QuadraticModel':
        if not inplace:
            return self.copy().relabel_variables(mapping, inplace=True)

        self.data.relabel_variables(mapping)
        return self

    def relabel_variables_as_integers(self, inplace: bool = True
                                      ) -> Tuple['QuadraticModel', Mapping[Variable, Variable]]:
        if not inplace:
            return self.copy().relabel_variables_as_integers(inplace=True)

        mapping = self.data.relabel_variables_as_integers()
        return self, mapping

    def remove_interaction(self, u: Variable, v: Variable):
        # This is needed for the views, but I am not sure how often users are
        # removing variables/interactions. For now let's leave it here so
        # we satisfy the ABC and see if it comes up. If not, in the future we
        # can consider removing __delitem__ from the various views.
        raise NotImplementedError("not yet implemented - please open a feature request")

    def remove_variable(self, v: Optional[Variable] = None) -> Variable:
        # see note in remove_interaction
        raise NotImplementedError("not yet implemented - please open a feature request")

    @forwarding_method
    def scale(self, scalar: Bias):
        return self.data.scale

    @forwarding_method
    def set_linear(self, v: Variable, bias: Bias):
        """Set the linear bias of `v`.

        Raises:
            TypeError: If `v` is not hashable.

        """
        return self.data.set_linear

    @forwarding_method
    def set_quadratic(self, u: Variable, v: Variable, bias: Bias):
        """Set the quadratic bias of `(u, v)`.

        Raises:
            TypeError: If `u` or `v` is not hashable.

        """
        return self.data.set_quadratic

    def spin_to_binary(self, inplace: bool = False) -> 'QuadraticModel':
        """Convert any SPIN variables to BINARY."""
        if not inplace:
            return self.copy().spin_to_binary(inplace=True)

        for s in self.variables:
            if self.vartype(s) is Vartype.SPIN:
                self.change_vartype(Vartype.BINARY, s)

        return self

    def to_file(self, *,
                spool_size: int = int(1e9),
                ) -> tempfile.SpooledTemporaryFile:
        """Serialize the QM to a file-like object.

        Args:
            spool_size: Defines the `max_size` passed to the constructor of
                :class:`tempfile.SpooledTemporaryFile`. Determines whether
                the returned file-like's contents will be kept on disk or in
                memory.

        Format Specification (Version 1.0):

            This format is inspired by the `NPY format`_

            The first 7 bytes are a magic string: exactly "DIMODQM".

            The next 1 byte is an unsigned byte: the major version of the file
            format.

            The next 1 byte is an unsigned byte: the minor version of the file
            format.

            The next 4 bytes form a little-endian unsigned int, the length of
            the header data HEADER_LEN.

            The next HEADER_LEN bytes form the header data. This is a
            json-serialized dictionary. The dictionary is exactly:

            .. code-block:: python

                data = dict(shape=qm.shape,
                            dtype=qm.dtype.name,
                            itype=qm.data.index_dtype.name,
                            type=type(qm).__name__,
                            variables=not qm.variables._is_range(),
                            )

            it is terminated by a newline character and padded with spaces to
            make the entire length of the entire header divisible by 64.

            The binary quadratic model data comes after the header.

        .. _NPY format: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html

        """
        # todo: document the serialization format sections

        file = SpooledTemporaryFile(max_size=spool_size)

        data = dict(shape=self.shape,
                    dtype=self.dtype.name,
                    itype=self.data.index_dtype.name,
                    type=type(self).__name__,
                    variables=not self.variables._is_range(),
                    )

        write_header(file, QM_MAGIC_PREFIX, data, version=(1, 0))

        # the vartypes
        file.write(VartypesSection(self).dumps())

        # offset
        file.write(OffsetSection(self).dumps())

        # linear
        file.write(LinearSection(self).dumps())

        # quadraic
        neighborhood_section = NeighborhoodSection(self)
        for vi in range(self.num_variables):
            file.write(neighborhood_section.dumps(vi=vi))

        # the labels (if needed)
        if data['variables']:
            file.write(VariablesSection(self.variables).dumps())

        file.seek(0)
        return file

    def update(self, other: 'QuadraticModel'):
        # this can be improved a great deal with c++, but for now let's use
        # python for simplicity

        if any(v in self.variables and self.vartype(v) != other.vartype(v)
               for v in other.variables):
            raise ValueError("given quadratic model has variables with conflicting vartypes")

        for v in other.variables:
            self.add_linear(self.add_variable(other.vartype(v), v), other.get_linear(v))

        for u, v, bias in other.iter_quadratic():
            self.add_quadratic(u, v, bias)

        self.offset += other.offset

    @forwarding_method
    def upper_bound(self, v: Variable) -> Bias:
        """Return the upper bound on variable `v`."""
        return self.data.upper_bound

    @forwarding_method
    def vartype(self, v: Variable) -> Vartype:
        """The variable type of the given variable."""
        return self.data.vartype


QM = QuadraticModel


def Integer(label: Variable, bias: Bias = 1,
            dtype: Optional[DTypeLike] = None) -> QuadraticModel:
    if label is None:
        raise TypeError("label cannot be None")
    qm = QM(dtype=dtype)
    v = qm.add_variable(Vartype.INTEGER, label)
    qm.set_linear(v, bias)
    return qm

# register fileview loader
load.register(QM_MAGIC_PREFIX, QuadraticModel.from_file)
