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

import json
import re
import tempfile
import uuid
import zipfile

from collections.abc import Iterable
from numbers import Number
from typing import Hashable, Optional, Union, BinaryIO, ByteString

import numpy as np

from dimod.binary.binary_quadratic_model import BinaryQuadraticModel, Binary, Spin
from dimod.sym import Comparison, Eq, Le, Ge, Sense
from dimod.serialization.fileview import SpooledTemporaryFile, _BytesIO
from dimod.serialization.fileview import load, read_header, write_header
from dimod.typing import Bias, Variable
from dimod.variables import Variables, serialize_variable, deserialize_variable
from dimod.vartypes import Vartype, as_vartype, VartypeLike

__all__ = ['ConstrainedQuadraticModel', 'CQM']


CQM_MAGIC_PREFIX = b'DIMODCQM'


class TypedVariables(Variables):
    """Tracks variable labels and the vartype of each variable."""
    def __init__(self):
        super().__init__()
        self.vartypes: list[Vartype] = []

    def _append(self, vartype: VartypeLike, v: Optional[Variable] = None,
                permissive: bool = False) -> Variable:
        if permissive and v is not None and self.count(v):
            if as_vartype(vartype) != self.vartype(v):
                raise ValueError("inconsistent vartype")
            return v
        else:
            v = super()._append(v)
            self.vartypes.append(as_vartype(vartype))
            return v

    def _extend(self, *args, **kwargs):
        raise NotImplementedError

    def vartype(self, v: Variable):
        return self.vartypes[self.index(v)]


class ConstrainedQuadraticModel:
    def __init__(self):
        self.variables = TypedVariables()
        self.labels = Variables()
        self.constraints = {}

    @property
    def objective(self) -> BinaryQuadraticModel:
        """The objective to be minimized."""
        try:
            return self._objective
        except AttributeError:
            pass

        objective = BinaryQuadraticModel('BINARY')
        self._objective: BinaryQuadraticModel = objective
        return objective

    def add_constraint(self, data, *args, **kwargs):
        """A convenience wrapper for other methods that add constraints."""
        # in python 3.8+ we can use singledispatchmethod
        if isinstance(data, BinaryQuadraticModel):
            self.add_constraint_from_bqm(data, *args, **kwargs)
        elif isinstance(data, Comparison):
            self.add_constraint_from_comparison(data, *args, **kwargs)
        elif isinstance(data, Iterable):
            self.add_constraint_from_iterable(data, *args, **kwargs)
        else:
            raise NotImplementedError

    def add_constraint_from_bqm(self,
                                bqm: BinaryQuadraticModel,
                                sense: Union[Sense, str],
                                rhs: Bias = 0,
                                label: Optional[Hashable] = None,
                                copy: bool = True) -> Hashable:
        """Add a constraint from a binary quadratic model.

        Args:
            bqm: A binary quadratic model.

            sense: One of `<=', '>=', '=='.

            rhs: The right hand side of the constraint.

            label: A label for the constraint. Must be unique. If no label
                is provided, then one is generated using :mod:`uuid`.

            copy: If `True`, the BQM is copied. This can be set to `False` to
                improve performance, but subsequently mutating the bqm can
                cause issues.

        """
        variables = self.variables

        # get sense as an enum
        if isinstance(sense, str):
            sense = Sense(sense)

        if label is None:
            # we support up to 100k constraints and :6 gives us 16777216
            # possible so pretty safe
            label = uuid.uuid4().hex[:6]
            while label in self.constraints:
                label = uuid.uuid4().hex[:6]
        elif label in self.constraints:
            raise ValueError("a constraint with that label already exists")

        vartype = bqm.vartype
        for v in bqm.variables:
            if v in variables and variables.vartype(v) != vartype:
                raise ValueError(f"mismatch between variable {v!r}")

        # ok, everything checks out so let's add it
        for v in bqm.variables:
            variables._append(vartype, v, permissive=True)

        if copy:
            bqm = bqm.copy()

        if sense is Sense.Le:
            self.constraints[label] = Le(bqm, rhs)
        elif sense is Sense.Ge:
            self.constraints[label] = Ge(bqm, rhs)
        elif sense is Sense.Eq:
            self.constraints[label] = Eq(bqm, rhs)
        else:
            raise RuntimeError("unexpected sense")

        return label

    def add_constraint_from_comparison(self,
                                       comp: Comparison,
                                       label: Optional[Hashable] = None,
                                       copy: bool = True) -> Hashable:
        """Add a constraint from a comparison.

        Args:
            comp: A comparison object.

            label: A label for the constraint. Must be unique. If no label
                is provided, then one is generated using :mod:`uuid`.

            copy: If `True`, the model is copied. This can be set to `False` to
                improve performance, but subsequently mutating the model can
                cause issues.

        """
        if not isinstance(comp.lhs, BinaryQuadraticModel):
            raise TypeError("comparison should have a BQM lhs")
        if not isinstance(comp.rhs, Number):
            raise TypeError("comparison should have a numeric rhs")

        return self.add_constraint_from_bqm(comp.lhs, comp.sense, rhs=comp.rhs,
                                            label=label, copy=copy)

    def add_constraint_from_iterable(self, iterable: Iterable,
                                     sense: Union[Sense, str],
                                     rhs: Bias = 0,
                                     label: Optional[Hashable] = None,
                                     ) -> Hashable:
        """Add a constraint from an iterable of tuples.

        Args:
            iterable: An iterable of terms as tuples. The variables must
                have already been added to the object.

            sense: One of `<=', '>=', '=='.

            rhs: The right hand side of the constraint.

            label: A label for the constraint. Must be unique. If no label
                is provided, then one is generated using :mod:`uuid`.

        """
        # use quadratic model in the future
        qm = BinaryQuadraticModel('BINARY')
        for *variables, bias in iterable:
            if len(variables) == 0:
                qm.offset += bias
            elif len(variables) == 1:
                v, = variables
                if self.variables.vartype(v) != qm.vartype:
                    raise ValueError
                qm.add_linear(v, bias)
            elif len(variables) == 2:
                u, v = variables
                if self.variables.vartype(u) != qm.vartype:
                    raise ValueError
                if self.variables.vartype(v) != qm.vartype:
                    raise ValueError
                qm.add_quadratic(u, v, bias)
            else:
                raise ValueError("terms must be constant, linear or quadratic")

        # use quadratic model in the future
        return self.add_constraint_from_bqm(
            qm, sense, rhs=rhs, label=label, copy=False)

    def add_variable(self, v: Variable, vartype: VartypeLike):
        """Add a variable to the model."""
        if self.variables.count(v):
            if as_vartype(vartype) != self.variables.vartype(v):
                raise ValueError("given variable has already been added with a different vartype")
        else:
            return self.variables._append(vartype, v)

    @classmethod
    def from_file(cls, fp: Union[BinaryIO, ByteString]) -> "ConstrainedQuadraticModel":
        """Construct from a file-like object.

        The inverse of :meth:`~ConstrainedQuadraticModel.to_file`.
        """
        if isinstance(fp, ByteString):
            file_like: BinaryIO = _BytesIO(fp)  # type: ignore[assignment]
        else:
            file_like = fp

        header_info = read_header(file_like, CQM_MAGIC_PREFIX)

        if header_info.version >= (2, 0):
            raise ValueError("cannot load a BQM serialized with version "
                             f"{version!r}, try upgrading your dimod version")

        # we don't actually need the data

        cqm = CQM()

        with zipfile.ZipFile(file_like, mode='r') as zf:
            cqm.set_objective(load(zf.read("objective")))

            constraint_labels = set()
            for arch in zf.namelist():
                # even on windows zip uses /
                match = re.match("constraints/([^/]+)/", arch)
                if match is not None:
                    constraint_labels.add(match.group(1))

            for constraint in constraint_labels:
                lhs = load(zf.read(f"constraints/{constraint}/lhs"))
                rhs = np.frombuffer(zf.read(f"constraints/{constraint}/rhs"), np.float64)[0]
                sense = zf.read(f"constraints/{constraint}/sense").decode('ascii')
                label = deserialize_variable(json.loads(constraint))
                cqm.add_constraint(lhs, rhs=rhs, sense=sense, label=label)

        return cqm

    def num_biases(self) -> int:
        """The number of biases accross the objective and constraints."""
        num_biases = len(self.objective.linear) + len(self.objective.quadratic)
        num_biases += sum(len(const.lhs.linear) + len(const.lhs.quadratic)
                          for const in self.constraints.values())
        return num_biases

    def set_objective(self, bqm: BinaryQuadraticModel):
        """Set the objective of the constrained quadratic model."""
        variables = self.variables

        vartype = bqm.vartype
        for v in bqm.variables:
            if v in variables and variables.vartype(v) != vartype:
                raise ValueError(f"mismatch between variable {v!r}")

        # ok, everything checks out so let's add it
        for v in bqm.variables:
            variables._append(vartype, v, permissive=True)

        self._objective = bqm

    def to_file(self, *, spool_size: int = int(1e9)) -> tempfile.SpooledTemporaryFile:
        """Serialize to a file-like object.

        Args:
            spool_size: Defines the `max_size` passed to the constructor of
                :class:`tempfile.SpooledTemporaryFile`. Determines whether
                the returned file-like's contents will be kept on disk or in
                memory.

        Format Specification (Version 1.0):

            This format is inspired by the `NPY format`_

            The first 8 bytes are a magic string: exactly "DIMODCQM".

            The next 1 byte is an unsigned byte: the major version of the file
            format.

            The next 1 byte is an unsigned byte: the minor version of the file
            format.

            The next 4 bytes form a little-endian unsigned int, the length of
            the header data HEADER_LEN.

            The next HEADER_LEN bytes form the header data. This is a
            json-serialized dictionary. The dictionary is exactly:

            .. code-block:: python

                dict(num_variables=len(cqm.variables),
                     num_constraints=len(cqm.constraints),
                     num_biases=cqm.num_biases(),
                     )

            it is terminated by a newline character and padded with spaces to
            make the entire length of the entire header divisible by 64.

            The constraint quadratic model data comes after the header. It is
            encoded as a zip file. The zip file will contain one file
            named `objective`, containing the objective as encoded as a file
            view. It will also contain a directory called `constraints`. The
            `constraints` directory will contain one subdirectory for each
            constraint, each containing `lhs`, `rhs` and `sense` encoding
            the `lhs` as a fileview, the `rhs` as a float and the sense
            as a string.

        .. _NPY format: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html

        """
        file = SpooledTemporaryFile(max_size=spool_size)

        data = dict(num_variables=len(self.variables),
                    num_constraints=len(self.constraints),
                    num_biases=self.num_biases(),
                    )

        write_header(file, CQM_MAGIC_PREFIX, data, version=(1, 0))

        # write the values
        with zipfile.ZipFile(file, mode='a') as zf:
            try:
                zf.writestr(
                    'objective', self.objective.to_file(spool_size=int(1e12))._file.getbuffer())
            except AttributeError:
                # no objective to write
                pass

            for label, constraint in self.constraints.items():
                # put everything in a constraints/label/ directory
                lstr = json.dumps(serialize_variable(label))

                lhs = constraint.lhs.to_file(spool_size=int(1e12))._file.getbuffer()
                zf.writestr(f'constraints/{lstr}/lhs', lhs)

                rhs = np.float64(constraint.rhs).tobytes()
                zf.writestr(f'constraints/{lstr}/rhs', rhs)

                sense = bytes(constraint.sense.value, 'ascii')
                zf.writestr(f'constraints/{lstr}/sense', sense)

        file.seek(0)
        return file


CQM = ConstrainedQuadraticModel


# register fileview loader
load.register(CQM_MAGIC_PREFIX, ConstrainedQuadraticModel.from_file)
