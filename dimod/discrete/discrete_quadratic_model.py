# Copyright 2020 D-Wave Systems Inc.
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

import collections.abc as abc
import io
import json
import tempfile

from collections import namedtuple
from operator import eq

import numpy as np

from dimod.discrete.cydiscrete_quadratic_model import cyDiscreteQuadraticModel
from dimod.sampleset import as_samples
from dimod.serialization.fileview import VariablesSection, _BytesIO


__all__ = ['DiscreteQuadraticModel', 'DQM']


# constants for serialization
DQM_MAGIC_PREFIX = b'DIMODDQM'
DATA_MAGIC_PREFIX = b'BIAS'
VERSION = bytes([1, 0])  # version 1.0


# todo: update BinaryQuadraticModel.to_numpy_vectors to also use namedtuple
DQMVectors = namedtuple(
    'DQMVectors', ['case_starts', 'linear_biases', 'quadratic', 'labels'])
QuadraticVectors = namedtuple(
    'QuadraticVectors', ['row_indices', 'col_indices', 'biases'])


# this is the third(!) variables implementation in dimod. It differs from
# dimod.variables in that it stores its labels sparsely. It has the same
# behaviour as the cyBQM ones, except that it rolls the logic up into a
# object. These need to be unified.
class _Variables(abc.Sequence, abc.Set):
    def __init__(self):
        self._label_to_idx = dict()
        self._idx_to_label = dict()
        self.stop = 0

    def __contains__(self, v):
        return v in self._label_to_idx or (isinstance(v, int)
                                           and 0 <= v < self.stop
                                           and v not in self._idx_to_label)

    def __eq__(self, other):
        if isinstance(other, abc.Sequence):
            return len(self) == len(other) and all(map(eq, self, other))
        elif isinstance(other, abc.Set):
            return not (self ^ other)
        else:
            return False

    def __getitem__(self, idx):

        if not isinstance(idx, int):
            raise TypeError("index must be an integer.")

        given = idx  # for error message

        # handle negative indexing
        if idx < 0:
            idx = self.stop + idx

        if idx >= self.stop:
            raise IndexError('index {} out of range'.format(given))

        return self._idx_to_label.get(idx, idx)

    def __len__(self):
        return self.stop

    def __ne__(self, other):
        return not (self == other)

    @property
    def is_range(self):
        return not self._label_to_idx

    def _append(self, v=None):
        """Append a new variable."""

        if v is None:
            # handle the easy case
            if self.is_range:
                self.stop += 1
                return

            raise NotImplementedError

        elif v in self:
            raise ValueError('{!r} is already a variable'.format(v))

        idx = self.stop

        if idx != v:
            self._label_to_idx[v] = idx
            self._idx_to_label[idx] = v

        self.stop += 1

        return

    def index(self, v):
        # todo: support start and end like list.index
        if v not in self:
            raise ValueError('unknown variable {!r}'.format(v))
        return self._label_to_idx.get(v, v)


class DiscreteQuadraticModel:
    """Encodes a discrete quadratic model.

    A discrete quadratic model is a polynomial over discrete variables with
    terms all of degree two or less.

    Examples:

        This example constructs a map coloring with Canadian provinces. To
        solve the problem we penalize adjacent provinces having the same color.

        >>> provinces = ["AB", "BC", "ON", "MB", "NB", "NL", "NS", "NT", "NU",
        ...              "PE", "QC", "SK", "YT"]
        >>> borders = [("BC", "AB"), ("BC", "NT"), ("BC", "YT"), ("AB", "SK"),
        ...            ("AB", "NT"), ("SK", "MB"), ("SK", "NT"), ("MB", "ON"),
        ...            ("MB", "NU"), ("ON", "QC"), ("QC", "NB"), ("QC", "NL"),
        ...            ("NB", "NS"), ("YT", "NT"), ("NT", "NU")]
        >>> colors = [0, 1, 2, 3]
        ...
        >>> dqm = dimod.DiscreteQuadraticModel()
        >>> for p in provinces:
        ...     _ = dqm.add_variable(4, label=p)
        >>> for p0, p1 in borders:
        ...     dqm.set_quadratic(p0, p1, {(c, c): 1 for c in colors})

        The next examples show how to view and manipulate the model biases.

        >>> dqm = dimod.DiscreteQuadraticModel()

        Add the variables to the model

        >>> u = dqm.add_variable(5)  # unlabeled variable with 5 cases
        >>> v = dqm.add_variable(3, label='v')  # labeled variable with 3 cases

        The linear biases default to 0. They can be read by case or by batch.

        >>> dqm.get_linear_case(u, 1)
        0.0
        >>> dqm.get_linear(u)
        array([0., 0., 0., 0., 0.])
        >>> dqm.get_linear(v)
        array([0., 0., 0.])

        The linear biases can be overwritten either by case or in a batch.

        >>> dqm.set_linear_case(u, 3, 17)
        >>> dqm.get_linear(u)
        array([ 0.,  0.,  0., 17.,  0.])
        >>> dqm.set_linear(v, [0, -1, 3])
        >>> dqm.get_linear(v)
        array([ 0., -1.,  3.])

        The quadratic biases can also be manipulated sparsely or densely.

        >>> dqm.set_quadratic(u, v, {(0, 2): 1.5})
        >>> dqm.get_quadratic(u, v)
        {(0, 2): 1.5}
        >>> dqm.get_quadratic(u, v, array=True)  # as a NumPy array
        array([[0. , 0. , 1.5],
               [0. , 0. , 0. ],
               [0. , 0. , 0. ],
               [0. , 0. , 0. ],
               [0. , 0. , 0. ]])
        >>> dqm.set_quadratic_case(u, 2, v, 1, -3)
        >>> dqm.get_quadratic(u, v, array=True)
        array([[ 0. ,  0. ,  1.5],
               [ 0. ,  0. ,  0. ],
               [ 0. , -3. ,  0. ],
               [ 0. ,  0. ,  0. ],
               [ 0. ,  0. ,  0. ]])
        >>> dqm.get_quadratic(u, v)  # doctest:+SKIP
        {(0, 2): 1.5, (2, 1): -3.0}

    """

    def __init__(self):
        self.variables = _Variables()
        self._cydqm = cyDiscreteQuadraticModel()

    @property
    def adj(self):
        """dict[hashable, set]: The adjacency structure of the variables."""
        return dict((self.variables[ui],
                     set(self.variables[vi] for vi in neighborhood))
                    for ui, neighborhood in enumerate(self._cydqm.adj))

    def add_variable(self, num_cases, label=None):
        """Add a discrete variable.

        Args:
            num_cases (int):
                The number of cases in the variable. Must be a positive
                integer.

            label (hashable, optional):
                A label for the variable. Can be any hashable except `None`.
                Defaults to the length of the discrete quadratic model, if that
                label is available. Otherwise defaults to the lowest available
                positive integer label.

        Returns:
            The label of the new variable.

        Raises:
            ValueError: If `label` already exists as a variable label.
            TypeError: If `label` is not hashable.

        """
        self.variables._append(label)
        variable_index = self._cydqm.add_variable(num_cases)
        assert variable_index + 1 == len(self.variables)
        return self.variables[-1]

    def energy(self, sample):
        energy, = self.energies(sample)
        return energy

    def energies(self, samples):
        samples, labels = as_samples(samples, dtype=self._cydqm.case_dtype)

        # reorder as needed
        if len(labels) != self.num_variables():
            raise ValueError(
                "Given sample(s) have incorrect number of variables")
        if self.variables != labels:
            # todo as part of discrete sampleset work
            raise NotImplementedError

        return np.asarray(self._cydqm.energies(samples))

    @classmethod
    def _from_file_numpy(cls, file_like):

        magic = file_like.read(len(DATA_MAGIC_PREFIX))
        if magic != DATA_MAGIC_PREFIX:
            raise ValueError("unknown file type, expected magic string {} but "
                             "got {}".format(DATA_MAGIC_PREFIX, magic))

        length = np.frombuffer(file_like.read(4), '<u4')[0]
        start = file_like.tell()

        data = np.load(file_like)

        obj = cls.from_numpy_vectors(data['case_starts'],
                                     data['linear_biases'],
                                     (data['quadratic_row_indices'],
                                      data['quadratic_col_indices'],
                                      data['quadratic_biases'],
                                      )
                                     )

        # move to the end of the data section
        file_like.seek(start+length, io.SEEK_SET)

        return obj

    @classmethod
    def from_file(cls, file_like):
        """Construct a DQM from a file-like object.

        The inverse of :meth:`~DiscreteQuadraticModel.to_file`.
        """

        if isinstance(file_like, (bytes, bytearray, memoryview)):
            fp = _BytesIO(fp)

        magic = file_like.read(len(DQM_MAGIC_PREFIX))
        if magic != DQM_MAGIC_PREFIX:
            raise ValueError("unknown file type, expected magic string {} but "
                             "got {}".format(DQM_MAGIC_PREFIX, magic))

        version = tuple(file_like.read(2))
        if version[0] != 1:
            raise ValueError("cannot load a DQM serialized with version {!r}, "
                             "try upgrading your dimod version"
                             "".format(version))

        header_len = np.frombuffer(file_like.read(4), '<u4')[0]

        header_data = json.loads(file_like.read(header_len).decode('ascii'))

        obj = cls._from_file_numpy(file_like)

        if header_data['variables']:
            obj.variables = _Variables()
            for v in VariablesSection.load(file_like):
                obj.variables._append(v)

            if len(obj.variables) != obj.num_variables():
                raise ValueError("mismatched labels to BQM in given file")

        return obj

    @classmethod
    def from_numpy_vectors(cls, case_starts, linear_biases, quadratic,
                           labels=None):
        """Construct a DQM from five numpy vectors.

        Args:
            case_starts (array-like): A length
                :meth:`~DiscreteQuadraticModel.num_variables` array. The cases
                associated with variable `v` are in the range `[case_starts[v],
                cases_starts[v+1])`.

            linear_biases (array-like): A length
              :meth:`~DiscreteQuadraticModel.num_cases` array. The linear
              biases.

            quadratic (tuple): A three tuple containing:

                - `irow`: A length
                  :meth:`~DiscreteQuadraticModel.num_interactions` array. If
                  the case interactions were defined in a sparse matrix, these
                  would be the row indices.
                - `icol`: A length
                  :meth:`~DiscreteQuadraticModel.num_interactions` array. If
                  the case interactions were defined in a sparse matrix, these
                  would be the column indices.
                - `quadratic_biases`: A length
                  :meth:`~DiscreteQuadraticModel.num_interactions` array. If
                  the case interactions were defined in a sparse matrix, these
                  would be the values.

            labels (list, optional):
                The variable labels. Defaults to index-labeled.

        Example:

            >>> dqm = dimod.DiscreteQuadraticModel()
            >>> u = dqm.add_variable(5)
            >>> v = dqm.add_variable(3, label='3var')
            >>> dqm.set_quadratic(u, v, {(0, 2): 1})
            >>> vectors = dqm.to_numpy_vectors()
            >>> new = dimod.DiscreteQuadraticModel.from_numpy_vectors(*vectors)

        See Also:
            :meth:`~DiscreteQuadraticModel.to_numpy_vectors`

        """

        obj = cls()

        obj._cydqm = cyDiscreteQuadraticModel.from_numpy_vectors(
            case_starts, linear_biases, quadratic)

        if labels is not None:
            if len(labels) != obj._cydqm.num_variables():
                raise ValueError(
                    "labels does not match the length of the DQM"
                    )

            for v in labels:
                obj.variables._append(v)
        else:
            for v in range(obj._cydqm.num_variables()):
                obj.variables._append()

        return obj

    def get_linear(self, v):
        """The linear biases associated with variable `v`.

        Args:
            v: A variable in the discrete quadratic model.

        Returns:
            :class:`~numpy.ndarray`: The linear biases in an array.

        """
        return self._cydqm.get_linear(self.variables.index(v))

    def get_linear_case(self, v, case):
        """The linear bias associated with case `case` of variable `v`.

        Args:
            v: A variable in the discrete quadratic model.

            case (int): The case of `v`.

        Returns:
            The linear bias.

        """
        return self._cydqm.get_linear_case(self.variables.index(v), case)

    def get_quadratic(self, u, v, array=False):
        """The biases associated with the interaction between `u` and `v`.

        Args:
            u: A variable in the discrete quadratic model.

            v: A variable in the discrete quadratic model.

            array (bool, optional, default=False): If True, a dense array is
            returned rather than a dict.

        Returns:
            The quadratic biases. If `array=False`, returns a dictionary of the
            form `{case_u, case_v: bias, ...}`
            If `array=True`, returns a
            :meth:`~DiscreteQuadraticModel.num_cases(u)` by
            :meth:`~DiscreteQuadraticModel.num_cases(v)` numpy array.

        """
        return self._cydqm.get_quadratic(
            self.variables.index(u),
            self.variables.index(v),
            array=array)

    def get_quadratic_case(self, u, u_case, v, v_case):
        """The bias associated with the interaction between two cases of `u`
        and `v`.

        Args:
            u: A variable in the discrete quadratic model.

            u_case (int): The case of `u`.

            v: A variable in the discrete quadratic model.

            v_case (int): The case of `v`.

        Returns:
            The quadratic bias.

        """
        return self._cydqm.get_quadratic_case(
            self.variables.index(u), u_case, self.variables.index(v), v_case)

    def num_cases(self, v=None):
        """If v is provided, the number of cases associated with v, otherwise
        the total number of cases in the DQM.
        """
        if v is None:
            return self._cydqm.num_cases()
        return self._cydqm.num_cases(self.variables.index(v))

    def num_case_interactions(self):
        """The total number of case interactions."""
        return self._cydqm.num_case_interactions()

    def num_variable_interactions(self):
        """The total number of variable interactions"""
        return self._cydqm.num_variable_interactions()

    def num_variables(self):
        """The number of variables in the discrete quadratic model."""
        return self._cydqm.num_variables()

    def set_linear(self, v, biases):
        """Set the linear biases associated with `v`.

        Args:
            v: A variable in the discrete quadratic model.

            biases (array-like): The linear biases in an array.

        """
        biases = np.asarray(biases, dtype=self._cydqm.dtype)
        self._cydqm.set_linear(self.variables.index(v), biases)

    def set_linear_case(self, v, case, bias):
        """The linear bias associated with case `case` of variable `v`.

        Args:
            v: A variable in the discrete quadratic model.

            case (int): The case of `v`.

            bias (float): The linear bias.

        """
        self._cydqm.set_linear_case(self.variables.index(v), case, bias)

    def set_quadratic(self, u, v, biases):
        """Set biases associated with the interaction between `u` and `v`.

        Args:
            u: A variable in the discrete quadratic model.

            v: A variable in the discrete quadratic model.

            biases (array-like/dict):
                The quadratic biases. If a dict, then a dictionary of the
                form `{case_u, case_v: bias, ...}`. Otherwise, then should be,
                a :meth:`~DiscreteQuadraticModel.num_cases(u)` by
                :meth:`~DiscreteQuadraticModel.num_cases(v)` array-like.

        """
        self._cydqm.set_quadratic(
            self.variables.index(u),
            self.variables.index(v),
            biases)

    def set_quadratic_case(self, u, u_case, v, v_case, bias):
        """Set the bias associated with the interaction between two cases of
        `u` and `v`.

        Args:
            u: A variable in the discrete quadratic model.

            u_case (int): The case of `u`.

            v: A variable in the discrete quadratic model.

            v_case (int): The case of `v`.

            bias (float): The quadratic bias.

        """
        self._cydqm.set_quadratic_case(
            self.variables.index(u), u_case,
            self.variables.index(v), v_case,
            bias)

    def _to_file_numpy(self, file, compressed):
        # the biases etc, saved using numpy

        # we'd like to just let numpy handle the header etc, but it doesn't
        # do a good job of cleaning up after itself in np.load, so we record
        # the section length ourselves
        file.write(DATA_MAGIC_PREFIX)
        file.write(b'    ')  # will be replaced by the length
        start = file.tell()

        vectors = self.to_numpy_vectors()

        if compressed:
            save = np.savez_compressed
        else:
            save = np.savez

        save(file,
             case_starts=vectors.case_starts,
             linear_biases=vectors.linear_biases,
             quadratic_row_indices=vectors.quadratic.row_indices,
             quadratic_col_indices=vectors.quadratic.col_indices,
             quadratic_biases=vectors.quadratic.biases,
             )

        # record the length
        end = file.tell()
        file.seek(start-4)
        file.write(np.dtype('<u4').type(end - start).tobytes())
        file.seek(end)

    def to_file(self, compressed=False, ignore_labels=False,
                spool_size=int(1e9)):
        """Convert the DQM to a file-like object.

        Args:
            compressed (bool, optional default=False):
                If True, most of the data will be compressed.

            ignore_labels (bool, optional, default=False):
                Treat the DQM as unlabeled. This is useful for large DQMs to
                save on space.

            spool_size (int, optional, default=int(1e9)):
                Defines the `max_size` passed to the constructor of
                :class:`tempfile.SpooledTemporaryFile`. Determines whether
                the returned file-like's contents will be kept on disk or in
                memory.

        Returns:
            :class:`tempfile.SpooledTemporaryFile`: A file-like object
            that can be used to construct a copy of the DQM.

        Format Specification (Version 1.0):

            This format is inspired by the `NPY format`_

            **Header**

            The first 8 bytes are a magic string: exactly `"DIMODDQM"`.

            The next 1 byte is an unsigned byte: the major version of the file
            format.

            The next 1 byte is an unsigned byte: the minor version of the file
            format.

            The next 4 bytes form a little-endian unsigned int, the length of
            the header data `HEADER_LEN`.

            The next `HEADER_LEN` bytes form the header data. This is a
            json-serialized dictionary. The dictionary is exactly:

            .. code-block:: python

                dict(num_variables=dqm.num_variables(),
                     num_cases=dqm.num_cases(),
                     num_case_interactions=dqm.num_case_interactions(),
                     num_variable_interactions=dqm.num_variable_interactions(),
                     variables=not (ignore_labels or dqm.variables.is_range),
                     )

            it is padded with spaces to make the entire length of the header
            divisible by 64.

            **DQM Data**

            The first 4 bytes are exactly `"BIAS"`

            The next 4 bytes form a little-endian unsigned int, the length of
            the DQM data `DATA_LEN`.

            The next `DATA_LEN` bytes are the vectors as returned by
            :meth:`DiscreteQuadraticModel.to_numpy_vectors` saved using
            :func:`numpy.save`.

            **Variable Data**

            The first 4 bytes are exactly "VARS".

            The next 4 bytes form a little-endian unsigned int, the length of
            the variables array `VARIABLES_LENGTH`.

            The next VARIABLES_LENGTH bytes are a json-serialized array. As
            constructed by `json.dumps(list(bqm.variables)).

        .. _NPY format: https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html

        See Also:
            :meth:`DiscreteQuadraticModel.from_file`

        """

        file = tempfile.SpooledTemporaryFile(max_size=spool_size)

        # attach the header
        header_parts = [DQM_MAGIC_PREFIX,
                        VERSION,
                        bytes(4),  # placeholder for HEADER_LEN
                        ]

        index_labeled = ignore_labels or self.variables.is_range

        header_data = json.dumps(
            dict(num_variables=self.num_variables(),
                 num_cases=self.num_cases(),
                 num_case_interactions=self.num_case_interactions(),
                 num_variable_interactions=self.num_variable_interactions(),
                 variables=not index_labeled,
                 ),
            sort_keys=True).encode('ascii')

        header_parts.append(header_data)

        # make the entire header length divisible by 64
        length = sum(len(part) for part in header_parts)
        if length % 64:
            padding = b' '*(64 - length % 64)
        else:
            padding = b''
        header_parts.append(padding)

        HEADER_LEN = len(padding) + len(header_data)
        header_parts[2] = np.dtype('<u4').type(HEADER_LEN).tobytes()

        for part in header_parts:
            file.write(part)

        # the section containing most of the data, encoded with numpy
        self._to_file_numpy(file, compressed)

        if not index_labeled:
            file.write(VariablesSection(self.variables).dumps())

        file.seek(0)

        return file

    def to_numpy_vectors(self):
        """Convert the DQM to five numpy vectors and the labels.

        Returns:
            :class:`DQMVectors`: A named tuple with fields `['case_starts',
            'linear_biases', 'quadratic', 'labels'].

            - `case_starts`: A length
              :meth:`~DiscreteQuadraticModel.num_variables` array. The cases
              associated with variable `v` are in the range `[case_starts[v],
              cases_starts[v+1])`.
            - `linear_biases`: A length
              :meth:`~DiscreteQuadraticModel.num_cases` array. The linear
              biases.
            - `quadratic`: A named tuple with fields `['row_indices',
              'col_indices', 'biases']`.

              * `row_indices`: A length
                :meth:`~DiscreteQuadraticModel.num_case_interactions` array. If
                the case interactions were defined in a sparse matrix, these
                would be the row indices.

              * `col_indices`: A length
                :meth:`~DiscreteQuadraticModel.num_case_interactions` array. If
                the case interactions were defined in a sparse matrix, these
                would be the column indices.

              * `biases`: A length
                :meth:`~DiscreteQuadraticModel.num_case_interactions` array. If
                the case interactions were defined in a sparse matrix, these
                would be the values.

            - `labels`: The variable labels in a
              :class:`~collections.abc.Sequence`.


            If `return_labels=True`, this method will instead return a tuple
            `(case_starts, linear_biases, (irow, icol, qdata), labels)` where
            `labels` is a list of the variable labels.

        See Also:
            :meth:`~DiscreteQuadraticModel.from_numpy_vectors`

        """
        case_starts, linear_biases, quadratic = self._cydqm.to_numpy_vectors()

        return DQMVectors(case_starts,
                          linear_biases,
                          QuadraticVectors(*quadratic),
                          self.variables,
                          )


DQM = DiscreteQuadraticModel  # alias
