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
import warnings

from collections import defaultdict, namedtuple
from typing import List, Tuple, Union, Generator, Iterator

import numpy as np

from dimod.discrete.cydiscrete_quadratic_model import cyDiscreteQuadraticModel
from dimod.sampleset import as_samples
from dimod.serialization.fileview import VariablesSection, _BytesIO, SpooledTemporaryFile
from dimod.serialization.fileview import load, read_header, write_header
from dimod.typing import QuadraticVectors, DQMVectors
from dimod.variables import Variables


LinearTriplets = Union[List[Tuple], Generator[Tuple, None, None]]


__all__ = ['DiscreteQuadraticModel', 'DQM', 'CaseLabelDQM']


# constants for serialization
DQM_MAGIC_PREFIX = b'DIMODDQM'
DATA_MAGIC_PREFIX = b'BIAS'


LegacyDQMVectors = namedtuple(
    'LegacyDQMVectors', ['case_starts', 'linear_biases', 'quadratic', 'labels'])


class VariableNeighborhood(abc.Set):
    # this really shouldn't be set-like because __contains__ is O(degree(v))
    # but for backwards compatiblity we'll leave it.
    __slots__ = ('_dqm', '_vi')

    def __init__(self, dqm, v):
        self._dqm = dqm
        self._vi = dqm.variables.index(v)  # raises ValueError

    def __contains__(self, u):
        return self._dqm.variables.index(u) in self._dqm._cydqm.adj[self._vi]

    def __iter__(self):
        for ui in self._dqm._cydqm.adj[self._vi]:
            yield self._dqm.variables[ui]

    def __len__(self):
        return self._dqm._cydqm.degree(self._vi)

    def __repr__(self):
        return str(dict(self))


class VariableAdjacency(abc.Mapping):
    __slots__ = ('_dqm',)

    def __init__(self, dqm):
        self._dqm = dqm

    def __getitem__(self, v):
        return VariableNeighborhood(self._dqm, v)

    def __iter__(self):
        yield from self._dqm.variables

    def __len__(self):
        return len(self._dqm.variables)

    def __repr__(self):
        return str(dict(self))


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
        self.variables = Variables()
        self._cydqm = cyDiscreteQuadraticModel()

    variables = None  # overwritten by __init__, here for the docstring
    """:class:`~.variables.Variables` of variable labels."""

    @property
    def adj(self):
        """dict[hashable, set]: The adjacency structure of the variables."""
        try:
            return self._adj
        except AttributeError:
            pass
        self._adj = adj = VariableAdjacency(self)
        return adj

    @property
    def offset(self):
        return self._cydqm.offset

    @offset.setter
    def offset(self, offset: float):
        self._cydqm.offset = offset

    def add_linear_equality_constraint(self, terms: LinearTriplets,
                                       lagrange_multiplier: float,
                                       constant: float):
        r"""Add a linear constraint as a quadratic objective.

        Adds a linear constraint of the form
        :math:`\sum_{i,k} a_{i,k} x_{i,k} + C = 0`
        to the discrete quadratic model as a quadratic objective.

        Args:
            terms: A list of tuples of the type (variable, case, bias).
                Each tuple is evaluated to the term (bias * variable_case).
                All terms in the list are summed.
            lagrange_multiplier: The coefficient or the penalty strength
            constant: The constant value of the constraint.

        """
        index_terms = ((self.variables.index(v), c, x) for v, c, x in terms)
        self._cydqm.add_linear_equality_constraint(
            index_terms, lagrange_multiplier, constant)

    def add_linear_inequality_constraint(self, terms: LinearTriplets,
                                         lagrange_multiplier: float,
                                         label: str,
                                         constant: int = 0,
                                         lb: int = np.iinfo(np.int64).min,
                                         ub: int = 0,
                                         slack_method: str = "log2",
                                         cross_zero: bool = False)\
            -> LinearTriplets:

        r"""Add a linear inequality constraint as a quadratic objective.

        Adds a linear inequality constraint of the form:

        math:'lb <= \sum_{i,k} a_{i,k} x_{i,k} + constant <= ub'
        to the discrete quadratic model as a quadratic objective.
        Coefficients should be integers.
        For constraints with fractional coefficients, multiply both sides of
        the inequality by an appropriate factor of ten to attain or approximate
        integer coefficients.

        Args:
            terms:
                A list of tuples of the type (variable, case, bias).
                Each tuple is evaluated to the term (bias * variable_case).
                All terms in the list are summed.
            lagrange_multiplier:
                A weight or the penalty strength. This value is multiplied by
                the entire constraint objective and added to the
                discrete quadratic model (it doesn't appear explicitly in the
+               equation above).
            label:
                Prefix used to label the slack variables used to create the new
                objective.
            constant:
                The constant value of the constraint.
            lb:
                lower bound for the constraint
            ub:
                upper bound for the constraint
            slack_method:
                "The method for adding slack variables. Supported methods are:
                - log2: Adds up to log2(ub - lb) number of dqm variables each
                        with two cases to the constraint.
                - log10: Adds log10 dqm variables each with up to 10 cases.
                - linear: Adds one dqm variable for each constraint with linear
                          number of cases.
            cross_zero:
                 When True, adds zero to the domain of constraint

        Returns:
            slack_terms:  A list of tuples of the type (variable, case, bias)
                for the new slack variables.
                Each tuple is evaluated to the term (bias * variable_case).
                All terms in the list are summed.

       """

        if slack_method not in ['log2', 'log10', 'linear']:
            raise ValueError(
                "expected slack_method to be 'log2', 'log10' or 'linear' "
                f"but got {slack_method!r}")

        if isinstance(terms, Iterator):
            terms = list(terms)
        if int(constant) != constant or int(lb) != lb or int(ub) != ub or any(
                int(bias) != bias for _, _, bias in terms):
            warnings.warn("For constraints with fractional coefficients, "
                          "multiply both sides of the inequality by an "
                          "appropriate factor of ten to attain or "
                          "approximate integer coefficients. ")

        terms_upper_bound = sum(v for _, _, v in terms if v > 0)
        terms_lower_bound = sum(v for _, _, v in terms if v < 0)
        ub_c = min(terms_upper_bound, ub - constant)
        lb_c = max(terms_lower_bound, lb - constant)

        if terms_upper_bound <= ub_c and terms_lower_bound >= lb_c:
            warnings.warn(
                f'Did not add constraint {label}.'
                ' This constraint is feasible'
                ' with any value for state variables.')
            return []

        if ub_c < lb_c:
            raise ValueError(
                f'The given constraint ({label}) is infeasible with any value'
                ' for state variables.')

        slack_upper_bound = int(ub_c - lb_c)
        if slack_upper_bound == 0:
            self.add_linear_equality_constraint(terms, lagrange_multiplier,
                                                -ub_c)
            return []
        else:
            slack_terms = []
            zero_constraint = False
            if cross_zero:
                if lb_c > 0 or ub_c < 0:
                    zero_constraint = True

            if slack_method == "log2":
                num_slack = int(np.floor(np.log2(slack_upper_bound)))
                slack_coefficients = [2 ** j for j in range(num_slack)]
                if slack_upper_bound - 2 ** num_slack >= 0:
                    slack_coefficients.append(
                        slack_upper_bound - 2 ** num_slack + 1)

                for j, s in enumerate(slack_coefficients):
                    sv = self.add_variable(2, f'slack_{label}_{j}')
                    slack_terms.append((sv, 1, s))

                if zero_constraint:
                    sv = self.add_variable(2, f'slack_{label}_{num_slack + 1}')
                    slack_terms.append((sv, 1, ub_c))

            elif slack_method == "log10":
                num_dqm_vars = int(np.ceil(np.log10(slack_upper_bound+1)))
                for j in range(num_dqm_vars):
                    slack_term = list(range(0, min(slack_upper_bound + 1,
                                                   10 ** (j + 1)), 10 ** j))[1:]
                    if j < num_dqm_vars - 1 or not zero_constraint:
                        sv = self.add_variable(len(slack_term) + 1,
                                               f'slack_{label}_{j}')
                    else:
                        sv = self.add_variable(len(slack_term) + 2,
                                               f'slack_{label}_{j}')
                    for i, val in enumerate(slack_term):
                        slack_terms.append((sv, i + 1, val))
                if zero_constraint:
                    slack_terms.append((sv, len(slack_term) + 1, ub_c))
            elif slack_method == 'linear':
                slack_term = list(range(1, slack_upper_bound + 1))
                if not zero_constraint:
                    sv = self.add_variable(len(slack_term) + 1,
                                           f'slack_{label}')
                else:
                    sv = self.add_variable(len(slack_term) + 2,
                                           f'slack_{label}')
                for i, val in enumerate(slack_term):
                    slack_terms.append((sv, i + 1, val))
                if zero_constraint:
                    slack_terms.append((sv, len(slack_term) + 1, ub_c))

            self.add_linear_equality_constraint(terms + slack_terms,
                                                lagrange_multiplier, -ub_c)
            return slack_terms

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

    # todo: support __copy__ and __deepcopy__
    def copy(self):
        """Return a copy of the discrete quadratic model."""
        new = type(self)()
        new._cydqm = self._cydqm.copy()
        for v in self.variables:
            new.variables._append(v)
        return new

    def degree(self, v):
        return self._cydqm.degree(self.variables.index(v))

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
            # need to reorder the samples
            label_to_idx = dict((v, i) for i, v in enumerate(labels))

            try:
                order = [label_to_idx[v] for v in self.variables]
            except KeyError:
                raise ValueError("given samples-like does not match labels")

            samples = samples[:, order]

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
                                      ),
                                     offset=data.get('offset', 0),
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
            file_like = _BytesIO(file_like)

        header_info = read_header(file_like, DQM_MAGIC_PREFIX)
        version = header_info.version
        header_data = header_info.data

        if version >= (2, 0):
            raise ValueError("cannot load a DQM serialized with version "
                             f"{version!r}, try upgrading your dimod version")

        obj = cls._from_file_numpy(file_like)

        if header_data['variables']:
            obj.variables = Variables()
            for v in VariablesSection.load(file_like):
                obj.variables._append(v)

            if len(obj.variables) != obj.num_variables():
                raise ValueError("mismatched labels to BQM in given file")

        return obj

    @classmethod
    def from_numpy_vectors(cls, case_starts, linear_biases, quadratic,
                           labels=None, offset=0):
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
                  :meth:`~DiscreteQuadraticModel.num_case_interactions` array. If
                  the case interactions were defined in a sparse matrix, these
                  would be the row indices.
                - `icol`: A length
                  :meth:`~DiscreteQuadraticModel.num_case_interactions` array. If
                  the case interactions were defined in a sparse matrix, these
                  would be the column indices.
                - `quadratic_biases`: A length
                  :meth:`~DiscreteQuadraticModel.num_case_interactions` array. If
                  the case interactions were defined in a sparse matrix, these
                  would be the values.

            labels (list, optional):
                The variable labels. Defaults to index-labeled.

            offset (float):
                Energy offset of the DQM.

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
            case_starts, linear_biases, quadratic, offset)

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

    def get_cases(self, v):
        """The cases of variable `v` as a sequence"""
        return range(self.num_cases(v))

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

    def relabel_variables(self, mapping, inplace=True):
        if not inplace:
            return self.copy().relabel_variables(mapping, inplace=True)
        self.variables._relabel(mapping)
        return self

    def relabel_variables_as_integers(self, inplace=True):
        """Relabel the variables of the DQM to integers.

        Args:
            inplace (bool, optional, default=True):
                If True, the discrete quadratic model is updated in-place;
                otherwise, a new discrete quadratic model is returned.

        Returns:
            tuple: A 2-tuple containing:

                A discrete quadratic model with the variables relabeled. If
                `inplace` is set to True, returns itself.

                dict: The mapping that will restore the original labels.

        """
        if not inplace:
            return self.copy().relabel_variables_as_integers(inplace=True)
        return self, self.variables._relabel_as_integers()

    def set_linear(self, v, biases):
        """Set the linear biases associated with `v`.

        Args:
            v: A variable in the discrete quadratic model.

            biases (array-like): The linear biases in an array.

        """
        biases = np.asarray(biases)

        # handle unsigned
        if np.issubdtype(biases.dtype, np.unsignedinteger):
            biases = np.asarray(biases, dtype=np.int64)

        self._cydqm.set_linear(self.variables.index(v), np.asarray(biases))

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

    def _to_file_numpy(self, file, compress):
        # the biases etc, saved using numpy

        # we'd like to just let numpy handle the header etc, but it doesn't
        # do a good job of cleaning up after itself in np.load, so we record
        # the section length ourselves
        file.write(DATA_MAGIC_PREFIX)
        file.write(b'    ')  # will be replaced by the length
        start = file.tell()

        vectors = self.to_numpy_vectors(return_offset=True)

        if compress:
            save = np.savez_compressed
        else:
            save = np.savez

        save(file,
            case_starts=vectors.case_starts,
            linear_biases=vectors.linear_biases,
            quadratic_row_indices=vectors.quadratic.row_indices,
            quadratic_col_indices=vectors.quadratic.col_indices,
            quadratic_biases=vectors.quadratic.biases,
            offset=vectors.offset,
            )

        # record the length
        end = file.tell()
        file.seek(start-4)
        file.write(np.dtype('<u4').type(end - start).tobytes())
        file.seek(end)

    def to_file(self, *, compress=False, compressed=None, ignore_labels=False,
                spool_size=int(1e9)):
        """Convert the DQM to a file-like object.

        Args:
            compress (bool, optional default=False):
                If True, most of the data will be compressed.

            compressed (bool, optional default=None):
                Deprecated; please use ``compress`` instead.

            ignore_labels (bool, optional, default=False):
                Treat the DQM as unlabeled. This is useful for large DQMs to
                save on space.

            spool_size (int, optional, default=int(1e9)):
                Defines the `max_size` passed to the constructor of
                :class:`tempfile.SpooledTemporaryFile`. Determines whether
                the returned file-like's contents will be kept on disk or in
                memory.

        Returns:
            A file-like object that can be used to construct a copy of the DQM.
            The class is a thin wrapper of
            :class:`tempfile.SpooledTemporaryFile` that includes some
            methods from :class:`io.IOBase`

        Format Specification (Version 1.0):

            This format is inspired by the `NPY format`_

            **Header**

            The first 8 bytes are a magic string: exactly ``"DIMODDQM"``.

            The next 1 byte is an unsigned byte: the major version of the file
            format.

            The next 1 byte is an unsigned byte: the minor version of the file
            format.

            The next 4 bytes form a little-endian unsigned int, the length of
            the header data `HEADER_LEN`.

            The next ``HEADER_LEN`` bytes form the header data. This is a
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
            the DQM data ``DATA_LEN``.

            The next ``DATA_LEN`` bytes are the vectors as returned by
            :meth:`DiscreteQuadraticModel.to_numpy_vectors` saved using
            :func:`numpy.save`.

            **Variable Data**

            The first 4 bytes are exactly ``"VARS"``.

            The next 4 bytes form a little-endian unsigned int, the length of
            the variables array ``VARIABLES_LENGTH``.

            The next VARIABLES_LENGTH bytes are a json-serialized array. As
            constructed by ``json.dumps(list(bqm.variables))``.

        .. _NPY format: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html

        See Also:
            :meth:`DiscreteQuadraticModel.from_file`

        .. deprecated:: 0.9.9

            The ``compressed`` keyword argument will be removed in dimod 0.12.0.
            Use ``compress`` instead.

        """

        file = SpooledTemporaryFile(max_size=spool_size)

        index_labeled = ignore_labels or self.variables.is_range

        data = dict(num_variables=self.num_variables(),
                    num_cases=self.num_cases(),
                    num_case_interactions=self.num_case_interactions(),
                    num_variable_interactions=self.num_variable_interactions(),
                    variables=not index_labeled,
                    )

        write_header(file, DQM_MAGIC_PREFIX, data, version=(1, 1))

        # the section containing most of the data, encoded with numpy
        if compressed is not None:
            warnings.warn(
                "Argument 'compressed' is deprecated since dimod 0.9.9 "
                "and will be removed in 0.12.0. "
                "Use 'compress' instead.",
                DeprecationWarning, stacklevel=2
                )
            compress = compressed or compress

        self._to_file_numpy(file, compress)

        if not index_labeled:
            file.write(VariablesSection(self.variables).dumps())

        file.seek(0)

        return file

    def to_numpy_vectors(self, return_offset: bool = False):
        """Convert the DQM to five numpy vectors and the labels.

        Args:
            return_offset: Boolean flag to optionally return energy offset value.

        Returns:
            :class:`DQMVectors`: A named tuple with fields `['case_starts',
            'linear_biases', 'quadratic', 'labels']`.

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
        if not return_offset:
            warnings.warn(
                "`return_offset` will default to `True` in the future.", DeprecationWarning,
                stacklevel=2
            )
            
            case_starts, linear_biases, quadratic = self._cydqm.to_numpy_vectors()

            return LegacyDQMVectors(case_starts,
                                    linear_biases,
                                    QuadraticVectors(*quadratic),
                                    self.variables)

        case_starts, linear_biases, quadratic, offset = self._cydqm.to_numpy_vectors(return_offset)

        return DQMVectors(
            case_starts, linear_biases, QuadraticVectors(*quadratic), self.variables, offset
        )


DQM = DiscreteQuadraticModel  # alias


# register fileview loader
load.register(DQM_MAGIC_PREFIX, DiscreteQuadraticModel.from_file)


class CaseLabelDQM(DQM):
    '''DiscreteQuadraticModel that allows assignment of arbitrary labels to
    cases of discrete variables.

    Two types of case labels are offered:

    1. Unique case labels are unique among variable labels and themselves.

    2. Shared case labels are unique among cases for a variable, but may be
       reused among variables.

    Examples:

        Declare variables with unique case labels.

        >>> dqm = dimod.CaseLabelDQM()
        >>> dqm.add_variable({'x1', 'x2', 'x3'})
        0
        >>> dqm.add_variable(['y1', 'y2', 'y3'])
        1

        Set linear biases

        >>> dqm.set_linear('x1', 0.5)
        >>> dqm.set_linear('y1', 1.5)

        Set quadratic biases

        >>> dqm.set_quadratic('x2', 'y3', -0.5)
        >>> dqm.set_quadratic('x3', 'y2', -1.5)

        Declare variables with shared case labels.

        >>> u = dqm.add_variable({'red', 'green', 'blue'}, shared_labels=True)
        >>> v = dqm.add_variable(['blue', 'yellow', 'brown'], label='v', shared_labels=True)

        Set linear biases

        >>> dqm.set_linear_case(u, 'red', 1)
        >>> dqm.set_linear_case(v, 'yellow', 2)

        Set quadratic biases

        >>> dqm.set_quadratic_case(u, 'green', v, 'blue', -0.5)
        >>> dqm.set_quadratic_case(u, 'blue', v, 'brown', -0.5)

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shared_case_label = defaultdict(dict)
        self._shared_label_case = defaultdict(dict)
        self._unique_case_label = {}
        self._unique_label_case = {}
        self._unique_label_vars = set()

    def add_variable(self, cases, label=None, shared_labels=False):
        """Add a discrete variable to the model.

        Args:
            cases (int or iterable):
                The number of cases in the variable, or an iterable containing
                the labels that will identify the cases of the variable.  Case
                labels can be any hashable.

            label (hashable, optional):
                A label for the variable. Can be any hashable except `None`.
                Defaults to the length of the discrete quadratic model, if that
                label is available. Otherwise defaults to the lowest available
                positive integer label.

            shared_labels (bool, optional, default=False):
                If True and `cases` is an iterable, shared case labels are
                created.  If False and `cases` is an iterable, unique case
                labels are created.  If `cases` is not an iterable, ignored.

        Returns:
            The label of the new variable.

        Raises:
            ValueError: If `label` already exists as a variable label, or if
                any of the case labels is not unique.

            TypeError: If `label` is not hashable, or if any of the case labels
                is not hashable.

        """
        if label in self._unique_label_case:
            raise ValueError(f'variable label {label} is not unique')

        if isinstance(cases, int):
            return super().add_variable(cases, label=label)

        else:
            if len(set(cases)) != len(cases):
                raise ValueError('case labels are not unique')

            if shared_labels:
                var = super().add_variable(len(cases), label=label)

                for k, case in enumerate(cases):
                    self._shared_label_case[var][case] = k
                    self._shared_case_label[var][k] = case

            else:
                for case in cases:
                    if (case in self.variables) or (case in self._unique_label_case):
                        raise ValueError(f'case label {case} is not unique')

                var = super().add_variable(len(cases), label=label)
                self._unique_label_vars.add(var)

                for k, case in enumerate(cases):
                    self._unique_label_case[case] = (var, k)
                    self._unique_case_label[(var, k)] = case

            return var

    def _lookup_shared_case(self, v, case):
        """Translate shared case label `case` of variable `v` to integer.

        Raises:
            ValueError: If `case` of `v` is unknown.

        """
        map_ = self._shared_label_case.get(v)
        if map_:
            if case not in map_:
                raise ValueError(f'unknown case {case} of variable {v}')
            return map_[case]
        return case

    def get_linear(self, v):
        """The linear biases associated with variable `v`.

        Args:
            v: A variable in the discrete quadratic model, or a unique case
                label.

        Returns:
            The linear biases.  If `v` is a variable, returns a NumPy array of
                size :meth:`~DiscreteQuadraticModel.num_cases(v)` by 1.

                If `v` is a unique case label, returns a float.

        """
        v_k = self._unique_label_case.get(v)
        if v_k:
            return super().get_linear_case(*v_k)
        else:
            return super().get_linear(v)

    def get_linear_case(self, v, case):
        """The linear bias associated with case `case` of variable `v`.

        Args:
            v: A variable in the discrete quadratic model.

            case: The case of `v`.

        Returns:
            The linear bias.

        """
        case = self._lookup_shared_case(v, case)
        return super().get_linear_case(v, case)

    def get_quadratic(self, u, v, array=False):
        """The biases associated with the interaction between `u` and `v`.

        Args:
            u: A variable in the discrete quadratic model, or a unique case
                label.  If `u` is a unique case label, `v` must be a unique
                case label.

            v: A variable in the discrete quadratic model, or a unique case
                label.  If `u` is a unique case label, `v` must be a unique
                case label.

            array (bool, optional, default=False): If True and `u` and `v` are
                variables, a dense array is returned rather than a dict.  If
                `u` and `v` are unique case labels, ignored.

        Returns:
            The quadratic biases.  If `array=False` and `u` and `v` are
            variables, returns a dictionary of the form
            `{case_u, case_v: bias, ...}`

            If `array=True` and `u` and `v` are variables, returns a NumPy
            array of size :meth:`~DiscreteQuadraticModel.num_cases(u)` by
            :meth:`~DiscreteQuadraticModel.num_cases(v)`.

            If `u` and `v` are unique case labels, returns a float.

        Raises:
            ValueError: If `u` is a unique case label and `v` is not.

        """
        u_k = self._unique_label_case.get(u)
        if u_k:
            if v not in self._unique_label_case:
                raise ValueError(f'unknown case label {v}')

            v_m = self._unique_label_case[v]
            return super().get_quadratic_case(*u_k, *v_m)
        else:
            return super().get_quadratic(u, v)

    def get_quadratic_case(self, u, u_case, v, v_case):
        """The bias associated with the interaction between two cases of `u`
        and `v`.

        Args:
            u: A variable in the discrete quadratic model.

            u_case: The case of `u`.

            v: A variable in the discrete quadratic model.

            v_case: The case of `v`.

        Returns:
            The quadratic bias.

        """
        u_case = self._lookup_shared_case(u, u_case)
        v_case = self._lookup_shared_case(v, v_case)
        return super().get_quadratic_case(u, u_case, v, v_case)

    def set_linear(self, v, biases):
        """Set the linear biases associated with `v`.

        Args:
            v: A variable in the discrete quadratic model, or a unique case
                label.

            biases (float or array-like):  If `v` is a variable, the linear
                biases is an array.  Otherwise, the linear bias is a real
                number.

        """
        v_k = self._unique_label_case.get(v)
        if v_k:
            super().set_linear_case(*v_k, biases)
        else:
            super().set_linear(v, biases)

    def set_linear_case(self, v, case, bias):
        """The linear bias associated with case `case` of variable `v`.

        Args:
            v: A variable in the discrete quadratic model.

            case: The case of `v`.

            bias (float): The linear bias.

        """
        case = self._lookup_shared_case(v, case)
        super().set_linear_case(v, case, bias)

    def set_quadratic(self, u, v, biases):
        """Set biases associated with the interaction between `u` and `v`.

        Args:
            u: A variable in the discrete quadratic model, or a unique case
                label.  If `u` is a unique case label, `v` must be a unique
                case label.

            v: A variable in the discrete quadratic model, or a unique case
                label.  If `u` is a unique case label, `v` must be a unique
                case label.

            biases (float or array-like/dict):
                The quadratic biases.  If `u` and `v` are variables, then
                `biases` may be a dictionary of the form
                `{case_u, case_v: bias, ...}` or a
                :meth:`~DiscreteQuadraticModel.num_cases(u)` by
                :meth:`~DiscreteQuadraticModel.num_cases(v)` array-like.

                If `u` and `v` are unique case labels, the quadratic bias is a
                real number.

        Raises:
            ValueError: If `u` is a unique case label and `v` is not.

        """
        u_k = self._unique_label_case.get(u)
        if u_k:
            if v not in self._unique_label_case:
                raise ValueError(f'unknown case label {v}')

            v_m = self._unique_label_case[v]
            super().set_quadratic_case(*u_k, *v_m, biases)
        else:
            super().set_quadratic(u, v, biases)

    def set_quadratic_case(self, u, u_case, v, v_case, bias):
        """Set the bias associated with the interaction between two cases of
        variables `u` and `v`.

        Args:
            u: A variable in the discrete quadratic model.

            u_case: The case of `u`.

            v: A variable in the discrete quadratic model.

            v_case: The case of `v`.

            bias (float): The quadratic bias.

        """
        u_case = self._lookup_shared_case(u, u_case)
        v_case = self._lookup_shared_case(v, v_case)
        super().set_quadratic_case(u, u_case, v, v_case, bias)

    def get_cases(self, v):
        """The cases of variable `v`.

        Returns:
            List of case labels for `v`, if case labels exist for `v`.

            If case labels do not exist for `v`, returns a list of integers
            from `0` to :meth:`~DiscreteQuadraticModel.num_cases(v)` - 1.

        """
        range_ = range(self.num_cases(v))
        map_ = self._shared_case_label.get(v)
        if map_:
            return [map_[case] for case in range_]

        elif v in self._unique_label_vars:
            return [self._unique_case_label[(v, case)] for case in range_]

        else:
            return list(range_)

    def to_file(self, *, ignore_labels=False, **kwargs):
        # We keep the default value the same as the super class, but if
        # we're ignoring the labels, the serialization is identical to
        # that of the unlabelled DQM
        if ignore_labels:
            return super().to_file(ignore_labels=True, **kwargs)

        raise NotImplementedError("serialization for CaseLabelDQM is not implemented, "
                                  "try using ignore_labels=True")

    def map_sample(self, sample):
        """Transform a sample to reflect case labels.

        Args:
            sample (dict): The sample to transform.

        Returns:
            The transformed sample.

        """
        new_sample = {}

        for var, value in sample.items():
            map_ = self._shared_case_label.get(var)
            if map_:
                new_sample[var] = map_[value]

            elif var in self._unique_label_vars:
                for case in range(self.num_cases(var)):
                    new_sample[self._unique_case_label[(var, case)]] = (value == case)

            else:
                new_sample[var] = value

        return new_sample
