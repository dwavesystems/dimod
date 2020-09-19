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

from operator import eq

import numpy as np

from dimod.sampleset import as_samples
from dimod.discrete.cydiscrete_quadratic_model import cyDiscreteQuadraticModel


__all__ = ['DiscreteQuadraticModel', 'DQM']


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
                - `qdata`: A length
                  :meth:`~DiscreteQuadraticModel.num_interactions` array. If
                  the case interactions were defined in a sparse matrix, these
                  would be the values.

            labels (list, optional):
                The variable labels. Defaults to index-labeled.

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

    def to_numpy_vectors(self, return_labels=False):
        """Convert the DQM to five numpy vectors.

        Args:
            return_labels (bool, optional, default=False):
                If True, the variable labels are returned.

        Returns:
            A tuple `(case_starts, linear_biases, (irow, icol, qdata))` of
            :class:`~numpy.ndarray`

            - `case_starts`: A length
              :meth:`~DiscreteQuadraticModel.num_variables` array. The cases
              associated with variable `v` are in the range `[case_starts[v],
              cases_starts[v+1])`.
            - `linear_biases`: A length
              :meth:`~DiscreteQuadraticModel.num_cases` array. The linear
              biases.
            - `irow`: A length
              :meth:`~DiscreteQuadraticModel.num_interactions` array. If the
              case interactions were defined in a sparse matrix, these would
              be the row indices.
            - `icol`: A length
              :meth:`~DiscreteQuadraticModel.num_interactions` array. If the
              case interactions were defined in a sparse matrix, these would
              be the column indices.
            - `qdata`: A length
              :meth:`~DiscreteQuadraticModel.num_interactions` array. If the
              case interactions were defined in a sparse matrix, these would
              be the values.

            If `return_labels=True`, this method will instead return a tuple
            `(case_starts, linear_biases, (irow, icol, qdata), labels)` where
            `labels` is a list of the variable labels.

        See Also:
            :meth:`~DiscreteQuadraticModel.from_numpy_vectors`

        """
        arrays = self._cydqm.to_numpy_vectors()

        if return_labels:
            arrays = list(arrays)
            arrays.append(list(self.variables))

        return arrays


DQM = DiscreteQuadraticModel  # alias
