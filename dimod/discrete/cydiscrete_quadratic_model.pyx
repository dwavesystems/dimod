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

cimport cython

from cython.operator cimport preincrement as inc, dereference as deref
from libcpp.algorithm cimport lower_bound, sort
from libcpp.unordered_set cimport unordered_set

import numpy as np

from dimod.utilities import asintegerarrays, asnumericarrays


cdef class cyDiscreteQuadraticModel:

    def __init__(self):
        self.dtype = np.float64
        self.case_dtype = np.int64

    @property
    def adj(self):
        return self.dqm_.adj_

    cpdef Py_ssize_t add_variable(self, Py_ssize_t num_cases) except -1:
        """Add a discrete variable.

        Args:
            num_cases (int): The number of cases in the variable.

        Returns:
            int: The label of the new variable.

        """

        if num_cases <= 0:
            raise ValueError("num_cases must be a positive integer")

        return self.dqm_.add_variable(num_cases)

    def copy(self):
        cdef cyDiscreteQuadraticModel dqm = type(self)()

        dqm.dqm_ = self.dqm_

        dqm.dtype = self.dtype
        dqm.case_dtype = self.dtype

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Bias[:] energies(self, CaseIndex[:, :] samples):

        if samples.shape[1] != self.num_variables():
            raise ValueError("Given sample(s) have incorrect number of variables")

        cdef Py_ssize_t num_samples = samples.shape[0]
        cdef VarIndex num_variables = samples.shape[1]

        cdef Bias[:] energies = np.empty(num_samples, dtype=self.dtype)

        cdef Py_ssize_t si, vi
        cdef CaseIndex cu, case_u, cv, case_v
        cdef VarIndex u, v
        for si in range(num_samples):
            for u in range(num_variables):
                case_u = samples[si, u]

                if case_u >= self.num_cases(u):
                    raise ValueError("invalid case")

        self.dqm_.get_energies( & samples[0, 0], num_samples, num_variables, & energies[0])
        return energies

    @classmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _from_numpy_vectors(cls,
                            Integral32plus[::1] case_starts,
                            Numeric32plus[::1] linear_biases,
                            Integral32plus[::1] irow,
                            Integral32plus[::1] icol,
                            Numeric32plus[::1] quadratic_biases):
        """Equivalent of from_numpy_vectors with fused types."""

        # some input checking
        # variable declarations we'll use throughout
        cdef Py_ssize_t u, v  # variables
        cdef Py_ssize_t ci, cj  # case indices
        cdef Py_ssize_t qi

        # constants
        cdef Py_ssize_t num_variables = case_starts.shape[0]
        cdef Py_ssize_t num_cases = linear_biases.shape[0]
        cdef Py_ssize_t num_interactions = irow.shape[0]

        # check that starts and linear_biases are correct and consistent with eachother
        for v in range(case_starts.shape[0] - 1):
            if case_starts[v+1] < case_starts[v]:
                raise ValueError("case_starts is not correctly ordered")

            if case_starts[v+1] >= linear_biases.shape[0]:
                raise ValueError("case_starts does not match linear_biases")

        # check that the quadratic are correct and consistent with eachother
        if not (irow.shape[0] == icol.shape[0] == quadratic_biases.shape[0]):
            raise ValueError("inconsistent lengths for irow, icol, qdata")
        for qi in range(irow.shape[0]):
            if not (0 <= irow[qi] < num_cases):
                raise ValueError("irow refers to case {} which is out of range"
                                 "".format(irow[qi]))
            if not (0 <= icol[qi] < num_cases):
                raise ValueError("icol refers to case {} which is out of range"
                                 "".format(icol[qi]))
            if irow[qi] == icol[qi]:
                raise ValueError("quadratic data contains a self-loop")

        cdef cyDiscreteQuadraticModel cyDQM = cls()

        cyDQM.dqm_ = cppAdjVectorDQM(case_starts.data, num_variables, linear_biases.data,
                                     num_cases, irow.data, icol.data, quadratic_biases.data, num_interactions)

        if cyDQM.dqm_.self_loop_present():
            raise ValueError("A variable has a self-loop")

        return cyDQM

    @classmethod
    def from_numpy_vectors(cls, case_starts, linear_biases, quadratic):

        cdef cyDiscreteQuadraticModel obj = cls()

        try:
            irow, icol, quadratic_biases = quadratic
        except ValueError:
            raise ValueError("quadratic should be a 3-tuple")

        # We need:
        # * numpy ndarrays
        # * contiguous memory
        # * irow.dtype == icol.dtype == case_starts.dtype
        # * ldata.dtype==qdata.dtype
        # * 32 or 64 bit dtypes
        case_starts, icol, irow = asintegerarrays(
            case_starts, icol, irow, min_itemsize=4, requirements='C')
        ldata, qdata = asnumericarrays(
            linear_biases, quadratic_biases, min_itemsize=4, requirements='C')

        return cls._from_numpy_vectors(case_starts, ldata, irow, icol, qdata)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_linear(self, VarIndex v):

        cdef Py_ssize_t num_cases = self.num_cases(v)

        biases = np.empty(num_cases, dtype=self.dtype)
        cdef Bias[:] biases_view = biases
        self.dqm_.get_linear(v, & biases_view[0])
        return biases

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Bias get_linear_case(self, VarIndex v, CaseIndex case) except? -45.3:

        # self.num_cases checks that the variable is valid

        if case < 0 or case >= self.num_cases(v):
            raise ValueError("case {} is invalid, variable only supports {} "
                             "cases".format(case, self.num_cases(v)))

        return self.dqm_.get_linear_case(v, case)

    def get_quadratic(self, VarIndex u, VarIndex v, bint array=False):

        # check that the interaction does in fact exist
        if u < 0 or u >= self.num_variables():
            raise ValueError("unknown variable")
        if v < 0 or v >= self.num_variables():
            raise ValueError("unknown variable")

        if not self.dqm_.connection_present(u,v):
            raise ValueError("there is no interaction between given variables")

        cdef CaseIndex ci
        cdef Py_ssize_t case_u, case_v
        cdef Bias[:, :] quadratic_view

        if array:
            # build a numpy array
            quadratic = np.empty((self.num_cases(u), self.num_cases(v)),
                                 dtype=self.dtype)
            quadratic_view = quadratic
            self.dqm_.get_quadratic(u, v, & quadratic_view[0, 0])

        else:
            # store in a dict
            quadratic = {}

            for ci in range(self.dqm_.case_starts_[u], self.dqm_.case_starts_[u+1]):

                span = self.dqm_.bqm_.neighborhood(
                    ci, self.dqm_.case_starts_[v])

                while (span.first != span.second and deref(span.first).first < self.dqm_.case_starts_[v+1]):
                    case_u = ci - self.dqm_.case_starts_[u]
                    case_v = deref(span.first).first - self.case_starts_[v]
                    quadratic[case_u, case_v] = deref(span.first).second

                    inc(span.first)

        # todo: support scipy sparse matrices?

        return quadratic

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Bias get_quadratic_case(self,
                                  VarIndex u, CaseIndex case_u,
                                  VarIndex v, CaseIndex case_v) except? -45.3:

        if case_u < 0 or case_u >= self.num_cases(u):
            raise ValueError("case {} is invalid, variable only supports {} "
                             "cases".format(case_u, self.num_cases(u)))

        if case_v < 0 or case_v >= self.num_cases(v):
            raise ValueError("case {} is invalid, variable only supports {} "
                             "cases".format(case_v, self.num_cases(v)))

        return self.dqm_.get_quadratic_case(u, case_u, v, case_v).first

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Py_ssize_t num_cases(self, Py_ssize_t v=-1) except -1:
        """If v is provided, the number of cases associated with v, otherwise
        the total number of cases in the DQM.
        """
        if v < 0:
            return self.dqm_.bqm_.num_variables()

        if v >= self.num_variables():
            raise ValueError("unknown variable {}".format(v))

        return self.dqm_.num_cases(v)

    cpdef Py_ssize_t num_case_interactions(self):
        """The total number of case interactions."""
        return self.dqm_.num_case_interactions()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Py_ssize_t num_variable_interactions(self) except -1:
        """The total number of case interactions."""
        return self.dqm_.num_variable_interactions()

    cpdef Py_ssize_t num_variables(self):
        """The number of discrete variables in the DQM."""
        return self.dqm_.num_variables()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Py_ssize_t set_linear(self, VarIndex v, Numeric[:] biases) except -1:

        # self.num_cases checks that the variable is valid

        if biases.shape[0] != self.num_cases(v):
            raise ValueError('Recieved {} bias(es) for a variable of degree {}'
                             ''.format(biases.shape[0], self.num_cases(v)))

        self.dqm_.set_linear(v, & biases[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Py_ssize_t set_linear_case(self, VarIndex v, CaseIndex case, Bias b) except -1:

        # self.num_cases checks that the variable is valid

        if case < 0:
            raise ValueError("case should be a positive integer")
        if case >= self.num_cases(v):
            raise ValueError("case {} is invalid, variable only supports {} "
                             "cases".format(case, self.num_cases(v)))

        self.dqm_.set_linear_case(v, case, b)

    def set_quadratic(self, VarIndex u, VarIndex v, biases):

        # check that the interaction does in fact exist
        if u < 0 or u >= self.num_variables():
            raise ValueError("unknown variable")
        if v < 0 or v >= self.num_variables():
            raise ValueError("unknown variable")
        if u == v:
            raise ValueError("there cannot be a quadratic interaction between "
                             "two cases in the same variable")

        cdef CaseIndex case_u, case_v
        cdef CaseIndex cu, cv
        cdef Bias bias

        cdef Py_ssize_t num_cases_u = self.num_cases(u)
        cdef Py_ssize_t num_cases_v = self.num_cases(v)

        cdef Bias[:, :] biases_view

        if isinstance(biases, abc.Mapping):
            for (case_u, case_v), bias in biases.items():
                if case_u < 0 or case_u >= num_cases_u:
                    raise ValueError("case {} is invalid, variable only supports {} "
                                     "cases".format(case_u, self.num_cases(u)))

                if case_v < 0 or case_v >= num_cases_v:
                    raise ValueError("case {} is invalid, variable only supports {} "
                                     "cases".format(case_v, self.num_cases(v)))

                self.dqm_.set_quadratic_case(u, case_u, v, case_v, bias)
        else:

            biases_view = np.asarray(biases, dtype=self.dtype).reshape(
                num_cases_u, num_cases_v)
            self.dqm_.set_quadratic(u, v, & biases_view[0, 0])

    cpdef Py_ssize_t set_quadratic_case(self,
                                        VarIndex u, CaseIndex case_u,
                                        VarIndex v, CaseIndex case_v,
                                        Bias bias) except -1:

        # self.num_cases checks that the variables are valid

        if case_u < 0 or case_u >= self.num_cases(u):
            raise ValueError("case {} is invalid, variable only supports {} "
                             "cases".format(case_u, self.num_cases(u)))

        if case_v < 0 or case_v >= self.num_cases(v):
            raise ValueError("case {} is invalid, variable only supports {} "
                             "cases".format(case_v, self.num_cases(v)))

        if u == v:
            raise ValueError("there cannot be a quadratic interaction between "
                             "two cases in the same variable")

        self.dqm_.set_quadratic_case(u, case_u, v, case_v, bias)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _into_numpy_vectors(self, Unsigned[:] starts, Bias[:] ldata,
                                  Unsigned[:] irow, Unsigned[:] icol, Bias[:] qdata):
        # we don't do array length checking so be careful! This can segfault
        # if the given arrays are incorrectly sized
        self.dqm_.extract_data( & starts[0], & ldata[0], & irow[0], & icol[0], & qdata[0])

    def to_numpy_vectors(self):

        cdef Py_ssize_t num_variables = self.num_variables()
        cdef Py_ssize_t num_cases = self.num_cases()
        cdef Py_ssize_t num_interactions = self.num_case_interactions()

        # use the minimum sizes of the various index types. We combine for
        # variables and cases and exclude int8 to keep the total number of
        # cases down
        if num_cases < 1 << 16:
            index_dtype = np.uint16
        elif num_cases < 1 << 32:
            index_dtype = np.uint32
        else:
            index_dtype = np.uint64

        starts = np.empty(num_variables, dtype=index_dtype)
        ldata = np.empty(num_cases, dtype=self.dtype)
        irow = np.empty(num_interactions, dtype=index_dtype)
        icol = np.empty(num_interactions, dtype=index_dtype)
        qdata = np.empty(num_interactions, dtype=self.dtype)

        if index_dtype == np.uint16:
            self._into_numpy_vectors[np.uint16_t](starts, ldata, irow, icol, qdata)
        elif index_dtype == np.uint32:
            self._into_numpy_vectors[np.uint32_t](starts, ldata, irow, icol, qdata)
        else:
            self._into_numpy_vectors[np.uint64_t](starts, ldata, irow, icol, qdata)

        return starts, ldata, (irow, icol, qdata)
