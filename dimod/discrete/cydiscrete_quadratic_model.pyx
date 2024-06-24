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
from cython.operator cimport preincrement as preinc  # for when it matters
from libcpp.algorithm cimport lower_bound, sort, unique
from libcpp.unordered_set cimport unordered_set

import numpy as np

from dimod.utilities import asintegerarrays, asnumericarrays
from dimod.cyutilities cimport as_numpy_float
from dimod.typing cimport Integer, int64_t, uint16_t, uint32_t, uint64_t

BIAS_DTYPE = np.float64
INDEX_DTYPE = np.int32

# developer note: for whatever reason this is the only way we can sort a vector
# of structs in clang. If we do it in pure cython we get segmentation faults
cdef extern from *:
    """
    #include <vector>
    #include <algorithm>
    #include <cstdint>

    struct LinearTerm {
        std::int64_t variable;
        std::int64_t case_v;
        double bias;
    };

    bool term_lt(const LinearTerm& lhs, const LinearTerm& rhs) {
        return lhs.case_v < rhs.case_v;
    }

    void sort_terms(std::vector<LinearTerm>& terms) {
        std::sort(terms.begin(), terms.end(), term_lt);
    }
    """
    struct LinearTerm:
        int64_t variable
        int64_t case "case_v"
        double bias
    void sort_terms(vector[LinearTerm]&)


cdef class cyDiscreteQuadraticModel:

    def __init__(self):
        self.case_starts_.push_back(0)

        self.dtype = np.dtype(BIAS_DTYPE)
        self.case_dtype = np.dtype(INDEX_DTYPE)

    @property
    def adj(self):
        return self.adj_

    @property
    def offset(self):
        cdef bias_type offset = self.cppbqm.offset()
        return as_numpy_float(offset)

    @offset.setter
    def offset(self, bias_type offset):
        self.cppbqm.set_offset(offset)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_linear_equality_constraint(self, object terms,
                                       bias_type lagrange_multiplier, bias_type constant):
        # adjust energy offset
        self.cppbqm.add_offset(lagrange_multiplier * constant * constant)

        # resolve the terms from a python object into a C++ object
        cdef vector[LinearTerm] cppterms

        # can allocate it if we already know the size
        if isinstance(terms, abc.Sized):
            cppterms.reserve(len(terms))

        # put the generator or list into our C++ objects
        cdef Py_ssize_t v, case_v
        cdef bias_type bias
        cdef LinearTerm term
        for v, case_v, bias in terms:
            if case_v >= self.num_cases(v):  # also checks variable
                raise ValueError("case out of range")

            term.variable = v
            term.case = case_v + self.case_starts_[v]
            term.bias = bias
            cppterms.push_back(term)

        # sort and sum duplicates in terms
        sort_terms(cppterms)

        first = cppterms.begin()
        last = cppterms.end()
        if first != last:
            result = first

            while preinc(first) != last:
                if deref(result).case == deref(first).case:
                    # first is a duplicate
                    deref(result).bias = deref(result).bias + deref(first).bias
                else:
                    # advance result and set it to equal the current first
                    inc(result)
                    if result != first:
                        deref(result).variable = deref(first).variable
                        deref(result).case = deref(first).case
                        deref(result).bias = deref(first).bias

            # finally advance result and delete the rest of the vector
            cppterms.erase(preinc(result), cppterms.end())


        # add the biases to the BQM
        cdef Py_ssize_t num_terms = cppterms.size()

        cdef index_type u
        cdef Py_ssize_t i, j
        cdef index_type cu, cv
        cdef bias_type lbias, qbias, u_bias, v_bias
        for i in range(num_terms):
            u = cppterms[i].variable
            cu = cppterms[i].case
            u_bias = cppterms[i].bias

            lbias = lagrange_multiplier * u_bias * (2 * constant + u_bias)
            self.cppbqm.set_linear(cu, lbias + self.cppbqm.linear(cu))

            for j in range(i + 1, num_terms):
                cv = cppterms[j].case
                v_bias = cppterms[j].bias

                # interactions between cases within a variable never contribute
                # energy
                if u == cppterms[j].variable:
                    continue

                qbias = 2 * lagrange_multiplier * u_bias * v_bias

                self.cppbqm.add_quadratic(cu, cv, qbias)

        # now track the variables we used
        cdef unordered_set[index_type] variable_set
        for i in range(cppterms.size()):
            variable_set.insert(cppterms[i].variable)

        # sort the variables.
        cdef vector[index_type] variables
        variables.insert(variables.begin(), variable_set.begin(), variable_set.end())
        sort(variables.begin(), variables.end())

        # finally fix the adjacency
        cdef Py_ssize_t vi, ni
        for i in range(variables.size()):
            v = variables[i]

            # this is optimized for when variables are relatively large
            # compared adj (at 5000 variables the crossover point is %13)
            vit = variables.begin()
            nit = self.adj_[v].begin()
            while vit != variables.end():
                if deref(vit) == v:
                    inc(vit)
                elif nit == self.adj_[v].end():
                    nit = self.adj_[v].insert(nit, deref(vit))
                    inc(nit)
                    inc(vit)
                elif deref(vit) < deref(nit):
                    nit = self.adj_[v].insert(nit, deref(vit))
                    inc(nit)
                    inc(vit)
                elif deref(vit) > deref(nit):
                    inc(nit)
                else:  # *vit == *nit
                    inc(nit)
                    inc(vit)

    cpdef Py_ssize_t add_variable(self, Py_ssize_t num_cases) except -1:
        """Add a discrete variable.

        Args:
            num_cases (int): The number of cases in the variable.

        Returns:
            int: The label of the new variable.

        """

        if num_cases <= 0:
            raise ValueError("num_cases must be a positive integer")

        cdef index_type v = self.adj_.size()  # index of new variable

        self.adj_.resize(v+1)

        cdef Py_ssize_t i
        for i in range(num_cases):
            self.cppbqm.add_variable()

        self.case_starts_.push_back(self.cppbqm.num_variables())

        return v

    def copy(self):
        cdef cyDiscreteQuadraticModel dqm = type(self)()

        dqm.cppbqm = self.cppbqm
        dqm.case_starts_ = self.case_starts_
        dqm.adj_ = self.adj_
        dqm.cppbqm.set_offset(self.cppbqm.offset())

        return dqm

    def degree(self, Py_ssize_t vi):
        if not 0 <= vi < self.adj_.size():
            raise ValueError("not a variable")
        return self.adj_[vi].size()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bias_type[:] energies(self, index_type[:, :] samples):
        
        if samples.shape[1] != self.num_variables():
            raise ValueError("Given sample(s) have incorrect number of variables")

        cdef Py_ssize_t num_samples = samples.shape[0]
        cdef index_type num_variables = samples.shape[1]

        cdef bias_type[:] energies = np.full(num_samples, self.offset, dtype=self.dtype)

        cdef Py_ssize_t si, vi
        cdef index_type cu, case_u, cv, case_v
        cdef index_type u, v
        for si in range(num_samples):  # this could be parallelized
            for u in range(num_variables):
                case_u = samples[si, u]

                if case_u >= self.num_cases(u):
                    raise ValueError("invalid case")

                cu = self.case_starts_[u] + case_u

                energies[si] += self.cppbqm.linear(cu)

                for vi in range(self.adj_[u].size()):
                    v = self.adj_[u][vi]

                    # we only care about the lower triangle
                    if v > u:
                        break

                    case_v = samples[si, v]

                    cv = self.case_starts_[v] + case_v

                    energies[si] += self.cppbqm.quadratic(cu, cv)

        return energies

    @classmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _from_numpy_vectors(cls,
                            Integer[::1] case_starts,
                            Numeric[::1] linear_biases,
                            Integer[::1] irow,
                            Integer[::1] icol,
                            Numeric[::1] quadratic_biases,
                            bias_type offset):
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

        cdef cyDiscreteQuadraticModel dqm = cls()

        # set the BQM
        if num_interactions:
            dqm.cppbqm.add_quadratic_from_coo(
                &irow[0], &icol[0], &quadratic_biases[0],
                num_interactions)

        # add the linear biases
        while dqm.cppbqm.num_variables() < num_cases:
            dqm.cppbqm.add_variable()
        for ci in range(num_cases):
            dqm.cppbqm.set_linear(ci, linear_biases[ci])

        # set the case starts
        dqm.case_starts_.resize(case_starts.shape[0] + 1)
        for v in range(case_starts.shape[0]):
            dqm.case_starts_[v] = case_starts[v]
        dqm.case_starts_[case_starts.shape[0]] = dqm.cppbqm.num_variables()

        # and finally the adj. This is not really the memory bottleneck so
        # we can build an intermediate (unordered) set version
        cdef vector[unordered_set[index_type]] adjset
        adjset.resize(num_variables)
        u = 0
        for ci in range(dqm.cppbqm.num_variables()):
            
            # we've been careful so don't need ui < case_starts.size() - 1
            while ci >= dqm.case_starts_[u+1]:
                u += 1

            span = dqm.cppbqm.neighborhood(ci)

            v = 0
            while span.first != span.second:
                cj = deref(span.first).first

                # see above note
                while cj >= dqm.case_starts_[v+1]:
                    v += 1

                adjset[u].insert(v)

                inc(span.first)

        # now put adjset into adj
        dqm.adj_.resize(num_variables)
        for v in range(num_variables):
            dqm.adj_[v].insert(dqm.adj_[v].begin(),
                               adjset[v].begin(), adjset[v].end())
            sort(dqm.adj_[v].begin(), dqm.adj_[v].end())

        # do one last final check for self-loops within a variable
        for v in range(num_variables):
            for ci in range(dqm.case_starts_[v], dqm.case_starts_[v+1]):
                span2 = dqm.cppbqm.neighborhood(ci, dqm.case_starts_[v])
                if span2.first == span2.second:
                    continue
                if deref(span2.first).first < dqm.case_starts_[v+1]:
                    raise ValueError("A variable has a self-loop")

        # add provided offset to dqm
        dqm.cppbqm.set_offset(offset)

        return dqm

    @classmethod
    def from_numpy_vectors(cls, case_starts, linear_biases, quadratic, offset = 0):

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

        return cls._from_numpy_vectors(case_starts, ldata, irow, icol, qdata, offset)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_linear(self, index_type v):

        cdef Py_ssize_t num_cases = self.num_cases(v)

        biases = np.empty(num_cases, dtype=np.float64)
        cdef bias_type[:] biases_view = biases

        cdef Py_ssize_t c
        for c in range(num_cases):
            biases_view[c] = self.cppbqm.linear(self.case_starts_[v] + c)

        return biases

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bias_type get_linear_case(self, index_type v, index_type case) except? -45.3:

        # self.num_cases checks that the variable is valid

        if case < 0 or case >= self.num_cases(v):
            raise ValueError("case {} is invalid, variable only supports {} "
                             "cases".format(case, self.num_cases(v)))

        return self.cppbqm.linear(self.case_starts_[v] + case)

    def get_quadratic(self, index_type u, index_type v, bint array=False):

        # check that the interaction does in fact exist
        if u < 0 or u >= self.adj_.size():
            raise ValueError("unknown variable")
        if v < 0 or v >= self.adj_.size():
            raise ValueError("unknown variable")

        it = lower_bound(self.adj_[u].begin(), self.adj_[u].end(), v)
        if it == self.adj_[u].end() or deref(it) != v:
            raise ValueError("there is no interaction between given variables")

        cdef index_type ci
        cdef Py_ssize_t case_u, case_v
        cdef bias_type[:, :] quadratic_view

        if array:
            # build a numpy array
            quadratic = np.zeros((self.num_cases(u), self.num_cases(v)),
                                 dtype=self.dtype)
            quadratic_view = quadratic

            for ci in range(self.case_starts_[u], self.case_starts_[u+1]):

                span = self.cppbqm.neighborhood(ci, self.case_starts_[v])

                while (span.first != span.second and deref(span.first).first < self.case_starts_[v+1]):
                    case_u = ci - self.case_starts_[u]
                    case_v = deref(span.first).first - self.case_starts_[v]
                    quadratic_view[case_u, case_v] = deref(span.first).second

                    inc(span.first)

        else:
            # store in a dict
            quadratic = {}

            for ci in range(self.case_starts_[u], self.case_starts_[u+1]):

                span = self.cppbqm.neighborhood(ci, self.case_starts_[v])

                while (span.first != span.second and deref(span.first).first < self.case_starts_[v+1]):
                    case_u = ci - self.case_starts_[u]
                    case_v = deref(span.first).first - self.case_starts_[v]
                    quadratic[case_u, case_v] = deref(span.first).second

                    inc(span.first)

        # todo: support scipy sparse matrices?

        return quadratic

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bias_type get_quadratic_case(self,
                                  index_type u, index_type case_u,
                                  index_type v, index_type case_v)  except? -45.3:

        if case_u < 0 or case_u >= self.num_cases(u):
            raise ValueError("case {} is invalid, variable only supports {} "
                             "cases".format(case_u, self.num_cases(u)))

        if case_v < 0 or case_v >= self.num_cases(v):
            raise ValueError("case {} is invalid, variable only supports {} "
                             "cases".format(case_v, self.num_cases(v)))


        cdef index_type cu = self.case_starts_[u] + case_u
        cdef index_type cv = self.case_starts_[v] + case_v

        return self.cppbqm.quadratic(cu, cv)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Py_ssize_t num_cases(self, Py_ssize_t v=-1) except -1:
        """If v is provided, the number of cases associated with v, otherwise
        the total number of cases in the DQM.
        """
        if v < 0:
            return self.cppbqm.num_variables()

        if v >= self.num_variables():
            raise ValueError("unknown variable {}".format(v))

        return self.case_starts_[v+1] - self.case_starts_[v]

    cpdef Py_ssize_t num_case_interactions(self):
        """The total number of case interactions."""
        return self.cppbqm.num_interactions()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Py_ssize_t num_variable_interactions(self) except -1:
        """The total number of case interactions."""
        cdef Py_ssize_t num = 0
        cdef Py_ssize_t v
        for v in range(self.num_variables()):
            num += self.adj_[v].size()
        return num // 2

    cpdef Py_ssize_t num_variables(self):
        """The number of discrete variables in the DQM."""
        return self.adj_.size()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Py_ssize_t set_linear(self, index_type v, Numeric[:] biases) except -1:

        # self.num_cases checks that the variable is valid

        if biases.shape[0] != self.num_cases(v):
            raise ValueError('Recieved {} bias(es) for a variable of degree {}'
                             ''.format(biases.shape[0], self.num_cases(v)))

        cdef Py_ssize_t c
        for c in range(biases.shape[0]):
            self.cppbqm.set_linear(self.case_starts_[v] + c, biases[c])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Py_ssize_t set_linear_case(self, index_type v, index_type case, bias_type b) except -1:
        
        # self.num_cases checks that the variable is valid

        if case < 0:
            raise ValueError("case should be a positive integer")
        if case >= self.num_cases(v):
            raise ValueError("case {} is invalid, variable only supports {} "
                             "cases".format(case, self.num_cases(v)))

        self.cppbqm.set_linear(self.case_starts_[v] + case, b)

    def set_quadratic(self, index_type u, index_type v, biases):

        # check that the interaction does in fact exist
        if u < 0 or u >= self.adj_.size():
            raise ValueError("unknown variable")
        if v < 0 or v >= self.adj_.size():
            raise ValueError("unknown variable")
        if u == v:
            raise ValueError("there cannot be a quadratic interaction between "
                             "two cases in the same variable")

        cdef index_type case_u, case_v
        cdef index_type cu, cv
        cdef bias_type bias

        cdef Py_ssize_t num_cases_u = self.num_cases(u)
        cdef Py_ssize_t num_cases_v = self.num_cases(v)

        cdef bias_type[:, :] biases_view

        if isinstance(biases, abc.Mapping):
            for (case_u, case_v), bias in biases.items():
                if case_u < 0 or case_u >= num_cases_u:
                    raise ValueError("case {} is invalid, variable only supports {} "
                                     "cases".format(case_u, self.num_cases(u)))

                if case_v < 0 or case_v >= num_cases_v:
                    raise ValueError("case {} is invalid, variable only supports {} "
                                     "cases".format(case_v, self.num_cases(v)))

                cu = self.case_starts_[u] + case_u
                cv = self.case_starts_[v] + case_v

                self.cppbqm.set_quadratic(cu, cv, bias)
        else:
            
            biases_view = np.asarray(biases, dtype=self.dtype).reshape(num_cases_u, num_cases_v)

            for case_u in range(biases_view.shape[0]):
                cu = self.case_starts_[u] + case_u
                for case_v in range(biases_view.shape[1]):
                     cv = self.case_starts_[v] + case_v

                     bias = biases_view[case_u, case_v]

                     if bias:
                         self.cppbqm.set_quadratic(cu, cv, bias)

        # track in adjacency
        low = lower_bound(self.adj_[u].begin(), self.adj_[u].end(), v)
        if low == self.adj_[u].end() or deref(low) != v:
            # need to add
            self.adj_[u].insert(low, v)
            self.adj_[v].insert(
                lower_bound(self.adj_[v].begin(), self.adj_[v].end(), u),
                u)

    cpdef Py_ssize_t set_quadratic_case(self,
                                        index_type u, index_type case_u,
                                        index_type v, index_type case_v,
                                        bias_type bias) except -1:

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


        cdef index_type cu = self.case_starts_[u] + case_u
        cdef index_type cv = self.case_starts_[v] + case_v

        self.cppbqm.set_quadratic(cu, cv, bias)

        # track in adjacency
        low = lower_bound(self.adj_[u].begin(), self.adj_[u].end(), v)
        if low == self.adj_[u].end() or deref(low) != v:
            # need to add
            self.adj_[u].insert(low, v)
            self.adj_[v].insert(
                lower_bound(self.adj_[v].begin(), self.adj_[v].end(), u),
                u)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _into_numpy_vectors(self,
                                  Integer[:] starts,
                                  bias_type[:] ldata,
                                  Integer[:] irow, Integer[:] icol, bias_type[:] qdata,
                                  ):
        # we don't do array length checking so be careful! This can segfault
        # if the given arrays are incorrectly sized

        cdef Py_ssize_t vi
        for vi in range(self.num_variables()):
            starts[vi] = self.case_starts_[vi]

        cdef Py_ssize_t ci = 0
        cdef Py_ssize_t qi = 0
        for ci in range(self.cppbqm.num_variables()):
            ldata[ci] = self.cppbqm.linear(ci)

            span = self.cppbqm.neighborhood(ci)
            while span.first != span.second and deref(span.first).first < ci:

                irow[qi] = ci
                icol[qi] = deref(span.first).first
                qdata[qi] = deref(span.first).second

                inc(span.first)
                qi += 1
        

    def to_numpy_vectors(self, bint return_offset=False):
        
        cdef Py_ssize_t num_variables = self.num_variables()
        cdef Py_ssize_t num_cases = self.num_cases()
        cdef Py_ssize_t num_interactions = self.cppbqm.num_interactions()

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
            self._into_numpy_vectors[uint16_t](starts, ldata, irow, icol, qdata)
        elif index_dtype == np.uint32:
            self._into_numpy_vectors[uint32_t](starts, ldata, irow, icol, qdata)
        else:
            self._into_numpy_vectors[uint64_t](starts, ldata, irow, icol, qdata)

        if return_offset:
            return starts, ldata, (irow, icol, qdata), self.offset

        return starts, ldata, (irow, icol, qdata)
