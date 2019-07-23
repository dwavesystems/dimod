# distutils: language = c++
# cython: language_level=3

# google c++ convention for names
# https://google.github.io/styleguide/cppguide.html#Naming

from numbers import Integral

from cython.operator cimport dereference as deref
from cython.operator cimport postincrement

from libcpp.map cimport map as cppmap
from libcpp.pair cimport pair
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

from dimod.bqm.cybqm cimport AdjArrayBQM

ctypedef np.float64_t Bias
ctypedef np.uint32_t VarIndex

bias_dtype = np.float64

ctypedef pair[VarIndex, Bias] OutVar
ctypedef cppmap[VarIndex, Bias] Neighbourhood
ctypedef pair[Neighbourhood, Bias] InVar


cdef class AdjMapBQM:
    """

    This can be instantiated in several ways:

        AdjMapBQM()
            Creates an empty binary quadratic model

        AdjMapBQM(n)
            Where n is the number of nodes.

        AdjMapBQM((linear, [quadratic, [offset]]))
            Where linear, quadratic are:
                dict[int, bias]
                sequence[bias]
            *NOT IMPLEMENTED YET*

        AdjMapBQM(bqm)
            Where bqm is another binary quadratic model (equivalent to
            bqm.to_adjvector())
            *NOT IMPLEMENTED YET*

        AdjMapBQM(D)
            Where D is a dense matrix D

    """
    cdef vector[InVar] adj_

    def __init__(self, object arg1=0):

        cdef Bias [:, :] D  # in case it's dense
        cdef size_t num_variables
        cdef VarIndex u, v
        cdef Bias b

        if isinstance(arg1, Integral):
            self.adj_.resize(arg1)  # default constructor
        elif isinstance(arg1, tuple):
            raise NotImplementedError  # update docstring
        elif hasattr(arg1, "to_adjvector"):
            # we might want a more generic is_bqm function or similar
            raise NotImplementedError  # update docstring
        else:
            # assume it's dense
            D = np.atleast_2d(np.asarray(arg1, dtype=bias_dtype))

            num_variables = D.shape[0]

            if D.ndim != 2 or num_variables != D.shape[1]:
                raise ValueError("expected dense to be a 2 dim square array")

            self.adj_.resize(num_variables)  # defaults values to 0

            for u in range(num_variables):
                for v in range(num_variables):
                    b = D[u, v]

                    if u == v and b != 0:
                        self.adj_[u].second = b
                    elif b != 0:  # ignore the 0 off-diagonal
                        self.adj_[u].first.insert((v, b))
                        self.adj_[v].first.insert((u, b))

    def __len__(self):
        return self.num_variables

    @property
    def num_variables(self):
        return self.adj_.size()

    @property
    def num_interactions(self):
        cdef size_t count = 0
        cdef vector[InVar].iterator it = self.adj_.begin()
        while it != self.adj_.end():
            count += deref(postincrement(it)).first.size()
        return count // 2

    @property
    def shape(self):
        return self.num_variables, self.num_interactions

    ###########################################################################
    # Variable/Interaction base methods
    ###########################################################################

    def iter_variables(self):
        # while index-labelled this is just iterating over the size
        cdef VarIndex v
        for v in range(self.adj_.size()):
            yield v

    def iter_interactions(self):
        raise NotImplementedError

    def iter_neighbours(self):
        raise NotImplementedError

    ###########################################################################
    # Linear base methods
    ###########################################################################

    def append_linear(self, Bias b):
        cdef InVar invar  # creates it as empty
        invar.second = b
        self.adj_.push_back(invar)

    def get_linear(self, VarIndex v):
        if v > self.adj_.size() or v < 0:
            raise ValueError("out of range variable {}".format(v))
        return self.adj_[v].second

    def pop_linear(self):
        if self.adj_.empty():  # need this check for following loop
            raise ValueError("cannot pop from an empty BQM")

        # remove any associated interactions
        cdef OutVar outvar
        cdef VarIndex v = self.adj_.size() - 1
        for outvar in deref(self.adj_.rbegin()).first:
            self.adj_[outvar.first].first.erase(v)

        # remove the variable
        self.adj_.pop_back()

    def set_linear(self, VarIndex v, Bias b):
        if v > self.adj_.size() or v < 0:
            raise ValueError("out of range variable {}".format(v))
        self.adj_[v].second = b

    ###########################################################################
    # Quadratic base methods
    ###########################################################################

    def get_quadratic(self, VarIndex u, VarIndex v):
        if u > self.adj_.size() or u < 0:
            raise ValueError("out of range variable {}".format(u))
        if v > self.adj_.size() or v < 0:
            raise ValueError("out of range variable {}".format(v))
        if u == v:
            raise ValueError("no self-loops allowed")

        cdef cppmap[VarIndex, Bias].iterator it = self.adj_[u].first.find(v)
        if it == self.adj_[u].first.end():
            raise ValueError("no interaction between {},{}".format(u, v))
        return deref(it).second

    def remove_quadratic(self, VarIndex u, VarIndex v):
        if u > self.adj_.size() or u < 0:
            raise ValueError("out of range variable {}".format(u))
        if v > self.adj_.size() or v < 0:
            raise ValueError("out of range variable {}".format(v))
        if u == v:
            raise ValueError("no self-loops allowed")

        self.adj_[u].first.erase(v)
        self.adj_[v].first.erase(u)

    def set_quadratic(self, VarIndex u, VarIndex v, Bias b):
        if u > self.adj_.size() or u < 0:
            raise ValueError("out of range variable {}".format(u))
        if v > self.adj_.size() or v < 0:
            raise ValueError("out of range variable {}".format(v))
        if u == v:
            raise ValueError("no self-loops allowed")

        self.adj_[u].first[v] = b
        self.adj_[v].first[u] = b


    ##
    # Methods
    #

    def to_adjarray(self):
        # this is always a copy

        # make a 0-length BQM but then manually resize it, note that this
        # treats them as vectors
        cdef AdjArrayBQM bqm = AdjArrayBQM()  # empty
        bqm.invars_.resize(self.adj_.size())
        bqm.outvars_.resize(2*self.num_interactions)

        cdef OutVar outvar
        cdef VarIndex u
        cdef size_t outvar_idx = 0
        for u in range(self.adj_.size()):
            
            # set the linear bias
            bqm.invars_[u].second = self.adj_[u].second
            bqm.invars_[u].first = outvar_idx

            # set the quadratic biases
            for outvar in self.adj_[u].first:
                bqm.outvars_[outvar_idx] = outvar
                outvar_idx += 1

        return bqm

    def to_lists(self, object sort_and_reduce=True):
        """Dump to a list of lists, mostly for testing"""
        return list((list(neighbourhood.items()), bias)
                    for neighbourhood, bias in self.adj_)
