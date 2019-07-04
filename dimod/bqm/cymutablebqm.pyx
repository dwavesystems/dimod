# distutils: language = c++
# cython: language_level=3

# google c++ convention for names
# https://google.github.io/styleguide/cppguide.html#Naming

from numbers import Integral

from cython.operator cimport dereference as deref, preincrement, predecrement, postincrement

from libc.math cimport isnan
from libcpp.algorithm cimport lower_bound
from libcpp.pair cimport pair
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

from dimod.bqm.cybqm cimport AdjArrayBQM


cdef extern from "<algorithm>" namespace "std" nogil:
    void stable_sort[Iter, Compare](Iter first, Iter last, Compare comp)

ctypedef np.float64_t Bias
ctypedef np.uint32_t VarIndex

ctypedef pair[VarIndex, Bias] OutVar
ctypedef pair[vector[OutVar], Bias] InVar

cdef bint lt_first(const OutVar a, const OutVar b):
    # we need a custom sorter for pairs that only looks at the variable index
    # for use in our stable sort
    return a.first < b.first


cdef class AdjVectorBQM:
    cdef vector[InVar] adj_
    cdef vector[bint] sorted_  # whether currently sorted

    def __init__(self, object num_variables=0):
        if isinstance(num_variables, Integral):
            for _ in range(num_variables):
                self.append_variable()
        else:
            raise NotImplementedError

    def __len__(self):
        # number of variables
        return self.adj_.size()

    @property
    def num_interactions(self):
        self.sort_and_reduce()

        # O(|V|) if we didn't have to resolve, we might just want to track this
        cdef InVar in_var
        cdef size_t count = 0
        for in_var in self.adj_:
            count += in_var.first.size()
        return count // 2

    @property
    def shape(self):
        return len(self), self.num_interactions

    def add_interaction(self, VarIndex u, VarIndex v, Bias b):
        if u >= len(self) or v >= len(self) or u < 0 or v < 0:
            # if VarIndex is an unsigned int, then < 0 will raise an OverflowError
            raise ValueError

        # don't worry if there are duplicates at this point, just append them
        # onto the end
        self.adj_[v].first.push_back((u, b))
        self.adj_[u].first.push_back((v, b))

    def append_variable(self, Bias bias=0):
        self.adj_.push_back(([], bias))  # empty neighbourhood

    def get_linear(self, VarIndex v):
        if v >= len(self) or v < 0:
            # if VarIndex is an unsigned int, then < 0 will raise an OverflowError
            raise ValueError
        return self.adj_[v].second

    def get_quadratic(self, VarIndex u, VarIndex v):
        if u >= len(self) or v >= len(self) or u < 0 or v < 0:
            # if VarIndex is an unsigned int, then < 0 will raise an OverflowError
            raise ValueError

        self.sort_and_reduce()  # we're reading, so resolve

        cdef OutVar target = (v, 0)

        # everything is now unique and sorted, so we can do a binary search
        cdef vector[OutVar].iterator it = lower_bound(self.adj_[u].first.begin(),
                                                      self.adj_[u].first.end(),
                                                      target,
                                                      lt_first)

        if it == self.adj_[u].first.end():
            raise ValueError

        cdef OutVar ans = deref(it)

        if ans.first != v:
            raise ValueError("There is no interaction between {} and {}".format(u, v))

        return ans.second

    def remove_interaction(self, VarIndex u, VarIndex v):
        self.add_interaction(u, v, float('nan'))  # is there a better way to get nan?

    def set_linear(self, VarIndex v, Bias bias):
        if v >= len(self) or v < 0:
            # if VarIndex is an unsigned int, then < 0 will raise an OverflowError
            raise ValueError
        self.adj_[v].second = bias

    def sort_and_reduce(self):
        cdef vector[OutVar].iterator first, last, result

        cdef vector[InVar].iterator it = self.adj_.begin()

        while it != self.adj_.end():

            first = deref(it).first.begin()
            last = deref(it).first.end()

            # sort each neighbourhood with a stable sort so that we can account
            # for the deletions (we need to know where the NaNs are)
            stable_sort(first, last, lt_first)

            if first != last:
                result = first

                while preincrement(first) != last:
                    if deref(result).first == deref(first).first:
                        if isnan(deref(first).second):
                            # there was a deletion in the queue, so step back
                            predecrement(result)
                        else:
                            # accumulate the biases
                            deref(result).second = deref(result).second + deref(first).second
                    elif isnan(deref(first).second):
                        # if the first of a new index is a NaN then we can
                        # ignore it
                        pass  
                    else:
                        # new index
                        preincrement(result)[0] = first[0]

                # finally resize down
                preincrement(result)
                if result != last:
                    deref(it).first.resize((result - deref(it).first.begin()))

            preincrement(it)

    def to_adjarray(self):
        # this is always a copy
        self.sort_and_reduce()

        cdef AdjArrayBQM bqm = AdjArrayBQM(len(self), self.num_interactions)

        # these are hard-coded for now, but we obviously want to import them
        cdef vector[pair[size_t, Bias]].iterator invar_iter = bqm.invars_.begin()
        cdef vector[pair[VarIndex, Bias]].iterator outvar_iter = bqm.outvars_.begin()

        cdef vector[InVar].iterator adj_iter = self.adj_.begin()
        cdef vector[OutVar].iterator neighbourhood_iter

        cdef size_t out_loc = 0

        while adj_iter != self.adj_.end():

            # set the linear bias
            deref(invar_iter).second = deref(adj_iter).second

            # set the quadratic biases
            deref(invar_iter).first = out_loc
            neighbourhood_iter = deref(adj_iter).first.begin()

            while neighbourhood_iter != deref(adj_iter).first.end():
                postincrement(outvar_iter)[0] = postincrement(neighbourhood_iter)[0]
                out_loc += 1

            # onto the next!
            preincrement(invar_iter)
            preincrement(adj_iter)

        return bqm

    def to_lists(self, object sort_and_reduce=True):
        """Dump to a list of lists, mostly for testing"""
        if sort_and_reduce:
            self.sort_and_reduce()
        return list((list(neighbourhood), bias)
                    for neighbourhood, bias in self.adj_)
