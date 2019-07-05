# distutils: language = c++
# cython: language_level=3

# google c++ convention for names
# https://google.github.io/styleguide/cppguide.html#Naming

from numbers import Integral

import numpy as np

bias_dtype = np.float64  # hardcoded, we might want to change this later

cdef class AdjArrayBQM:
    """

    This can be instantiated in several ways:

        AdjVectorBQM()
            Creates an empty binary quadratic model

        AdjVectorBQM(n)
            Where n is the number of nodes.

        AdjVectorBQM((linear, [quadratic, [offset]]))
            Where linear, quadratic are:
                dict[int, bias]
                sequence[bias]
            *NOT IMPLEMENTED YET*

        AdjVectorBQM(bqm)
            Where bqm is another binary quadratic model (equivalent to
            bqm.to_adjvector())
            *NOT IMPLEMENTED YET*

        AdjVectorBQM(D)
            Where D is a dense matrix D

    """

    def __init__(self, object arg1=0):
        # this is the only method that we treat invars_ and outvars_ as vectors
        # rather than arrays
        
        cdef Bias [:, :] D  # in case it's dense
        cdef size_t num_variables, num_interactions, degree
        cdef VarIndex u, v
        cdef Bias b

        if isinstance(arg1, Integral):
            self.invars_.resize(arg1)
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

            self.invars_.resize(num_variables)

            # we could grow the vectors going through it one at a time, but
            # in the interest of future-proofing we will go through once,
            # resize the outvars_ then go through it again to fill

            # figure out the degree of each variable and consequently the
            # number of interactions
            num_interactions = 0  # 2x num_interactions because count degree
            for u in range(num_variables):
                degree = 0
                for v in range(num_variables):
                    if u != v and (D[v, u] or D[u, v]):
                        degree += 1

                if u < num_variables - 1:
                    self.invars_[u + 1].first = degree + self.invars_[u].first

                num_interactions += degree

            self.outvars_.resize(num_interactions)

            for u in range(num_variables):
                degree = 0
                for v in range(num_variables):
                    if u == v:
                        self.invars_[u].second = D[u, v]
                    elif D[v, u] or D[u, v]:
                        self.outvars_[self.invars_[u].first + degree].first = v
                        self.outvars_[self.invars_[u].first + degree].second = D[v, u] + D[u, v]
                        degree += 1

                    
    def __len__(self):
        return self.invars_.size()

    @property
    def num_interactions(self):
        return self.outvars_.size() // 2

    @property
    def num_variables(self):
        return len(self)

    @property
    def shape(self):
        return self.num_variables, self.num_interactions

    def to_lists(self):
        """Dump to two lists, mostly for testing"""
        return list(self.invars_), list(self.outvars_)
