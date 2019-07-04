# distutils: language = c++
# cython: language_level=3

# google c++ convention for names
# https://google.github.io/styleguide/cppguide.html#Naming

cdef class AdjArrayBQM:

    def __cinit__(self, size_t num_variables, size_t num_interactions):
        # this is the only place that we treat these as vectors rather than
        # arrays
        self.invars_.resize(num_variables + 1)
        self.outvars_.resize(2 * num_interactions)

    def __len__(self):
        return self.invars_.size() - 1

    @property
    def shape(self):
        return len(self), self.invars_.back().first // 2

    def get_linear(self, VarIndex v):
        if not self.has_variable(v):
            raise ValueError
        return self.invars_[v].second

    def has_variable(self, VarIndex u):
        # nb: fails for negative with OverflowError, but we check anyway
        # in case we change to signed at some point
        return u < len(self) and u >= 0

    def to_lists(self, object ignore_trailing=True):
        """Dump to two lists, mostly for testing"""
        if ignore_trailing:
            return list(self.invars_[:-1]), list(self.outvars_)
        else:
            return list(self.invars_), list(self.outvars_)
