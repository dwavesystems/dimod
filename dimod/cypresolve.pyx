# Copyright 2022 D-Wave Systems Inc.
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

from libcpp.vector cimport vector
from libcpp.utility cimport move as cppmove

import numpy as np

from dimod.constrained.cyconstrained cimport cyConstrainedQuadraticModel
from dimod.cyutilities cimport ConstNumeric
from dimod.sampleset import as_samples
from dimod.variables import Variables

cdef class cyPreSolver:
    def __init__(self, cyConstrainedQuadraticModel cqm, bint move = False):
        self.variables = cqm.variables  # always a copy
        
        if move:
            self.cpppresolver = cppPreSolver[bias_type, index_type, double](cppmove(cqm.cppcqm))
            # todo: finish deconstructing cqm
        else:
            self.cpppresolver = cppPreSolver[bias_type, index_type, double](cqm.cppcqm)

    def apply(self):
        self.cpppresolver.apply()

    def load_default_presolvers(self):
        self.cpppresolver.load_default_presolvers()

    def clear_cqm(self):
        pass  # currently does nothing.


    def _restore_samples(self, ConstNumeric[:, ::1] samples):

        cdef Py_ssize_t num_samples = samples.shape[0]
        cdef Py_ssize_t num_variables = samples.shape[1]

        cdef double[:, ::1] original_samples = np.empty((num_samples, self.variables.size()), dtype=np.double)

        cdef vector[double] original
        cdef vector[double] reduced
        for i in range(num_samples):
            reduced.clear()
            for vi in range(num_variables):
                reduced.push_back(samples[i, vi])

            original = self.cpppresolver.postsolver().apply(reduced)

            for vi in range(original.size()):
                original_samples[i, vi] = original[vi]

        return original_samples

    def restore_samples(self, samples_like):
        samples, labels = as_samples(samples_like, labels_type=Variables)

        if not labels.is_range:
            raise ValueError("expected samples to be integer labelled")

        if samples.shape[1] != self.cpppresolver.model().num_variables():
            raise ValueError

        # we need contiguous and unsigned. as_samples actually enforces contiguous
        # but no harm in double checking for some future-proofness
        samples = np.ascontiguousarray(
                samples,
                dtype=f'i{samples.dtype.itemsize}' if np.issubdtype(samples.dtype, np.unsignedinteger) else None,
                )

        restored = self._restore_samples(samples)

        return np.asarray(restored)
