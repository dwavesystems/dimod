# distutils: sources = dwave/preprocessing/src/presolve.cpp dwave/preprocessing/src/exceptions.cpp

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

import enum as pyenum

cimport cython

from libcpp.vector cimport vector
from libcpp.utility cimport move as cppmove

import numpy as np

import dimod

from dimod.libcpp cimport ConstrainedQuadraticModel as cppConstrainedQuadraticModel
from dimod.constrained.cyconstrained cimport cyConstrainedQuadraticModel, make_cqm

from dwave.preprocessing.libcpp cimport Feasibility as cppFeasibility
from dwave.preprocessing.libcpp cimport TechniqueFlags as cppTechniqueFlags
from dwave.preprocessing.libcpp cimport duration
from dwave.preprocessing.presolve.exceptions import InvalidModelError

# We want to establish a relationship between presolveimpl.hpp and this file, so that
# changes to presolveimpl.hpp will trigger a rebuild.
cdef extern from "../src/presolveimpl.hpp" namespace "dwave::presolve" nogil:
    pass


# In Cython3 we'll be able to import this from C++ directly, but for now we duplicate
# Dev note: must be kept synced with cypresolve.pyi and with the C++ version
class Feasibility(pyenum.Enum):
    """An :py:class:`~enum.Enum` to signal whether a model is feasible or not.

    Attributes:
        Infeasible: The model is known to be infeasible
        Feasible: The model is known to be feasible
        Unknown: It is not known if the model is feasible or not

    """
    Infeasible = 0
    Feasible = 1
    Unknown = 2


# In Cython3 we'll be able to import this from C++ directly, but for now we duplicate
# Dev note: must be kept synced with cypresolve.pyi and with the C++ version
class TechniqueFlags(pyenum.IntFlag):
    """An :py:class:`~enum.IntFlag` to define which presolve techniques will be
    used by the presolver.

    Attributes:
        None_: No techniques.

        RemoveRedundantConstraints:
            Remove redundant constraints.
            See Achterberg et al., section 3.1.

        RemoveSmallBiases:
            Remove small biases from the objective and constraints.
            See Achterberg et al., section 3.1.

        DomainPropagation:
            Use constraints to tighten the bounds on variables.
            See Achterberg et al., section 3.2.

        All:
            All techniques.

        Default:
            Currently equivalent to ``All``, though this may change in the future.

    """
    None_ = 0

    RemoveRedundantConstraints = 1 << 0

    RemoveSmallBiases = 1 << 1

    DomainPropagation = 1 << 2

    All = 0xffffffffffffffff

    Default = All


cdef class cyPresolver:
    def __cinit__(self, cyConstrainedQuadraticModel cqm, *, bint move = False):
        self._original_variables = cqm.variables.copy()  # todo: implement Variables.swap()

        if move:
            self.cpppresolver = new cppPresolver[bias_type, index_type, double](cppmove(cqm.cppcqm))

            # clear out any remaining variables etc in the original model
            cqm.clear()
        else:
            self.cpppresolver = new cppPresolver[bias_type, index_type, double](cqm.cppcqm)

        # we need this because we may detach the model later
        self._model_num_variables = self.cpppresolver.model().num_variables()

    def __init__(self, cyConstrainedQuadraticModel cqm, *, bint move = False):
        pass

    def __dealloc__(self):
        if self.cpppresolver is not NULL:
            del self.cpppresolver
            self.cpppresolver = NULL

    # add_techniques is implemeted at the Python level because it doesn't need access
    # to the underlying C++ objects

    cpdef bint apply(self) except*:
        """Normalize and presolve the held constraint quadratic model.

        Returns:
            A boolean indicating whether the model was modified by presolve.

        Raises:
            :exc:`InvalidModelError`: If the model is ill-constructed or otherwise
                not valid. This is disctinct from infeasibility.
        """
        # Use | to avoid short-circuiting
        return self.normalize() | self.presolve()

    def clear_model(self):
        """Clear the held constrained quadratic model. This is useful to save memory."""
        self.cpppresolver.detach_model()

    def copy_model(self):
        """Return a copy of the held constrained quadratic model."""
        cdef cppConstrainedQuadraticModel[bias_type, index_type] tmp = self.cpppresolver.model()  # copy
        return make_cqm(cppmove(tmp))  # then move

    def detach_model(self):
        """Create a :class:`dimod.ConstrainedQuadraticModel` from the held model.

        Subsequent attempts to access the held model raise a :exc:`RuntimeError`.
        """
        return make_cqm(cppmove(self.cpppresolver.detach_model()))

    def feasibility(self):
        """Return the feasibility of the model."""
        # Cython3 will support this automatically but for now we just do the
        # explicit check

        if self.cpppresolver.feasibility() == cppFeasibility.Infeasible:
            return Feasibility.Infeasible
        elif self.cpppresolver.feasibility() == cppFeasibility.Unknown:
            return Feasibility.Unknown
        elif self.cpppresolver.feasibility() == cppFeasibility.Feasible:
            return Feasibility.Feasible
        else:
            # sanity check
            raise RuntimeError("unexpected Feasibility")

    cpdef bint normalize(self) except*:
        """Normalize the held constrained quadratic model.

        Returns:
            A boolean indicating whether the model was modified by presolve.

        Raises:
            :exc:`InvalidModelError`: If the model is ill-constructed or otherwise
                not valid. This is disctinct from infeasibility.
        """
        cdef bint changes = False

        try:
            with nogil:
                self.mutex.lock()  # do this once the gil has been released to avoid deadlocks
                changes = self.cpppresolver.normalize()
        except RuntimeError as err:
            # The C++ InvalidModelError is interpreted by Cython as a RuntimeError
            # We could put in a bunch of code to reinterpret it, but because this
            # is the only error type that should be raised by normalize() we just
            # do it naively. 
            raise InvalidModelError(err)
        finally:
            self.mutex.unlock()

        # Save the new size for later use in restore_samples
        self._model_num_variables = self.cpppresolver.model().num_variables()

        return changes

    cpdef bint presolve(self, double time_limit_s = float("inf")) except*:
        """Apply any loaded presolve techniques to the held constrained quadratic model.

        Must be called after :meth:`normalize`.

        Args:
            time_limit_s:
                A time limit in seconds.
                The presolve rounds will terminate after the time limit is exceeded.
                Defaults to ``float("inf")``.

        Returns:
            A boolean indicating whether the model was modified by presolve.

        Raises:
            TypeError: If called before :class:`normalize()`.
        """
        cdef bint changes = False

        try:
            with nogil:
                self.mutex.lock()  # do this once the gil has been released to avoid deadlocks
                changes = self.cpppresolver.presolve(duration[double](time_limit_s))
        except RuntimeError as err:
            # The C++ logic_error is interpreted by Cython as a RuntimeError.
            # The only errors here should be for a model that's not normalized.
            raise TypeError(err)
        finally:
            self.mutex.unlock()  # it's ok to do this inside the GIL

        # Save the new size for later use in restore_samples
        self._model_num_variables = self.cpppresolver.model().num_variables()

        return changes

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Py_ssize_t restore_sample(self, const Numeric[::1] reduced_sample, double[::1] original_sample) except -1 nogil:
        """Restore a sample.

        Note that this method is not thread safe, that is it does not aquire
        the mutex. The mutex should be aquired before calling this method.

        Args:
            reduced_sample: a solution to the presolved model.
            original_sample: a memoryview that will hold the restored solution.

        """
        cdef Py_ssize_t num_variables = reduced_sample.shape[0]

        # Convert the sample into a vector
        cdef vector[double] reduced
        for vi in range(num_variables):
            reduced.push_back(reduced_sample[vi])

        cdef vector[double] original

        original = self.cpppresolver.restore(reduced)

        if <Py_ssize_t>original.size() != original_sample.shape[0]:
            raise RuntimeError("unexpected reduced variables size")

        for vi in range(<Py_ssize_t>original.size()):
            original_sample[vi] = original[vi]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _restore_samples(self, const Numeric[:, ::1] samples):
        cdef Py_ssize_t num_samples = samples.shape[0]
        cdef Py_ssize_t num_variables = samples.shape[1]

        cdef double[:, ::1] original_samples = np.empty((num_samples, self._original_variables.size()), dtype=np.double)

        try:
            with nogil:
                self.mutex.lock()  # do this once the gil has been released to avoid deadlocks
                for i in range(num_samples):
                    self.restore_sample(samples[i, :], original_samples[i, :])
        finally:
            self.mutex.unlock()  # it's ok to do this inside the GIL

        return original_samples

    def restore_samples(self, samples_like):
        """Restore the original variable labels to a set of reduced samples.

        Args:
            samples_like: A :class:`dimod.types.SamplesLike`. The samples must
                be index-labeled.

        Returns:
            Tuple:
                A 2-tuple where the first entry is the restored samples and the second
                is the original labels.

        """
        samples, labels = dimod.as_samples(samples_like, labels_type=dimod.variables.Variables)

        if not labels.is_range:
            raise ValueError("expected samples to be integer labelled")

        if samples.shape[1] != self._model_num_variables:
            raise ValueError(f"sample(s) must have {self._model_num_variables} variables, "
                             f"given sample(s) have {samples.shape[1]}")

        # we need contiguous and unsigned. as_samples actually enforces contiguous
        # but no harm in double checking for some future-proofness
        samples = np.ascontiguousarray(
                samples,
                dtype=f'i{samples.dtype.itemsize}' if np.issubdtype(samples.dtype, np.unsignedinteger) else None,
                )

        restored = self._restore_samples(samples)

        return np.asarray(restored), self._original_variables

    def set_techniques(self, techniques):
        """Set the presolve techniques to be used by the presolver.

        Args:
            techniques (:class:`TechniqueFlags`):
                The techniques to be used.

        Returns:
            :class:`TechniqueFlags`: The currently loaded presolve techniques.

        """
        if not 0 <= techniques.value < (1 << 64):
            raise ValueError("techniques.value must be castable to uint64")

        self.cpppresolver.set_techniques(<cppTechniqueFlags>(techniques.value))
        return self.techniques()

    def techniques(self):
        """Report the presolve techniques to be used by the presolver.

        Returns:
            :class:`TechniqueFlags`: The currently loaded presolve techniques.

        """
        # Convert from C++ to Python
        return TechniqueFlags(self.cpppresolver.techniques())
