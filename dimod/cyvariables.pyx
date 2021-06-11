# Copyright 2021 D-Wave Systems Inc.
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

from numbers import Number

from cpython.long cimport PyLong_Check
from cpython.dict cimport PyDict_Size, PyDict_Contains
from cpython.ref cimport PyObject

from dimod.utilities import iter_safe_relabels

cdef extern from "Python.h":
    # not yet available as of cython 0.29.22
    PyObject* PyDict_GetItemWithError(object p, object key) except? NULL


cdef class cyVariables:
    def __init__(self, object iterable=None):
        self._index_to_label = dict()
        self._label_to_index = dict()
        self._stop = 0

        if iterable is not None:
            for v in iterable:
                self._append(v, permissive=True)

    def __contains__(self, v):
        return bool(self.count(v))

    # todo: support slices
    def __getitem__(self, Py_ssize_t idx):
        return self.at(idx)

    def __len__(self):
        return self.size()

    cpdef object _append(self, object v=None, bint permissive=False):
        """Append a new variable.

        Args:
            v (hashable, optional):
                Add a new variable. If `None`, a new label will be generated.
                The generated label will be the index of the new variable if
                that index is available, otherwise it will be the lowest
                available non-negative integer.

            permissive (bool, optional, default=False):
                If `False`, appending a variable that already exists will raise
                a `ValueError`. If `True`, appending a variable that already
                exists will not change the container.

        Returns:
            hashable: The label of the appended variable.

        Raises:
            ValueError: If the variable is present and `permissive` is
            False.

        This method is semi-public. it is intended to be used by
        classes that have :class:`.Variables` as an attribute, not by the
        the user.
        """
        if v is None:
            v = self._stop

            if not self._is_range() and self.count(v):
                v = 0
                while self.count(v):
                    v += 1

        elif self.count(v):
            if permissive:
                return v
            else:
                raise ValueError('{!r} is already a variable'.format(v))

        idx = self._stop

        if idx != v:
            self._label_to_index[v] = idx
            self._index_to_label[idx] = v

        self._stop += 1
        return v

    cpdef bint _is_range(self):
        """Return whether the variables are currently labelled [0, n)."""
        return not PyDict_Size(self._label_to_index)

    cpdef object _extend(self, object iterable, bint permissive=False):
        """Add new variables.

        Args:
            iterable (iterable[hashable], optional):
                An iterable of hashable objects.

            permissive (bool, optional, default=False):
                If `False`, appending a variable that already exists will raise
                a `ValueError`. If `True`, appending a variable that already
                exists will not change the container.

        Raises:
            ValueError: If a variable is present and `permissive` is
            False.

        This method is semi-public. it is intended to be used by
        classes that have :class:`.Variables` as an attribute, not by the
        the user.
        """
        # todo: performance improvements for range etc
        for v in iterable:
            self._append(v, permissive=permissive)

    cpdef object _pop(self):
        """Remove the last variable.

        This method is semi-public. it is intended to be used by
        classes that have :class:`.Variables` as an attribute, not by the
        the user.
        """
        if not self:
            raise IndexError("Cannot pop when Variables is empty")

        self._stop = idx = self._stop - 1

        label = self._index_to_label.pop(idx, idx)
        self._label_to_index.pop(label, None)
        return label

    def _relabel(self, mapping):
        """Relabel the variables in-place.

        Args:
            mapping (dict):
                Mapping from current variable labels to new, as a dict. If
                an incomplete mapping is specified, unmapped variables keep
                their current labels.

        This method is semi-public. it is intended to be used by
        classes that have :class:`.Variables` as an attribute, not by the
        the user.
        """
        for submap in iter_safe_relabels(mapping, self):
            for old, new in submap.items():
                if old == new:
                    continue

                idx = self._label_to_index.pop(old, old)

                if new != idx:
                    self._label_to_index[new] = idx
                    self._index_to_label[idx] = new  # overwrites old idx
                else:
                    self._index_to_label.pop(idx, None)

    def _relabel_as_integers(self):
        """Relabel the variables as integers in-place.

        Returns:
            dict: A mapping that will restore the original labels.

        Examples:

            >>> variables = dimod.variables.Variables(['a', 'b', 'c', 'd'])
            >>> print(variables)
            Variables(['a', 'b', 'c', 'd'])
            >>> mapping = variables._relabel_as_integers()
            >>> print(variables)
            Variables([0, 1, 2, 3])
            >>> variables._relabel(mapping)  # restore the original labels
            >>> print(variables)
            Variables(['a', 'b', 'c', 'd'])

        This method is semi-public. it is intended to be used by
        classes that have :class:`.Variables` as an attribute, not by the
        the user.
        """
        mapping = self._index_to_label.copy()
        self._index_to_label.clear()
        self._label_to_index.clear()
        return mapping

    cdef object at(self, Py_ssize_t idx):
        """Get variable `idx`.

        This method is useful for accessing from cython since __getitem__ goes
        through python.
        """
        if idx < 0:
            idx = self._stop + idx

        if idx >= self._stop:
            raise IndexError('index out of range')

        cdef object v
        cdef object pyidx = idx
        cdef PyObject* obj
        if self._is_range():
            v = pyidx
        else:
            # faster than self._index_to_label.get
            obj = PyDict_GetItemWithError(self._index_to_label, pyidx)
            if obj == NULL:
                v = pyidx
            else:
                v = <object>obj  # correctly handles the ref count

        return v

    cdef Py_ssize_t _count_int(self, object v) except -1:
        # only works when v is an int
        cdef Py_ssize_t vi = v

        if self._is_range():
            return 0 <= vi < self._stop

        # need to make sure that we're not using the integer elsewhere
        return (0 <= vi < self._stop
                and not PyDict_Contains(self._index_to_label, v)
                or PyDict_Contains(self._label_to_index, v))

    cpdef Py_ssize_t count(self, object v) except -1:
        """Return the number of times `v` appears in the variables.

        Because the variables are always unique, this will always return 1 or 0.
        """
        if PyLong_Check(v):
            return self._count_int(v)

        # handle other numeric types
        if isinstance(v, Number):
            v_int = int(v)  # assume this is safe because it's a number
            if v_int == v:
                return self._count_int(v_int)  # it's an integer afterall!

        try:
            return v in self._label_to_index
        except TypeError:
            # unhashable
            return False

    cpdef Py_ssize_t index(self, object v, bint permissive=False) except -1:
        """Return the index of `v`.

        Args:
            v (hashable):
                A variable.

            permissive (bool, optional, default=False):
                If True, the variable will be inserted, guaranteeing an index
                can be returned.

        Returns:
            int: The index of the given variable.

        Raises:
            ValueError: If the variable is not present and `permissive` is
            False.

        """
        if permissive:
            self._append(v, permissive=True)
        if not self.count(v):
            raise ValueError('unknown variable {!r}'.format(v))

        if self._is_range():
            return v if PyLong_Check(v) else int(v)

        # faster than self._label_to_index.get
        cdef PyObject* obj = PyDict_GetItemWithError(self._label_to_index, v)
        if obj == NULL:
            pyobj = v
        else:
            pyobj = <object>obj  # correctly updates ref count

        return pyobj if PyLong_Check(pyobj) else int(pyobj)

    cdef Py_ssize_t size(self):
        """The number of variables.

        This method is useful for accessing from cython since __len__ goes
        through python.
        """
        return self._stop
