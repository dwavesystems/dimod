# Copyright 2018 D-Wave Systems Inc.
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
#
# ================================================================================================
from __future__ import division

import itertools

from collections import ItemsView, Iterable, Mapping, Sized, ValuesView
from collections import namedtuple

import numpy as np

from dimod.decorators import vartype_argument
from dimod.utilities import resolve_label_conflict
from dimod.vartypes import Vartype


class Response(Iterable, Sized):

    @vartype_argument('vartype')
    def __init__(self, record, labels, info, vartype):

        # make sure that record is a numpy recarray and that it has the expected fields
        if not isinstance(record, np.recarray):
            raise TypeError("input record must be a numpy recarray")
        elif not {'sample', 'energy', 'num_occurrences'}.issubset(record.dtype.fields):
            raise ValueError("input record must have 'sample', 'energy' and 'num_occurrences' as fields")
        self._record = record

        num_samples, num_variables = record.sample.shape

        if not isinstance(labels, list):
            labels = list(labels)
        if len(labels) != num_variables:
            msg = ("mismatch between number of variables in record.sample ({}) "
                   "and labels ({})").format(num_variables, len(labels))
            raise ValueError(msg)
        self._labels = labels
        self._label_to_idx = {v: idx for idx, v in enumerate(labels)}

        # cast info to a dict if it's a mapping or similar
        if not isinstance(info, dict):
            info = dict(info)
        self._info = info

        # vartype is checked by vartype_argument decorator
        self._vartype = vartype

    def __len__(self):
        """The number of rows in record."""
        return self.record.__len__()

    def __iter__(self):
        """Iterate over the samples, low energy to high."""
        return self.samples(sorted_by='energy')

    def __repr__(self):
        return 'Response({}, {}, {}, {})'.format(self.record.__repr__(),
                                                 self.variable_labels.__repr__(),
                                                 self.info.__repr__(),
                                                 self.vartype.name.__repr__())

    ###############################################################################################
    # Properties
    ###############################################################################################

    @property
    def record(self):
        if hasattr(self, '_future'):
            self._resolve_future()
        # subsequent calls will bypass the _future check
        self.__dict__['record'] = record = self._record
        return record

    @property
    def variable_labels(self):
        if hasattr(self, '_future'):
            self._resolve_future()
        # subsequent calls will bypass the _future check
        self.__dict__['variable_labels'] = labels = self._labels
        return labels

    @property
    def label_to_idx(self):
        if hasattr(self, '_future'):
            self._resolve_future()
        # subsequent calls will bypass the _future check
        self.__dict__['label_to_idx'] = label_to_idx = self._label_to_idx
        return label_to_idx

    @property
    def info(self):
        if hasattr(self, '_future'):
            self._resolve_future()
        # subsequent calls will bypass the _future check
        self.__dict__['info'] = info = self._info
        return info

    @property
    def vartype(self):
        if hasattr(self, '_future'):
            self._resolve_future()
        # subsequent calls will bypass the _future check
        self.__dict__['vartype'] = vartype = self._vartype
        return self._vartype

    ###############################################################################################
    # Views
    ###############################################################################################

    def done(self):
        return (not hasattr(self, '_future')) or self._future.done()

    def samples(self, n=None, sorted_by='energy'):
        if n is None:
            for sample, in self.data(['sample'], sorted_by=sorted_by, name=None):
                yield sample
        else:
            for sample in itertools.islice(self.samples(n=None, sorted_by=sorted_by), n):
                yield sample

    def data(self, fields=None, sorted_by='energy', name='Sample'):
        record = self.record

        if fields is None:
            # make sure that sample, energy is first
            fields = ['sample', 'energy', 'num_occurrences'] + [field for field in record.dtype.fields
                                                                if field not in {'sample', 'energy', 'num_occurrences'}]

        if sorted_by is None:
            order = np.arange(len(self))
        else:
            order = np.argsort(record[sorted_by])

        if name is None:
            # yielding a tuple
            def _pack(values):
                return tuple(values)
        else:
            # yielding a named tuple
            SampleTuple = namedtuple(name, fields)

            def _pack(values):
                return SampleTuple(*values)

        def _values(idx):
            for field in fields:
                if field == 'sample':
                    yield SampleView(idx, self)
                else:
                    yield record[field][idx]

        for idx in order:
            yield _pack(_values(idx))

    ###############################################################################################
    # Constructors
    ###############################################################################################

    @classmethod
    def from_future(cls, future, result_hook=None):
        obj = cls.__new__(cls)
        obj._future = future

        if result_hook is None:
            def result_hook(future):
                return future.result()
        elif not callable(result_hook):
            raise TypeError("expected result_hook to be callable")

        obj._result_hook = result_hook
        return obj

    def _resolve_future(self):
        response = self._result_hook(self._future)
        self.__init__(response.record, response.variable_labels, response.info, response.vartype)
        del self._future
        del self._result_hook

    @classmethod
    def from_samples(cls, samples_like, vectors, info, vartype, labels=None):

        # there is no np.is_array_like so we use a try-except block
        try:
            # trying to cast it to int8 rules out list of dictionaries. If we didn't try to cast
            # then it would just create a vector of np.object
            samples = np.asarray(samples_like, dtype=np.int8)
        except TypeError:
            # if labels are None, they are set here
            samples, labels = _samples_dicts_to_array(samples_like, labels)

        assert samples.dtype == np.int8, 'sanity check'

        record = data_struct_array(samples, **vectors)

        # if labels are still None, set them here. We could do this in an else in the try-except
        # block, but the samples-array might not have the correct shape
        if labels is None:
            __, num_variables = record.sample.shape
            labels = list(range(num_variables))

        return cls(record, labels, info, vartype)

    ###############################################################################################
    # Methods
    ###############################################################################################

    def copy(self):
        """Create a shallow copy.
        """
        return Response(self.record.copy(), list(self.variable_labels), self.info.copy(), self.vartype)

    @vartype_argument('vartype')
    def change_vartype(self, vartype, energy_offset=0.0, inplace=True):
        if not inplace:
            return self.copy().change_vartype(vartype, energy_offset, inplace=True)

        if energy_offset:
            self.record.energy = self.record.energy + energy_offset

        if vartype is self.vartype:
            return self  # we're done!

        if vartype is Vartype.SPIN and self.vartype is Vartype.BINARY:
            self.record.sample = 2 * self.record.sample - 1
            self._vartype = vartype
        elif vartype is Vartype.BINARY and self.vartype is Vartype.SPIN:
            self.record.sample = (self.record.sample + 1) // 2
            self._vartype = vartype
        else:
            raise ValueError("Cannot convert from {} to {}".format(self.vartype, vartype))

        return self

    def relabel_variables(self, mapping, inplace=True):
        if not inplace:
            return self.copy().relabel_variables(mapping, inplace=True)

        try:
            old_labels = set(mapping.keys())
            new_labels = set(mapping.values())
        except TypeError:
            raise ValueError("mapping targets must be hashable objects")

        for v in self.variable_labels:
            if v in new_labels and v not in old_labels:
                raise ValueError(('A variable cannot be relabeled "{}" without also relabeling '
                                  "the existing variable of the same name").format(v))

        shared = old_labels & new_labels
        if shared:
            old_to_intermediate, intermediate_to_new = resolve_label_conflict(mapping, old_labels, new_labels)

            self.relabel_variables(old_to_intermediate, inplace=True)
            self.relabel_variables(intermediate_to_new, inplace=True)
            return self

        self._labels = labels = [mapping.get(v, v) for v in self.variable_labels]
        self._label_to_idx = {v: idx for idx, v in enumerate(labels)}
        return self

    ###############################################################################################
    # Deprecated properties
    ###############################################################################################

    @property
    def samples_matrix(self):
        """:obj:`numpy.ndarray`: Samples as a NumPy 2D array of data type int8.

        Examples:
            This example shows the samples of dimod package's ExactSolver reference sampler
            formatted as a NumPy array.

            >>> import dimod
            >>> response = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> response.samples_matrix
            array([[-1, -1],
                   [ 1, -1],
                   [ 1,  1],
                   [-1,  1]])

        """
        import warnings
        warnings.warn("Response.samples_matrix is deprecated, please use Response.record.sample instead.",
                      DeprecationWarning)
        return self.record['sample']

    @samples_matrix.setter
    def samples_matrix(self, mat):
        import warnings
        warnings.warn("Response.samples_matrix is deprecated, please use Response.record.sample instead.",
                      DeprecationWarning)
        self.record['sample'] = mat

    @property
    def data_vectors(self):
        """dict[field, :obj:`numpy.array`/list]: Per-sample data as a dict, where keys are the
        data labels and values are each a vector of the same length as record.samples.

        Examples:
            This example shows the returned energies of dimod package's ExactSolver
            reference sampler.

            >>> import dimod
            >>> response = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> response.data_vectors['energy']
            array([-1.5, -0.5, -0.5,  2.5])

        """
        import warnings
        warnings.warn("Response.data_vectors is deprecated, please use Response.record instead.",
                      DeprecationWarning)
        rec = self.record

        return {field: rec[field] for field in rec.dtype.fields if field != 'sample'}


def _samples_dicts_to_array(samples_dicts, labels):
    """Convert an iterable of samples where each sample is a dict to a numpy 2d array. Also
    determines the labels is they are None.
    """
    itersamples = iter(samples_dicts)

    first_sample = next(itersamples)

    if labels is None:
        labels = list(first_sample)

    num_variables = len(labels)

    def _iter_samples():
        yield np.fromiter((first_sample[v] for v in labels),
                          count=num_variables, dtype=np.int8)

        try:
            for sample in itersamples:
                yield np.fromiter((sample[v] for v in labels),
                                  count=num_variables, dtype=np.int8)
        except KeyError:
            msg = ("Each dict in 'samples' must have the same keys.")
            raise ValueError(msg)

    return np.stack(list(_iter_samples())), labels


class SampleView(Mapping):
    """View each row of the samples record as if it was a dict."""
    def __init__(self, idx, response):
        self._idx = idx  # row of response.record
        self._response = response

    def __getitem__(self, key):
        label_mapping = self._response.label_to_idx
        if label_mapping is not None:
            key = label_mapping[key]
        return int(self._response.record.sample[self._idx, key])

    def __iter__(self):
        # iterate over the variables
        label_mapping = self._response.label_to_idx
        if label_mapping is None:
            __, num_variables = self._response.record.sample.shape
            return iter(range(num_variables))
        return label_mapping.__iter__()

    def __len__(self):
        __, num_variables = self._response.record.sample.shape
        return num_variables

    def __repr__(self):
        """Represents itself as as a dictionary"""
        return dict(self).__repr__()

    def values(self):
        return SampleValuesView(self)

    def items(self):
        return SampleItemsView(self)


class SampleItemsView(ItemsView):
    """Faster read access to the numpy array"""
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        labels = self._mapping._response.variable_labels
        samples_matrix = self._mapping._response.record.sample
        idx = self._mapping._idx
        if labels is None:
            for v, val in enumerate(np.nditer(samples_matrix[idx, :], order='C', op_flags=['readonly'])):
                yield (v, int(val))
        else:
            for v, val in zip(labels, np.nditer(samples_matrix[idx, :], order='C', op_flags=['readonly'])):
                yield (v, int(val))


class SampleValuesView(ValuesView):
    """Faster read access to the numpy array"""
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        samples_matrix = self._mapping._response.record.sample
        for val in np.nditer(samples_matrix[self._mapping._idx, :], op_flags=['readonly']):
            yield int(val)


def data_struct_array(sample, **vectors):  # data_struct_array(sample, *, energy, **vectors):
    """Combine samples and per-sample data into a numpy structured array.

    Args:
        sample (array_like):
            Samples, in any form that can be converted into a numpy array.

        energy (array_like, required):
            Required keyword argument. Energies, in any form that can be converted into a numpy
            1-dimensional array.

        **kwargs (array_like):
            Other per-sample data, in any form that can be converted into a numpy array.

    Returns:
        :obj:`~numpy.ndarray`: A numpy structured array. Has fields ['sample', 'energy', 'num_occurrences', **kwargs]

    """
    if not len(sample):
        # if samples are empty
        sample = np.zeros((0, 0), dtype=np.int8)
    else:
        sample = np.asarray(sample, dtype=np.int8)

        if sample.ndim < 2:
            sample = np.expand_dims(sample, 0)

    num_samples, num_variables = sample.shape

    if 'num_occurrences' not in vectors:
        vectors['num_occurrences'] = [1] * num_samples

    datavectors = {}
    datatypes = [('sample', np.dtype(np.int8), (num_variables,))]

    for kwarg, vector in vectors.items():
        datavectors[kwarg] = vector = np.asarray(vector)

        if len(vector.shape) < 1 or vector.shape[0] != num_samples:
            msg = ('{} and sample have a mismatched shape {}, {}. They must have the same size '
                   'in the first axis.').format(kwarg, vector.shape, sample.shape)
            raise ValueError(msg)

        datatypes.append((kwarg, vector.dtype, vector.shape[1:]))

    if 'energy' not in datavectors:
        # consistent error with the one thrown in python3
        raise TypeError('data_struct_array() needs keyword-only argument energy')
    elif datavectors['energy'].shape != (num_samples,):
        raise ValueError('energy should be a vector of length {}'.format(num_samples))

    data = np.rec.array(np.zeros(num_samples, dtype=datatypes))

    data['sample'] = sample

    for kwarg, vector in datavectors.items():
        data[kwarg] = vector

    return data
