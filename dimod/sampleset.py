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
from collections import Iterable, Sized, Mapping, Iterator
from collections import namedtuple

import numpy as np

from dimod.decorators import vartype_argument
from dimod.utilities import resolve_label_conflict
from dimod.vartypes import Vartype
from dimod.views import VariableIndexView, SampleView

__all__ = ['SampleSet']


class SampleSet(Iterable, Sized):
    __slots__ = ('_info',
                 '_future',
                 '_record',
                 '_result_hook',
                 '_variables',
                 '_vartype')

    _REQUIRED_FIELDS = ['sample', 'energy', 'num_occurrences']

    ###############################################################################################
    # Construction
    ###############################################################################################

    @vartype_argument('vartype')
    def __init__(self, record, variables, info, vartype):

        # make sure that record is a numpy recarray and that it has the expected fields
        if not isinstance(record, np.recarray):
            raise TypeError("input record must be a numpy recarray")
        elif not set(self._REQUIRED_FIELDS).issubset(record.dtype.fields):
            raise ValueError("input record must have {}, {} and {} as fields".format(*self._REQUIRED_FIELDS))
        self._record = record

        num_samples, num_variables = record.sample.shape

        self._variables = variables = VariableIndexView(variables)
        if len(variables) != num_variables:
            msg = ("mismatch between number of variables in record.sample ({}) "
                   "and labels ({})").format(num_variables, len(variables))
            raise ValueError(msg)

        # cast info to a dict if it's a mapping or similar
        if not isinstance(info, dict):
            info = dict(info)
        self._info = info

        # vartype is checked by vartype_argument decorator
        self._vartype = vartype

    @classmethod
    def from_samples(cls, samples_like, vartype, info=None,
                     energy=None, num_occurrences=None, **kwargs):

        # get the samples, variable labels
        samples, variables = as_samples(samples_like)

        num_samples, num_variables = samples.shape

        energy = np.asarray(energy)

        # num_occurrences
        if num_occurrences is None:
            num_occurrences = np.ones(num_samples, dtype=int)
        else:
            num_occurrences = np.asarray(num_occurrences)

        # now construct the record
        datatypes = [('sample', samples.dtype, (num_variables,)),
                     ('energy', energy.dtype),
                     ('num_occurrences', num_occurrences.dtype)]
        for key, vector in kwargs.items():
            kwargs[key] = vector = np.asarray(vector)

            if len(vector.shape) < 1 or vector.shape[0] != num_samples:
                msg = ('{} and sample have a mismatched shape {}, {}. They must have the same size '
                       'in the first axis.').format(kwarg, vector.shape, sample.shape)
                raise ValueError(msg)

            datatypes.append((kwarg, vector.dtype, vector.shape[1:]))

        record = np.rec.array(np.zeros(num_samples, dtype=datatypes))
        record['sample'] = samples
        record['energy'] = energy
        record['num_occurrences'] = num_occurrences
        for kwarg, vector in kwargs.items():
            record[kwargs] = vector

        if info is None:
            info = {}

        return cls(record, variables, info, vartype)

    @classmethod
    def from_future(cls, future, result_hook=None):
        """Construct a response referencing the result of a future computation.

        Args:
            future (object):
                An object that contains or will contain the information needed to construct a
                response. If future has a :meth:`~concurrent.futures.Future.done` method then
                this will determine the value returned by :meth:`.Response.done`.

            result_hook (callable, optional):
                A function that is called to resolve the future. Must accept the future and return
                a :obj:`.Response`. If not provided then set to

                .. code-block:: python

                    def result_hook(future):
                        return future.result()

        Returns:
            :obj:`.Response`

        Notes:
            The future is resolved on the first read of any of the response's properties.

        Examples:
            Run a dimod sampler on a single thread and load the returned future into response.

            >>> import dimod
            >>> from concurrent.futures import ThreadPoolExecutor
            ...
            >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {('a', 'b'): -1})
            >>> with ThreadPoolExecutor(max_workers=1) as executor:
            ...     future = executor.submit(dimod.ExactSolver().sample, bqm)
            ...     response = dimod.Response.from_future(future)
            >>> response.record
            rec.array([([-1, -1], -1., 1), ([ 1, -1],  1., 1), ([ 1,  1], -1., 1),
                       ([-1,  1],  1., 1)],
                      dtype=[('sample', 'i1', (2,)), ('energy', '<f8'), ('num_occurrences', '<i8')])

        """
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

    ###############################################################################################
    # Special Methods
    ###############################################################################################

    def __len__(self):
        """The number of rows in record."""
        return self.record.__len__()

    def __iter__(self):
        """Iterate over the samples, low energy to high."""
        return self.samples(sorted_by='energy')

    def __eq__(self, other):
        """Response equality."""

        if not isinstance(other, SampleSet):
            return False

        if self.vartype != other.vartype or self.info != other.info:
            return False

        # check that all the fields match in record, order doesn't matter
        if self.record.dtype.fields != other.record.dtype.fields:
            return False
        for field in self.record.dtype.fields:
            if field == 'sample':
                continue
            if not (self.record[field] == other.record[field]).all():
                return False

        # now check the actual samples.
        if self.variables == other.variables:
            return (self.record.sample == other.record.sample).all()

        try:
            other_idx = [other.variables.index(v) for v in self.variables]
        except ValueError:
            # mismatched variables
            return False

        return (self.record.sample == other.record.sample[:, other_idx]).all()

    def __repr__(self):
        return "{}({!r}, {}, {}, {!r})".format(self.__class__.__name__,
                                               self.record,
                                               self.variables,
                                               self.info,
                                               self.vartype.name)

    ###############################################################################################
    # Properties
    ###############################################################################################

    @property
    def first(self):
        """Return the `Sample(sample={...}, energy, num_occurrences)` with
        lowest energy.
        """
        return next(self.data(sorted_by='energy', name='Sample'))

    @property
    def info(self):
        """dict: Information about the response as a whole formatted as a dict."""
        if hasattr(self, '_future'):
            self._resolve_future()
        return self._info

    @property
    def record(self):
        """:obj:`numpy.recarray` The samples, energies, number of occurences and other sample data.

        Examples:
            >>> import dimod
            ...
            >>> sampler = dimod.ExactSolver()
            >>> response = sampler.sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1.0})
            >>> response.record
            rec.array([([-1, -1], -1.5, 1), ([ 1, -1], -0.5, 1), ([ 1,  1], -0.5, 1),
                       ([-1,  1],  2.5, 1)],
                      dtype=[('sample', 'i1', (2,)), ('energy', '<f8'), ('num_occurrences', '<i8')])
            >>> response.record.sample
            array([[-1, -1],
                   [ 1, -1],
                   [ 1,  1],
                   [-1,  1]], dtype=int8)
            >>> response.record.energy
            array([-1.5, -0.5, -0.5,  2.5])
            >>> response.record.num_occurrences
            array([1, 1, 1, 1])

        """
        if hasattr(self, '_future'):
            self._resolve_future()
        return self._record

    @property
    def variables(self):
        """:obj:`.VariableIndexView`: Variable labels.

        Corresponds to the columns of the sample field of :attr:`.Response.record`.

        """
        if hasattr(self, '_future'):
            self._resolve_future()
        return self._variables

    @property
    def vartype(self):
        """:class:`.Vartype`: Vartype of the samples."""
        if hasattr(self, '_future'):
            self._resolve_future()
        return self._vartype

    ###############################################################################################
    # Views
    ###############################################################################################

    def done(self):
        """True if any pending computation is done.

        Only relevant when the response is constructed with :meth:`Response.from_future`.

        Examples:
            This example uses a :class:`~concurrent.futures.Future` object directly. Normally
            the future would have it's result set by an :class:`~concurrent.futures.Executor`
            (see documentation for :mod:`concurrent.futures`).

            >>> import dimod
            >>> from concurrent.futures import Future
            ...
            >>> future = Future()
            >>> response = dimod.Response.from_future(future)
            >>> future.done()
            False
            >>> future.set_result(dimod.ExactSolver().sample_ising({0: -1}, {}))
            >>> future.done()
            True
            >>> response.record.sample
            array([[-1],
                   [ 1]], dtype=int8)

        """
        return (not hasattr(self, '_future')) or (not hasattr(self._future, 'done')) or self._future.done()

    def samples(self, n=None, sorted_by='energy'):
        """Iterate over the samples in the response.

        Args:
            n (int, optional, default=None):
                The maximum number of samples to provide. If None, all are provided.

            sorted_by (str/None, optional, default='energy'):
                Selects the record field used to sort the samples. If None, the samples are yielded
                in record order.

        Yields:
            :obj:`.SampleView`: A view object mapping the variable labels to their values. Acts like
            a read-only dict.

        Examples:

            >>> import dimod
            ...
            >>> response = dimod.ExactSolver().sample_ising({'a': 0.0, 'b': 0.0}, {('a', 'b'): -1})
            >>> for sample in response.samples():   # doctest: +SKIP
            ...     print(sample)
            {'a': -1, 'b': -1}
            {'a': 1, 'b': 1}
            {'a': 1, 'b': -1}
            {'a': -1, 'b': 1}

        """
        if n is None:
            for sample, in self.data(['sample'], sorted_by=sorted_by, name=None):
                yield sample
        else:
            for sample in itertools.islice(self.samples(n=None, sorted_by=sorted_by), n):
                yield sample

    def data(self, fields=None, sorted_by='energy', name='Sample'):
        """Iterate over the data in the response.

        Args:
            fields (list, optional, default=None):
                If specified, only these fields' values are included in the yielded tuples.
                The special field name 'sample' can be used to view the samples.

            sorted_by (str/None, optional, default='energy'):
                Selects the record field used to sort the samples. If None, the samples are yielded
                in record order.

            name (str/None, optional, default='Sample'):
                Name of the yielded namedtuples or None to yield regular tuples.

        Yields:
            namedtuple/tuple: The data in the response, in the order specified by the input
            `fields`.

        Examples:

            >>> import dimod
            ...
            >>> response = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> for datum in response.data():   # doctest: +SKIP
            ...     print(datum)
            Sample(sample={'a': -1, 'b': -1}, energy=-1.5)
            Sample(sample={'a': 1, 'b': -1}, energy=-0.5)
            Sample(sample={'a': 1, 'b': 1}, energy=-0.5)
            Sample(sample={'a': -1, 'b': 1}, energy=2.5)
            >>> for energy, in response.data(fields=['energy'], sorted_by='energy'):
            ...     print(energy)
            ...
            -1.5
            -0.5
            -0.5
            2.5
            >>> print(next(response.data(fields=['energy'], name='ExactSolverSample')))
            ExactSolverSample(energy=-1.5)

        """
        record = self.record

        if fields is None:
            # make sure that sample, energy is first
            fields = self._REQUIRED_FIELDS + [field for field in record.dtype.fields
                                              if field not in self._REQUIRED_FIELDS]

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
                    yield SampleView(self.variables, record.sample[idx, :])
                else:
                    yield record[field][idx]

        for idx in order:
            yield _pack(_values(idx))

    ###############################################################################################
    # Methods
    ###############################################################################################

    def copy(self):
        """Create a shallow copy."""
        return self.__class__(self.record.copy(),
                              self.variables,  # a new one is made in all cases
                              self.info.copy(),
                              self.vartype)

    @vartype_argument('vartype')
    def change_vartype(self, vartype, energy_offset=0.0, inplace=True):
        """Create a new response with the given vartype.

        Args:
            vartype (:class:`.Vartype`/str/set):
                Variable type to use for the new response. Accepted input values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            energy_offset (number, optional, defaul=0.0):
                Constant value applied to the 'energy' field of :attr:`Response.record`.

            inplace (bool, optional, default=True):
                If True, the response is updated in-place, otherwise a new response is returned.

        Returns:
            :obj:`.Response`: Response with changed vartype. If inplace=True, returns itself.

        Examples:
            Create a binary copy of a spin-valued response

            >>> import dimod
            ...
            >>> response = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> response_binary = response.change_vartype(dimod.BINARY, energy_offset=1.0, inplace=False)
            >>> response_binary.vartype is dimod.BINARY
            True
            >>> for datum in response_binary.data():    # doctest: +SKIP
            ...    print(datum)
            Sample(sample={'a': 0, 'b': 0}, energy=-0.5, num_occurrences=1)
            Sample(sample={'a': 1, 'b': 0}, energy=0.5, num_occurrences=1)
            Sample(sample={'a': 1, 'b': 1}, energy=0.5, num_occurrences=1)
            Sample(sample={'a': 0, 'b': 1}, energy=3.5, num_occurrences=1)


        """
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
        """Relabel a response's variables as per a given mapping.

        Args:
            mapping (dict):
                Dict mapping current variable labels to new. If an incomplete mapping is
                provided, unmapped variables keep their original labels

            inplace (bool, optional, default=True):
                If True, the original response is updated; otherwise a new response is returned.

        Returns:
            :class:`.Response`: Response with relabeled variables. If inplace=True, returns
            itself.

        Examples:
            Create a relabeled copy of a response

            >>> import dimod
            ...
            >>> response = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> new_response = response.relabel_variables({'a': 0, 'b': 1}, inplace=False)
            >>> response.variable_labels    # doctest: +SKIP
            [0, 1]

        """
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

        self._variables = VariableIndexView(mapping.get(v, v) for v in self.variable_labels)
        return self


def as_samples(samples_like, dtype=None):

    if isinstance(samples_like, tuple) and len(samples_like) == 2:
        # (samples_like, labels)
        samples_like, labels = samples_like

        samples, __ = as_samples(samples_like)

        labels = list(labels)  # coerce and/or shallow copy

        if len(labels) != samples.shape[1]:
            raise ValueError("labels and samples_like dimensions do not match")

        return samples, labels

    if isinstance(samples_like, Iterator):
        raise TypeError('samples_like cannot be an iterator')

    if isinstance(samples_like, Iterable) and all(isinstance(sample, Mapping) for sample in samples_like):
        # list of dicts
        return _samples_dicts_to_array(samples_like)

    # anything else should be array_like, which covers ndarrays, lists of lists, etc

    # if no dtype is specified and the array_like doesn't already have a dtype, we default to int8
    if dtype is None and not hasattr(samples_like, 'dtype'):
        dtype = np.int8

    try:
        samples_like = np.asarray(samples_like, dtype=dtype)
    except (ValueError, TypeError):
        raise TypeError("unknown format for samples_like")

    # want 2D array
    if samples_like.ndim == 1:
        samples_like = np.expand_dims(samples_like, 0)
    elif samples_like.ndim > 2:
        ValueError("expected sample_like to be <= 2 dimensions")

    return samples_like, list(range(samples_like.shape[1]))


def _samples_dicts_to_array(samples_dicts):
    """Convert an iterable of samples where each sample is a dict to a numpy 2d array. Also
    determines the labels is they are None.
    """
    itersamples = iter(samples_dicts)

    first_sample = next(itersamples)

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
