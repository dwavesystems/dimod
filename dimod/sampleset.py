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
import itertools
import numbers

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from collections import namedtuple

import numpy as np

from numpy.lib import recfunctions

from dimod.decorators import vartype_argument
from dimod.serialization.format import sampleset_to_string
from dimod.variables import Variables
from dimod.vartypes import Vartype
from dimod.views import SampleView

__all__ = 'as_samples', 'concatenate', 'SampleSet'


def as_samples(samples_like, dtype=None, copy=False, order='C'):
    """Convert a samples_like object to a NumPy array and list of labels.

    Args:
        samples_like (samples_like):
            A collection of raw samples. `samples_like` is an extension of
            NumPy's array_like_ structure. See examples below.

        dtype (data-type, optional):
            dtype for the returned samples array. If not provided, it is either
            derived from `samples_like`, if that object has a dtype, or set to
            :class:`numpy.int8`.

        copy (bool, optional, default=False):
            If true, then samples_like is guaranteed to be copied, otherwise
            it is only copied if necessary.

        order ({'K', 'A', 'C', 'F'}, optional, default='C'):
            Specify the memory layout of the array. See :func:`numpy.array`.

    Returns:
        tuple: A 2-tuple containing:

            :obj:`numpy.ndarray`: Samples.

            list: Variable labels

    Examples:
        The following examples convert a variety of samples_like objects:

        NumPy arrays

        >>> dimod.as_samples(np.ones(5, dtype='int8'))
        (array([[1, 1, 1, 1, 1]], dtype=int8), [0, 1, 2, 3, 4])
        >>> dimod.as_samples(np.zeros((5, 2), dtype='int8'))
        (array([[0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0]], dtype=int8), [0, 1])

        Lists

        >>> dimod.as_samples([-1, +1, -1])
        (array([[-1,  1, -1]], dtype=int8), [0, 1, 2])
        >>> dimod.as_samples([[-1], [+1], [-1]])
        (array([[-1],
                [ 1],
                [-1]], dtype=int8), [0])

        Dicts

        >>> dimod.as_samples({'a': 0, 'b': 1, 'c': 0}) # doctest: +SKIP
        (array([[0, 1, 0]], dtype=int8), ['a', 'b', 'c'])
        >>> dimod.as_samples([{'a': -1, 'b': +1}, {'a': 1, 'b': 1}]) # doctest: +SKIP
        (array([[-1,  1],
                [ 1,  1]], dtype=int8), ['a', 'b'])

        A 2-tuple containing an array_like object and a list of labels

        >>> dimod.as_samples(([-1, +1, -1], ['a', 'b', 'c']))
        (array([[-1,  1, -1]], dtype=int8), ['a', 'b', 'c'])
        >>> dimod.as_samples((np.zeros((5, 2), dtype='int8'), ['in', 'out']))
        (array([[0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0]], dtype=int8), ['in', 'out'])

    .. _array_like: https://docs.scipy.org/doc/numpy/user/basics.creation.html

    """
    if isinstance(samples_like, tuple) and len(samples_like) == 2:
        samples_like, labels = samples_like

        if not isinstance(labels, list) and labels is not None:
            labels = list(labels)
    else:
        labels = None

    if isinstance(samples_like, abc.Iterator):
        # if we don't check this case we can get unexpected behaviour where an
        # iterator can be depleted
        raise TypeError('samples_like cannot be an iterator')

    if isinstance(samples_like, abc.Mapping):
        return as_samples(([samples_like], labels), dtype=dtype)

    if (isinstance(samples_like, list) and samples_like and
            isinstance(samples_like[0], numbers.Number)):
        # this is not actually necessary but it speeds up the
        # samples_like = [1, 0, 1,...] case significantly
        return as_samples(([samples_like], labels), dtype=dtype)

    if not isinstance(samples_like, np.ndarray):
        if any(isinstance(sample, abc.Mapping) for sample in samples_like):
            # go through samples-like, turning the dicts into lists
            samples_like, old = list(samples_like), samples_like

            if labels is None:
                first = samples_like[0]
                if isinstance(first, abc.Mapping):
                    labels = list(first)
                else:
                    labels = list(range(len(first)))

            for idx, sample in enumerate(old):
                if isinstance(sample, abc.Mapping):
                    try:
                        samples_like[idx] = [sample[v] for v in labels]
                    except KeyError:
                        raise ValueError("samples_like and labels do not match")

    if dtype is None and not hasattr(samples_like, 'dtype'):
        dtype = np.int8

    # samples-like should now be array-like
    arr = np.array(samples_like, dtype=dtype, copy=copy, order=order)

    if arr.ndim > 2:
        raise ValueError("expected samples_like to be <= 2 dimensions")
    if arr.ndim < 2:
        if arr.size:
            arr = np.atleast_2d(arr)
        elif labels:  # is not None and len > 0
            arr = arr.reshape((0, len(labels)))
        else:
            arr = arr.reshape((0, 0))

    # ok we're basically done, just need to check against the labels
    if labels is None:
        return arr, list(range(arr.shape[1]))
    elif len(labels) != arr.shape[1]:
        print(arr, arr.shape, samples_like, labels, len(labels))
        raise ValueError("samples_like and labels dimensions do not match")
    else:
        return arr, labels


def concatenate(samplesets, defaults=None):
    """Combine SampleSets.

    Args:
        samplesets (iterable[:obj:`.SampleSet`):
            An iterable of sample sets.

        defaults (dict, optional):
            Dictionary mapping data vector names to the corresponding default values.

    Returns:
        :obj:`.SampleSet`: A sample set with the same vartype and variable order as the first
        given in `samplesets`.

    Examples:
        >>> a = dimod.SampleSet.from_samples(([-1, +1], 'ab'), dimod.SPIN, energy=-1)
        >>> b = dimod.SampleSet.from_samples(([-1, +1], 'ba'), dimod.SPIN, energy=-1)
        >>> ab = dimod.concatenate((a, b))
        >>> ab.record.sample
        array([[-1,  1],
               [ 1, -1]], dtype=int8)

    """

    itertup = iter(samplesets)

    try:
        first = next(itertup)
    except StopIteration:
        raise ValueError("samplesets must contain at least one SampleSet")

    vartype = first.vartype
    variables = first.variables

    records = [first.record]
    records.extend(_iter_records(itertup, vartype, variables))

    # dev note: I was able to get ~2x performance boost when trying to
    # implement the same functionality here by hand (I didn't know that
    # this function existed then). However I think it is better to use
    # numpy's function and rely on their testing etc. If however this becomes
    # a performance bottleneck in the future, it might be worth changing.
    record = recfunctions.stack_arrays(records, defaults=defaults,
                                       asrecarray=True, usemask=False)

    return SampleSet(record, variables, {}, vartype)


def _iter_records(samplesets, vartype, variables):
    # coerce each record into the correct vartype and variable-order
    for samples in samplesets:

        # coerce vartype
        if samples.vartype is not vartype:
            samples = samples.change_vartype(vartype, inplace=False)

        if samples.variables != variables:
            new_record = samples.record.copy()
            order = [samples.variables.index[v] for v in variables]
            new_record.sample = samples.record.sample[:, order]
            yield new_record
        else:
            # order matches so we're done
            yield samples.record


class SampleSet(abc.Iterable, abc.Sized):
    """Samples and any other data returned by dimod samplers.

    Args:
        record (:obj:`numpy.recarray`)
            A NumPy record array. Must have 'sample', 'energy' and 'num_occurrences' as fields.
            The 'sample' field should be a 2D NumPy array where each row is a sample and each
            column represents the value of a variable.

        variables (iterable):
            An iterable of variable labels, corresponding to columns in `record.samples`.

        info (dict):
            Information about the :class:`SampleSet` as a whole, formatted as a dict.

        vartype (:class:`.Vartype`/str/set):
            Variable type for the :class:`SampleSet`. Accepted input values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

    Examples:
        This example creates a SampleSet out of a samples_like object (a NumPy array).

        >>> import dimod
        >>> import numpy as np
        ...
        >>> dimod.SampleSet.from_samples(np.ones(5, dtype='int8'), 'BINARY', 0)   # doctest: +SKIP
        SampleSet(rec.array([([1, 1, 1, 1, 1], 0, 1)],
        ...       dtype=[('sample', 'i1', (5,)), ('energy', '<i4'), ('num_occurrences', '<i4')]),
        ...       [0, 1, 2, 3, 4], {}, 'BINARY')

    """

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

        self._variables = variables = Variables(variables)
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
    def from_samples(cls, samples_like, vartype, energy, info=None,
                     num_occurrences=None, aggregate_samples=False, **vectors):
        """Build a :class:`SampleSet` from raw samples.

        Args:
            samples_like:
                A collection of raw samples. 'samples_like' is an extension of NumPy's array_like_.
                See :func:`.as_samples`.

            vartype (:class:`.Vartype`/str/set):
                Variable type for the :class:`SampleSet`. Accepted input values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            energy (array_like):
                Vector of energies.

            info (dict, optional):
                Information about the :class:`SampleSet` as a whole formatted as a dict.

            num_occurrences (array_like, optional):
                Number of occurrences for each sample. If not provided, defaults to a vector of 1s.

            aggregate_samples (bool, optional, default=False):
                If true, returned :obj:`.SampleSet` will have all unique samples.

            **vectors (array_like):
                Other per-sample data.

        Returns:
            :obj:`.SampleSet`

        Examples:
            This example creates a SampleSet out of a samples_like object (a dict).

            >>> import dimod
            >>> import numpy as np
            ...
            >>> dimod.SampleSet.from_samples(dimod.as_samples({'a': 0, 'b': 1, 'c': 0}),
            ...                              'BINARY', 0)   # doctest: +SKIP
            SampleSet(rec.array([([0, 1, 0], 0, 1)],
            ...       dtype=[('sample', 'i1', (3,)), ('energy', '<i4'), ('num_occurrences', '<i4')]),
            ...       ['a', 'b', 'c'], {}, 'BINARY')

        .. _array_like:  https://docs.scipy.org/doc/numpy/user/basics.creation.html#converting-python-array-like-objects-to-numpy-arrays
        """
        if aggregate_samples:
            return cls.from_samples(samples_like, vartype, energy,
                                    info=info, num_occurrences=num_occurrences,
                                    aggregate_samples=False,
                                    **vectors).aggregate()

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
        for key, vector in vectors.items():
            vectors[key] = vector = np.asarray(vector)
            datatypes.append((key, vector.dtype, vector.shape[1:]))

        record = np.rec.array(np.zeros(num_samples, dtype=datatypes))
        record['sample'] = samples
        record['energy'] = energy
        record['num_occurrences'] = num_occurrences
        for key, vector in vectors.items():
            record[key] = vector

        if info is None:
            info = {}

        return cls(record, variables, info, vartype)

    @classmethod
    def from_samples_bqm(cls, samples_like, bqm, **kwargs):
        """Build a SampleSet from raw samples using a BinaryQuadraticModel to get energies and vartype.

        Args:
            samples_like:
                A collection of raw samples. 'samples_like' is an extension of NumPy's array_like.
                See :func:`.as_samples`.

            bqm (:obj:`.BinaryQuadraticModel`):
                A binary quadratic model. It is used to calculate the energies
                and set the vartype.

            info (dict, optional):
                Information about the :class:`SampleSet` as a whole formatted as a dict.

            num_occurrences (array_like, optional):
                Number of occurrences for each sample. If not provided, defaults to a vector of 1s.

            aggregate_samples (bool, optional, default=False):
                If true, returned :obj:`.SampleSet` will have all unique samples.

            **vectors (array_like):
                Other per-sample data.

        Returns:
            :obj:`.SampleSet`

        Examples:

            >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {('a', 'b'): -1})
            >>> samples = dimod.SampleSet.from_samples_bqm({'a': -1, 'b': 1}, bqm)
            >>> samples =dimod.SampleSet.from_samples_bqm([[-1, 1], [1, -1]], bqm)

        """
        # more performant to do this once, here rather than again in bqm.energies
        # and in cls.from_samples
        samples_like = as_samples(samples_like)

        energies = bqm.energies(samples_like)

        return cls.from_samples(samples_like, energy=energies, vartype=bqm.vartype, **kwargs)

    @classmethod
    def from_future(cls, future, result_hook=None):
        """Construct a :class:`SampleSet` referencing the result of a future computation.

        Args:
            future (object):
                Object that contains or will contain the information needed to construct a
                :class:`SampleSet`. If `future` has a :meth:`~concurrent.futures.Future.done` method,
                this determines the value returned by :meth:`.SampleSet.done`.

            result_hook (callable, optional):
                A function that is called to resolve the future. Must accept the future and return
                a :obj:`.SampleSet`. If not provided, set to

                .. code-block:: python

                    def result_hook(future):
                        return future.result()

        Returns:
            :obj:`.SampleSet`

        Notes:
            The future is resolved on the first read of any of the :class:`SampleSet` properties.

        Examples:
            Run a dimod sampler on a single thread and load the returned future into :class:`SampleSet`.

            >>> import dimod
            >>> from concurrent.futures import ThreadPoolExecutor
            ...
            >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {('a', 'b'): -1})
            >>> with ThreadPoolExecutor(max_workers=1) as executor:
            ...     future = executor.submit(dimod.ExactSolver().sample, bqm)
            ...     sampleset = dimod.SampleSet.from_future(future)
            >>> sampleset.record
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
        samples = self._result_hook(self._future)
        self.__init__(samples.record, samples.variables, samples.info, samples.vartype)
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
        """SampleSet equality."""

        if not isinstance(other, SampleSet):
            return False

        if self.vartype != other.vartype or self.info != other.info:
            return False

        # check that all the fields match in record, order doesn't matter
        if self.record.dtype.fields.keys() != other.record.dtype.fields.keys():
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

    def __str__(self):
        return sampleset_to_string(self)   # use default parameters

    ###############################################################################################
    # Properties
    ###############################################################################################

    @property
    def first(self):
        """Sample with the lowest-energy.

        Raises:
            ValueError: If empty.

        Example:

            >>> sampleset = dimod.ExactSolver().sample_ising({'a': 1}, {('a', 'b'): 1})
            >>> sampleset.first
            Sample(sample={'a': -1, 'b': -1}, energy=-1.0, num_occurrences=1)

        """
        try:
            return next(self.data(sorted_by='energy', name='Sample'))
        except StopIteration:
            raise ValueError('{} is empty'.format(self.__class__.__name__))

    @property
    def info(self):
        """Dict of information about the :class:`SampleSet` as a whole."""
        if hasattr(self, '_future'):
            self._resolve_future()
        return self._info

    @property
    def record(self):
        """:obj:`numpy.recarray` containing the samples, energies, number of occurences, and other sample data.

        Examples:
            >>> import dimod
            ...
            >>> sampler = dimod.ExactSolver()
            >>> sampleset = sampler.sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1.0})
            >>> sampleset.record
            rec.array([([-1, -1], -1.5, 1), ([ 1, -1], -0.5, 1), ([ 1,  1], -0.5, 1),
                       ([-1,  1],  2.5, 1)],
                      dtype=[('sample', 'i1', (2,)), ('energy', '<f8'), ('num_occurrences', '<i8')])
            >>> sampleset.record.sample
            array([[-1, -1],
                   [ 1, -1],
                   [ 1,  1],
                   [-1,  1]], dtype=int8)
            >>> sampleset.record.energy
            array([-1.5, -0.5, -0.5,  2.5])

        """
        if hasattr(self, '_future'):
            self._resolve_future()
        return self._record

    @property
    def variables(self):
        """:obj:`.VariableIndexView` of variable labels.

        Corresponds to columns of the sample field of :attr:`.SampleSet.record`.

        """
        if hasattr(self, '_future'):
            self._resolve_future()
        return self._variables

    @property
    def vartype(self):
        """:class:`.Vartype` of the samples."""
        if hasattr(self, '_future'):
            self._resolve_future()
        return self._vartype

    ###############################################################################################
    # Views
    ###############################################################################################

    def done(self):
        """Return True if a pending computation is done.

        Used when a :class:`SampleSet` is constructed with :meth:`SampleSet.from_future`.

        Examples:
            This example uses a :class:`~concurrent.futures.Future` object directly. Typically
            a :class:`~concurrent.futures.Executor` sets the result of the future
            (see documentation for :mod:`concurrent.futures`).

            >>> import dimod
            >>> from concurrent.futures import Future
            ...
            >>> future = Future()
            >>> sampleset = dimod.SampleSet.from_future(future)
            >>> future.done()
            False
            >>> future.set_result(dimod.ExactSolver().sample_ising({0: -1}, {}))
            >>> future.done()
            True
            >>> sampleset.record.sample
            array([[-1],
                   [ 1]], dtype=int8)

        """
        return (not hasattr(self, '_future')) or (not hasattr(self._future, 'done')) or self._future.done()

    def samples(self, n=None, sorted_by='energy'):
        """Iterate over the samples in the :class:`SampleSet`.

        Args:
            n (int, optional, default=None):
                Maximum number of samples to yield. If None, all are yielded.

            sorted_by (str/None, optional, default='energy'):
                Selects the record field used to sort the samples. If None, samples are yielded
                in record order.

        Yields:
            :obj:`.SampleView`: A view object mapping variable labels to values. Acts as
            a read-only dict.

        Examples:

            >>> import dimod
            ...
            >>> sampleset = dimod.ExactSolver().sample_ising({'a': 0.0, 'b': 0.0}, {('a', 'b'): -1})
            >>> for sample in sampleset.samples():   # doctest: +SKIP
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

    def data(self, fields=None, sorted_by='energy', name='Sample', reverse=False,
             sample_dict_cast=True):
        """Iterate over the data in the :class:`SampleSet`.

        Args:
            fields (list, optional, default=None):
                If specified, only these fields are included in the yielded tuples.
                The special field name 'sample' can be used to view the samples.

            sorted_by (str/None, optional, default='energy'):
                Selects the record field used to sort the samples. If None, the samples are yielded
                in record order.

            name (str/None, optional, default='Sample'):
                Name of the yielded namedtuples or None to yield regular tuples.

            reverse (bool, optional, default=False):
                If True, yield in reverse order.

            sample_dict_cast (bool, optional, default=False):
                If True, samples are returned as dicts rather than
                `.SampleView`s. Note that this can lead to very heavy memory
                usage.

        Yields:
            namedtuple/tuple: The data in the :class:`SampleSet`, in the order specified by the input
            `fields`.

        Examples:

            >>> import dimod
            ...
            >>> sampleset = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> for datum in sampleset.data(fields=['sample', 'energy']):   # doctest: +SKIP
            ...     print(datum)
            Sample(sample={'a': -1, 'b': -1}, energy=-1.5)
            Sample(sample={'a': 1, 'b': -1}, energy=-0.5)
            Sample(sample={'a': 1, 'b': 1}, energy=-0.5)
            Sample(sample={'a': -1, 'b': 1}, energy=2.5)
            >>> for energy, in sampleset.data(fields=['energy'], sorted_by='energy'):
            ...     print(energy)
            ...
            -1.5
            -0.5
            -0.5
            2.5
            >>> print(next(sampleset.data(fields=['energy'], name='ExactSolverSample')))
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

        if reverse:
            order = np.flip(order)

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
                    sample = SampleView(self.variables, record.sample[idx, :])
                    if sample_dict_cast:
                        sample = dict(sample)
                    yield sample
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
        """Return the :class:`SampleSet` with the given vartype.

        Args:
            vartype (:class:`.Vartype`/str/set):
                Variable type to use for the new :class:`SampleSet`. Accepted input values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            energy_offset (number, optional, defaul=0.0):
                Constant value applied to the 'energy' field of :attr:`SampleSet.record`.

            inplace (bool, optional, default=True):
                If True, the instantiated :class:`SampleSet` is updated; otherwise, a new
                :class:`SampleSet` is returned.

        Returns:
            :obj:`.SampleSet`: SampleSet with changed vartype. If `inplace` is True, returns itself.

        Examples:
            This example creates a binary copy of a spin-valued :class:`SampleSet`.

            >>> import dimod
            ...
            >>> sampleset = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> sampleset_binary = sampleset.change_vartype(dimod.BINARY, energy_offset=1.0, inplace=False)
            >>> sampleset_binary.vartype is dimod.BINARY
            True
            >>> for datum in sampleset_binary.data(fields=['sample', 'energy', 'num_occurrences']):    # doctest: +SKIP
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
        """Relabel the variables of a :class:`SampleSet` according to the specified mapping.

        Args:
            mapping (dict):
                Mapping from current variable labels to new, as a dict. If incomplete mapping is
                specified, unmapped variables keep their current labels.

            inplace (bool, optional, default=True):
                If True, the current :class:`SampleSet` is updated; otherwise, a new
                :class:`SampleSet` is returned.

        Returns:
            :class:`.SampleSet`: SampleSet with relabeled variables. If `inplace` is True, returns
            itself.

        Examples:
            This example creates a relabeled copy of a :class:`SampleSet`.

            >>> import dimod
            ...
            >>> sampleset = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> new_sampleset = sampleset.relabel_variables({'a': 0, 'b': 1}, inplace=False)
            >>> sampleset.variable_labels    # doctest: +SKIP
            [0, 1]

        """
        if not inplace:
            return self.copy().relabel_variables(mapping, inplace=True)

        self._variables.relabel(mapping)
        return self

    def aggregate(self):
        """Create a new SampleSet with repeated samples aggregated.

        Returns:
            :obj:`.SampleSet`

        Note:
            :attr:`.SampleSet.record.num_occurrences` are accumulated but no
            other fields are.

        """

        _, indices, inverse = np.unique(self.record.sample, axis=0,
                                        return_index=True, return_inverse=True)

        record = self.record[indices]

        # fix the number of occurrences
        record.num_occurrences = 0
        for old_idx, new_idx in enumerate(inverse):
            record[new_idx].num_occurrences += self.record[old_idx].num_occurrences

        # dev note: we don't check the energies as they should be the same
        # for individual samples

        return type(self)(record, self.variables, self.info, self.vartype)

    def truncate(self, n, sorted_by='energy'):
        """Create a new SampleSet with rows truncated after n.

        Args:
            n (int):
                Maximum number of rows in the returned sample set.

            sorted_by (str/None, optional, default='energy'):
                Selects the record field used to sort the samples before
                truncating. Note that sample order is maintained in the
                underlying array.

        Returns:
            :obj:`.SampleSet`

        Examples:
            >>> sampleset = dimod.SampleSet.from_samples(np.ones((5, 5)), dimod.SPIN, energy=5)
            >>> print(sampleset)
                0   1   2   3   4  energy  num_occ.
            0  +1  +1  +1  +1  +1       5         1
            1  +1  +1  +1  +1  +1       5         1
            2  +1  +1  +1  +1  +1       5         1
            3  +1  +1  +1  +1  +1       5         1
            4  +1  +1  +1  +1  +1       5         1

            [ 5 rows, 5 variables ]
            >>> print(sampleset.truncate(2))
                0   1   2   3   4  energy  num_occ.
            0  +1  +1  +1  +1  +1       5         1
            1  +1  +1  +1  +1  +1       5         1

            [ 2 rows, 5 variables ]

        """
        record = self.record

        if sorted_by is None:
            record = record[:n]
        else:
            sort_indices = np.argsort(record[sorted_by])
            record = record[sort_indices[:n]]

        return type(self)(record, self.variables, self.info, self.vartype)

    ###############################################################################################
    # Serialization
    ###############################################################################################

    def to_serializable(self):
        """Convert a :class:`SampleSet` to a serializable object.

        Returns:
            dict: Object that can be serialized.

        Examples:
            This example encodes using JSON.

            >>> import dimod
            >>> import json
            ...
            >>> samples = dimod.SampleSet.from_samples([-1, 1, -1], dimod.SPIN, energy=-.5)
            >>> s = json.dumps(samples.to_serializable())

        See also:
            :meth:`~.SampleSet.from_serializable`

        """
        from dimod.serialization.json import DimodEncoder
        return DimodEncoder().default(self)

    @classmethod
    def from_serializable(cls, obj):
        """Deserialize a :class:`SampleSet`.

        Args:
            obj (dict):
                A :class:`SampleSet` serialized by :meth:`~.SampleSet.to_serializable`.

        Returns:
            :obj:`.SampleSet`

        Examples:
            This example encodes and decodes using JSON.

            >>> import dimod
            >>> import json
            ...
            >>> samples = dimod.SampleSet.from_samples([-1, 1, -1], dimod.SPIN, energy=-.5)
            >>> s = json.dumps(samples.to_serializable())
            >>> new_samples = dimod.SampleSet.from_serializable(json.loads(s))

        See also:
            :meth:`~.SampleSet.to_serializable`

        """
        from dimod.serialization.json import sampleset_decode_hook
        return sampleset_decode_hook(obj, cls=cls)

    ###############################################################################################
    # Export to dataframe
    ###############################################################################################

    def to_pandas_dataframe(self, sample_column=False):
        """Convert a SampleSet to a Pandas DataFrame

        Returns:
            :obj:`pandas.DataFrame`

        Examples:
            >>> samples = dimod.SampleSet.from_samples([{'a': -1, 'b': +1, 'c': -1},
                                                        {'a': -1, 'b': -1, 'c': +1}],
                                                       dimod.SPIN, energy=-.5)
            >>> samples.to_pandas_dataframe()
               a  b  c  energy  num_occurrences
            0 -1  1 -1    -0.5                1
            1 -1 -1  1    -0.5                1
            >>> samples.to_pandas_dataframe(sample_column=True)
                                   sample  energy  num_occurrences
            0  {'a': -1, 'b': 1, 'c': -1}    -0.5                1
            1  {'a': -1, 'b': -1, 'c': 1}    -0.5                1

        """
        import pandas as pd

        if sample_column:
            df = pd.DataFrame(self.data(sorted_by=None, sample_dict_cast=True))

        else:
            # work directly with the record, it's much faster
            df = pd.DataFrame(self.record.sample, columns=self.variables)

            for field in sorted(self.record.dtype.fields):  # sort for consistency
                if field == 'sample':
                    continue

                df.loc[:, field] = self.record[field]

        return df
