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
# =============================================================================
import base64
import copy
import itertools
import json
import numbers
import collections.abc as abc

from collections import namedtuple

import numpy as np
from numpy.lib import recfunctions
from warnings import warn

from dimod.exceptions import WriteableError
from dimod.serialization.format import Formatter
from dimod.serialization.utils import (pack_samples as _pack_samples,
                                       unpack_samples,
                                       serialize_ndarray,
                                       deserialize_ndarray,
                                       serialize_ndarrays,
                                       deserialize_ndarrays)
from dimod.utilities import LockableDict
from dimod.variables import Variables, iter_deserialize_variables
from dimod.vartypes import as_vartype, Vartype, DISCRETE
from dimod.views.samples import SampleView, SamplesArray

__all__ = ['append_data_vectors', 'append_variables', 'as_samples', 'concatenate', 'SampleSet']

def append_data_vectors(sampleset, **vectors):
    """Create a new :obj:`.SampleSet` with additional fields in
    :attr:`SampleSet.record`.

    Args:
        sampleset (:obj:`.SampleSet`):
            :obj:`.SampleSet` to build from.

        **vectors (list):
            Per-sample data to be appended to :attr:`SampleSet.record`. Each
            keyword is a new field name and each keyword parameter should be a
            list of scalar values or numpy arrays (lists and tuples will be
            converted to numpy arrays).

    Returns:
        :obj:`.SampleSet`: SampleSet

    Examples:
        The following example appends a field of lists to :attr:`SampleSet.record`.

        >>> sampleset = dimod.SampleSet.from_samples([[-1,  1], [-1,  1]], energy=[-1.4, -1.4], vartype='SPIN')
        >>> print(sampleset)
           0  1 energy num_oc.
        0 -1 +1   -1.4       1
        1 -1 +1   -1.4       1
        ['SPIN', 2 rows, 2 samples, 2 variables]

        >>> sampleset = dimod.append_data_vectors(sampleset, new=[[0, 1], [1, 2]])
        >>> print(sampleset)
           0  1 energy num_oc.   new
        0 -1 +1   -1.4       1 [0 1]
        1 -1 +1   -1.4       1 [1 2]
        ['SPIN', 2 rows, 2 samples, 2 variables]

        >>> print(sampleset.record.dtype)
        (numpy.record, [('sample', 'i1', (2,)), ('energy', '<f8'), ('num_occurrences', '<i8'), ('new', '<i8', (2,))])
    
    """
    record = sampleset.record

    for name, vector in vectors.items():
        if len(vector) != len(record.energy):
            raise ValueError("Length of vector {} must be equal to number of samples.".format(name))

        try:
            vector = np.asarray(vector)

            if vector.ndim == 1:
                record = recfunctions.append_fields(record, name, vector, usemask=False, asrecarray=True)
            else:
                # np's append_fields cannot append a vector with a shape that
                # doesn't match the base array's, so appending non-scalar data
                # requires a workaround
                dtype = np.dtype([(name, vector[0].dtype, vector[0].shape)]) 
                new_arr = recfunctions.unstructured_to_structured(vector, dtype=dtype)
                record = recfunctions.merge_arrays((record, new_arr), flatten=True, asrecarray=True)              
        
        except (TypeError, AttributeError):
            raise ValueError("Field value type not supported.")

    return SampleSet(record, sampleset.variables, sampleset.info, sampleset.vartype)

def append_variables(sampleset, samples_like, sort_labels=True):
    """Create a new :obj:`.SampleSet` with the given variables and values.

    Not defined for empty sample sets. If `sample_like` is a
    :obj:`.SampleSet`, its data vectors and info are ignored.

    Args:
        sampleset (:obj:`.SampleSet`):
            :obj:`.SampleSet` to build from.

        samples_like:
            Samples to add to the sample set. Either a single
            sample or identical in length to the sample set.
            'samples_like' is an extension of NumPy's array_like_.
            See :func:`.as_samples`.

        sort_labels (bool, optional, default=True):
            Return :attr:`.SampleSet.variables` in sorted order. For mixed
            (unsortable) types, the given order is maintained.

    Returns:
        :obj:`.SampleSet`: New sample set with the variables/values added.

    Examples:

        >>> sampleset = dimod.SampleSet.from_samples([{'a': -1, 'b': +1},
        ...                                           {'a': +1, 'b': +1}],
        ...                                          dimod.SPIN,
        ...                                          energy=[-1.0, 1.0])
        >>> new = dimod.append_variables(sampleset, {'c': -1})
        >>> print(new)
           a  b  c energy num_oc.
        0 -1 +1 -1   -1.0       1
        1 +1 +1 -1    1.0       1
        ['SPIN', 2 rows, 2 samples, 3 variables]

        Add variables from another sample set to the previous example. Note
        that the energies remain unchanged.

        >>> another = dimod.SampleSet.from_samples([{'c': -1, 'd': +1},
        ...                                         {'c': +1, 'd': +1}],
        ...                                        dimod.SPIN,
        ...                                        energy=[-2.0, 1.0])
        >>> new = dimod.append_variables(sampleset, another)
        >>> print(new)
           a  b  c  d energy num_oc.
        0 -1 +1 -1 +1   -1.0       1
        1 +1 +1 +1 +1    1.0       1
        ['SPIN', 2 rows, 2 samples, 4 variables]

    .. _array_like: https://docs.scipy.org/doc/numpy/user/basics.creation.html

    """
    samples, labels = as_samples(samples_like)

    num_samples = len(sampleset)

    # we don't handle multiple values
    if samples.shape[0] == num_samples:
        # we don't need to do anything, it's already the correct shape
        pass
    elif samples.shape[0] == 1 and num_samples:
        samples = np.repeat(samples, num_samples, axis=0)
    else:
        msg = ("mismatched shape. The samples to append should either be "
                "a single sample or should match the length of the sample "
                "set. Empty sample sets cannot be appended to.")
        raise ValueError(msg)

    # append requires the new variables to be unique
    variables = sampleset.variables
    if any(v in variables for v in labels):
        msg = "Appended samples cannot contain variables in sample set"
        raise ValueError(msg)

    new_variables = list(variables) + labels
    new_samples = np.hstack((sampleset.record.sample, samples))

    return type(sampleset).from_samples((new_samples, new_variables),
                                        sampleset.vartype,
                                        info=copy.deepcopy(sampleset.info),  # make a copy
                                        sort_labels=sort_labels,
                                        **sampleset.data_vectors)

def as_samples(samples_like, dtype=None, copy=False, order='C'):
    """Convert a samples_like object to a NumPy array and list of labels.

    Args:
        samples_like (samples_like):
            A collection of raw samples. `samples_like` is an extension of
            NumPy's array_like_ structure. See examples below.

        dtype (data-type, optional):
            dtype for the returned samples array. If not provided, it is either
            derived from `samples_like`, if that object has a dtype, or set to
            the smallest dtype that can hold the given values.

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

        >>> import numpy as np
        ...
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
    if isinstance(samples_like, SampleSet):
        # we implicitely support this by handling an iterable of mapping but
        # it is much faster to just do this here.
        labels = list(samples_like.variables)
        if dtype is None:
            return samples_like.record.sample, labels
        else:
            return samples_like.record.sample.astype(dtype), labels

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

    if dtype is None:
        if not hasattr(samples_like, 'dtype'):
            # we want to use the smallest dtype available, not yet doing any
            # copying or whatever, although we do make a new array to speed
            # this up
            samples_like = np.asarray(samples_like)

            max_ = max(-samples_like.min(initial=0),
                       +samples_like.max(initial=0))

            if max_ <= np.iinfo(np.int8).max:
                dtype = np.int8
            elif max_ <= np.iinfo(np.int16).max:
                dtype = np.int16
            elif max_ < np.iinfo(np.int32).max:
                dtype = np.int32
            elif max_ < np.iinfo(np.int64).max:
                dtype = np.int64
            else:
                raise RuntimeError
        else:
            dtype = samples_like.dtype

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
        raise ValueError("samples_like and labels dimensions do not match")
    else:
        return arr, labels


def concatenate(samplesets, defaults=None):
    """Combine sample sets.

    Args:
        samplesets (iterable[:obj:`.SampleSet`):
            Iterable of sample sets.

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


def infer_vartype(samples_like):
    """Infer the vartype of the given samples-like.

    Args:
        A collection of samples. 'samples_like' is an extension of NumPy's
        array_like_. See :func:`.as_samples`.

    Returns:
        The :class:`.Vartype`, or None in the case that it is ambiguous.

    """
    if isinstance(samples_like, SampleSet):
        return samples_like.vartype

    samples, _ = as_samples(samples_like)

    ones_mask = (samples == 1)

    if ones_mask.all():
        # either empty or all 1s, in either case ambiguous
        return None

    if (ones_mask ^ (samples == 0)).all():
        return Vartype.BINARY

    if (ones_mask ^ (samples == -1)).all():
        return Vartype.SPIN

    raise ValueError("given samples_like is of an unknown vartype")


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
            * :class:`.ExtendedVartype.DISCRETE`, ``'DISCRETE'``

    Examples:
        This example creates a SampleSet out of a samples_like object (a NumPy array).

        >>> import numpy as np
        ...
        >>> sampleset =  dimod.SampleSet.from_samples(np.ones(5, dtype='int8'),
        ...                                           'BINARY', 0)
        >>> sampleset.variables
        Variables([0, 1, 2, 3, 4])

    """

    _REQUIRED_FIELDS = ['sample', 'energy', 'num_occurrences']

    ###############################################################################################
    # Construction
    ###############################################################################################

    def __init__(self, record, variables, info, vartype):

        vartype = as_vartype(vartype, extended=True)

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

        self._info = LockableDict(info)

        # vartype is checked by vartype_argument decorator
        self._vartype = vartype

    @classmethod
    def from_samples(cls, samples_like, vartype, energy, info=None,
                     num_occurrences=None, aggregate_samples=False,
                     sort_labels=True, **vectors):
        """Build a :class:`SampleSet` from raw samples.

        Args:
            samples_like:
                A collection of raw samples. 'samples_like' is an extension of NumPy's array_like_.
                See :func:`.as_samples`.

            vartype (:class:`.Vartype`/str/set):
                Variable type for the :class:`SampleSet`. Accepted input values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``
                * :class:`.ExtendedVartype.DISCRETE`, ``'DISCRETE'``

            energy (array_like):
                Vector of energies.

            info (dict, optional):
                Information about the :class:`SampleSet` as a whole formatted as a dict.

            num_occurrences (array_like, optional):
                Number of occurrences for each sample. If not provided, defaults to a vector of 1s.

            aggregate_samples (bool, optional, default=False):
                If True, all samples in returned :obj:`.SampleSet` are unique,
                with `num_occurrences` accounting for any duplicate samples in
                `samples_like`.

            sort_labels (bool, optional, default=True):
                Return :attr:`.SampleSet.variables` in sorted order. For mixed
                (unsortable) types, the given order is maintained.

            **vectors (array_like):
                Other per-sample data.

        Returns:
            :obj:`.SampleSet`

        Examples:
            This example creates a SampleSet out of a samples_like object (a dict).

            >>> import numpy as np
            ...
            >>> sampleset = dimod.SampleSet.from_samples(
            ...   dimod.as_samples({'a': 0, 'b': 1, 'c': 0}), 'BINARY', 0)
            >>> sampleset.variables
            Variables(['a', 'b', 'c'])

        .. _array_like:  https://docs.scipy.org/doc/numpy/user/basics.creation.html#converting-python-array-like-objects-to-numpy-arrays
        """
        if aggregate_samples:
            return cls.from_samples(samples_like, vartype, energy,
                                    info=info, num_occurrences=num_occurrences,
                                    aggregate_samples=False,
                                    **vectors).aggregate()

        # get the samples, variable labels
        samples, variables = as_samples(samples_like)

        if sort_labels and variables:  # need something to sort
            try:
                reindex, new_variables = zip(*sorted(enumerate(variables),
                                                     key=lambda tup: tup[1]))
            except TypeError:
                # unlike types are not sortable in python3, so we do nothing
                pass
            else:
                if new_variables != variables:
                    # avoid the copy if possible
                    samples = samples[:, reindex]
                    variables = new_variables

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

    # todo: this works with DQM/BinaryPolynomial, should change the name and/or
    # update the docs.
    @classmethod
    def from_samples_bqm(cls, samples_like, bqm, **kwargs):
        """Build a sample set from raw samples and a binary quadratic model.

        The binary quadratic model is used to calculate energies and set the
        :class:`vartype`.

        Args:
            samples_like:
                A collection of raw samples. 'samples_like' is an extension of NumPy's array_like.
                See :func:`.as_samples`.

            bqm (:obj:`.BinaryQuadraticModel`):
                A binary quadratic model.

            info (dict, optional):
                Information about the :class:`SampleSet` as a whole formatted as a dict.

            num_occurrences (array_like, optional):
                Number of occurrences for each sample. If not provided, defaults to a vector of 1s.

            aggregate_samples (bool, optional, default=False):
                If True, all samples in returned :obj:`.SampleSet` are unique,
                with `num_occurrences` accounting for any duplicate samples in
                `samples_like`.

            sort_labels (bool, optional, default=True):
                Return :attr:`.SampleSet.variables` in sorted order. For mixed
                (unsortable) types, the given order is maintained.

            **vectors (array_like):
                Other per-sample data.

        Returns:
            :obj:`.SampleSet`

        Examples:

            >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {('a', 'b'): -1})
            >>> sampleset = dimod.SampleSet.from_samples_bqm({'a': -1, 'b': 1}, bqm)

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

            >>> from concurrent.futures import ThreadPoolExecutor
            ...
            >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {('a', 'b'): -1})
            >>> with ThreadPoolExecutor(max_workers=1) as executor:
            ...     future = executor.submit(dimod.ExactSolver().sample, bqm)
            ...     sampleset = dimod.SampleSet.from_future(future)
            >>> sampleset.first.energy    # doctest: +SKIP

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

    ###############################################################################################
    # Special Methods
    ###############################################################################################

    def __len__(self):
        """The number of rows in record."""
        return self.record.__len__()

    def __iter__(self):
        """Iterate over the samples, low energy to high."""
        # need to make it an iterator rather than just an iterable
        return iter(self.samples(sorted_by='energy'))

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

    def __getstate__(self):
        # Ensure that any futures are resolved before pickling.
        self.resolve()
        # we'd prefer to do super().__getstate__ but unfortunately that's not
        # present, so instead we recreate the (documented) behaviour
        return self.__dict__

    def __repr__(self):
        return "{}({!r}, {}, {}, {!r})".format(self.__class__.__name__,
                                               self.record,
                                               self.variables,
                                               self.info,
                                               self.vartype.name)

    def __str__(self):
        return Formatter().format(self)

    ###############################################################################################
    # Properties
    ###############################################################################################

    @property
    def data_vectors(self):
        """The per-sample data in a vector.

        Returns:
            dict: A dict where the keys are the fields in the record and the
            values are the corresponding arrays.

        Examples:
            >>> sampleset = dimod.SampleSet.from_samples([[-1, 1], [1, 1]], dimod.SPIN,
                                                         energy=[-1, 1])
            >>> sampleset.data_vectors['energy']
            array([-1,  1])

            Note that this is equivalent to, and less performant than:

            >>> sampleset = dimod.SampleSet.from_samples([[-1, 1], [1, 1]], dimod.SPIN,
                                                         energy=[-1, 1])
            >>> sampleset.record['energy']
            array([-1,  1])


        """
        return {field: self.record[field] for field in self.record.dtype.names
                if field != 'sample'}

    @property
    def first(self):
        """Sample with the lowest-energy.

        Raises:
            ValueError: If empty.

        Example:

            >>> sampleset = dimod.ExactSolver().sample_ising({'a': 1}, {('a', 'b'): 1})
            >>> sampleset.first
            Sample(sample={'a': -1, 'b': 1}, energy=-2.0, num_occurrences=1)

        """
        try:
            return next(self.data(sorted_by='energy', name='Sample'))
        except StopIteration:
            raise ValueError('{} is empty'.format(self.__class__.__name__))

    @property
    def info(self):
        """Dict of information about the :class:`SampleSet` as a whole.

        Examples:
           This example shows the type of information that might be returned by
           a dimod sampler by submitting a BQM that sets a value on a D-Wave
           system's first listed coupler.

           >>> from dwave.system import DWaveSampler    # doctest: +SKIP
           >>> sampler = DWaveSampler()    # doctest: +SKIP
           >>> bqm = dimod.BQM({}, {sampler.edgelist[0]: -1}, 0, dimod.SPIN)   # doctest: +SKIP
           >>> sampler.sample(bqm).info   # doctest: +SKIP
           {'timing': {'qpu_sampling_time': 315,
            'qpu_anneal_time_per_sample': 20,
            'qpu_readout_time_per_sample': 274,
            # Snipped above response for brevity
        """
        self.resolve()
        return self._info

    @property
    def record(self):
        """:obj:`numpy.recarray` containing the samples, energies, number of occurences, and other sample data.

        Examples:
            >>> sampler = dimod.ExactSolver()
            >>> sampleset = sampler.sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1.0})
            >>> sampleset.record.sample     # doctest: +SKIP
            array([[-1, -1],
                   [ 1, -1],
                   [ 1,  1],
                   [-1,  1]], dtype=int8)
            >>> len(sampleset.record.energy)
            4

        """
        self.resolve()
        return self._record

    @property
    def variables(self):
        """:class:`.VariableIndexView` of variable labels.

        Corresponds to columns of the sample field of :attr:`.SampleSet.record`.

        """
        self.resolve()
        return self._variables

    @property
    def vartype(self):
        """:class:`.Vartype` of the samples."""
        self.resolve()
        return self._vartype

    @property
    def is_writeable(self):
        return getattr(self, '_writeable', True)

    @is_writeable.setter
    def is_writeable(self, b):
        b = bool(b)  # cast

        self._writeable = b

        self.record.flags.writeable = b
        self.variables.is_writeable = b
        self.info.is_writeable = b

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

            >>> from concurrent.futures import Future
            ...
            >>> future = Future()
            >>> sampleset = dimod.SampleSet.from_future(future)
            >>> future.done()
            False
            >>> future.set_result(dimod.ExactSolver().sample_ising({0: -1}, {}))
            >>> future.done()
            True
            >>> sampleset.first.energy
            -1.0

        """
        return (not hasattr(self, '_future')) or (not hasattr(self._future, 'done')) or self._future.done()

    def samples(self, n=None, sorted_by='energy'):
        """Return an iterable over the samples.

        Args:
            n (int, optional, default=None):
                Maximum number of samples to return in the view.

            sorted_by (str/None, optional, default='energy'):
                Selects the record field used to sort the samples. If None,
                samples are returned in record order.

        Returns:
            :obj:`.SamplesArray`: A view object mapping variable labels to
            values.

        Examples:

            >>> sampleset = dimod.ExactSolver().sample_ising({'a': 0.1, 'b': 0.0},
            ...                                              {('a', 'b'): 1})
            >>> for sample in sampleset.samples():   # doctest: +SKIP
            ...     print(sample)
            {'a': -1, 'b': 1}
            {'a': 1, 'b': -1}
            {'a': -1, 'b': -1}
            {'a': 1, 'b': 1}

            >>> sampleset = dimod.ExactSolver().sample_ising({'a': 0.1, 'b': 0.0},
            ...                                              {('a', 'b'): 1})
            >>> samples = sampleset.samples()
            >>> samples[0]
            {'a': -1, 'b': 1}
            >>> samples[0, 'a']
            -1
            >>> samples[0, ['b', 'a']]
            array([ 1, -1], dtype=int8)
            >>> samples[1:, ['a', 'b']]
            array([[ 1, -1],
                   [-1, -1],
                   [ 1,  1]], dtype=int8)

        """
        if n is not None:
            return self.samples(sorted_by=sorted_by)[:n]

        if sorted_by is None:
            samples = self.record.sample
        else:
            order = np.argsort(self.record[sorted_by])
            samples = self.record.sample[order]

        return SamplesArray(samples, self.variables)

    def data(self, fields=None, sorted_by='energy', name='Sample', reverse=False,
             sample_dict_cast=True, index=False):
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

            sample_dict_cast (bool, optional, default=True):
                Samples are returned as dicts rather than
                :class:`.SampleView`, which requires heavy memory
                usage. Set to False to reduce load on memory.

            index (bool, optional, default=False):
                If True, `datum.idx` gives the corresponding index of the
                :attr:`.SampleSet.record`.

        Yields:
            namedtuple/tuple: The data in the :class:`SampleSet`, in the order specified by the input
            `fields`.

        Examples:

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
            if index:
                fields.append('idx')

        if sorted_by is None:
            order = np.arange(len(self))
        elif index:
            # we want a stable sort but it can be slower
            order = np.argsort(record[sorted_by], kind='stable')
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
                    sample = SampleView(record.sample[idx, :], self.variables)
                    if sample_dict_cast:
                        sample = dict(sample)
                    yield sample
                elif field == 'idx':
                    yield idx
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

        Notes:
            This function is non-blocking unless `inplace==True`, in which case
            the sample set is resolved.

        Examples:
            This example creates a binary copy of a spin-valued :class:`SampleSet`.

            >>> sampleset = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> sampleset_binary = sampleset.change_vartype(dimod.BINARY, energy_offset=1.0, inplace=False)
            >>> sampleset_binary.vartype is dimod.BINARY
            True
            >>> sampleset_binary.first.sample
            {'a': 0, 'b': 0}

        """
        if not inplace:
            return self.copy().change_vartype(vartype, energy_offset, inplace=True)

        if not self.done():
            def hook(sampleset):
                sampleset.resolve()
                return sampleset.change_vartype(vartype, energy_offset)
            return self.from_future(self, hook)

        if not self.is_writeable:
            raise WriteableError("SampleSet is not writeable")

        vartype = as_vartype(vartype, extended=True)  # cast to correct vartype

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

        Notes:
            This function is non-blocking unless `inplace==True`, in which case
            the sample set is resolved.

        Examples:
            This example creates a relabeled copy of a :class:`SampleSet`.

            >>> sampleset = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> new_sampleset = sampleset.relabel_variables({'a': 0, 'b': 1}, inplace=False)
            >>> new_sampleset.variables
            Variables([0, 1])

        """
        if not inplace:
            return self.copy().relabel_variables(mapping, inplace=True)

        if not self.done():
            def hook(sampleset):
                sampleset.resolve()
                return sampleset.relabel_variables(mapping, inplace=True)
            return self.from_future(self, hook)

        self.variables.relabel(mapping)
        return self

    def resolve(self):
        """Ensure that the sampleset is resolved if constructed from a future.
        """
        # if it doesn't have the attribute then it is already resolved
        if hasattr(self, '_future'):
            samples = self._result_hook(self._future)
            self.__init__(samples.record, samples.variables, samples.info, samples.vartype)
            del self._future
            del self._result_hook

    def aggregate(self):
        """Create a new SampleSet with repeated samples aggregated.

        Returns:
            :obj:`.SampleSet`

        Note:
            :attr:`.SampleSet.record.num_occurrences` are accumulated but no
            other fields are.

        Examples:
            This examples aggregates a sample set with two identical samples
            out of three.

            >>> sampleset = dimod.SampleSet.from_samples([[0, 0, 1], [0, 0, 1],
            ...                                           [1, 1, 1]],
            ...                                           dimod.BINARY,
            ...                                           [0, 0, 1])
            >>> print(sampleset)
               0  1  2 energy num_oc.
            0  0  0  1      0       1
            1  0  0  1      0       1
            2  1  1  1      1       1
            ['BINARY', 3 rows, 3 samples, 3 variables]
            >>> print(sampleset.aggregate())
               0  1  2 energy num_oc.
            0  0  0  1      0       2
            1  1  1  1      1       1
            ['BINARY', 2 rows, 3 samples, 3 variables]
        """
        _, indices, inverse = np.unique(self.record.sample, axis=0,
                                        return_index=True, return_inverse=True)

        # unique also sorts the array which we don't want, so we undo the sort
        order = np.argsort(indices)
        indices = indices[order]

        # and on the inverse
        revorder = np.empty(len(order), dtype=order.dtype)
        revorder[order] = np.arange(len(order))
        inverse = revorder[inverse]

        record = self.record[indices]

        # fix the number of occurrences
        record.num_occurrences = 0
        for old_idx, new_idx in enumerate(inverse):
            record[new_idx].num_occurrences += self.record[old_idx].num_occurrences

        # dev note: we don't check the energies as they should be the same
        # for individual samples

        return type(self)(record, self.variables, copy.deepcopy(self.info),
                          self.vartype)

    def append_variables(self, samples_like, sort_labels=True):
        """Deprecated in favor of `dimod.append_variables`."""
        
        warn("SampleSet.append_variables is deprecated; please use "
             "`dimod.append_variables` instead.", DeprecationWarning)

        return append_variables(self, samples_like, sort_labels)

    def lowest(self, rtol=1.e-5, atol=1.e-8):
        """Return a sample set containing the lowest-energy samples.

        A sample is included if its energy is within tolerance of the lowest
        energy in the sample set. The following equation is used to determine
        if two values are equivalent:

        absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

        See :func:`numpy.isclose` for additional details and caveats.

        Args:
            rtol (float, optional, default=1.e-5):
                The relative tolerance (see above).

            atol (float, optional, default=1.e-8):
                The absolute tolerance (see above).

        Returns:
            :obj:`.SampleSet`: A new sample set containing the lowest energy
            samples as delimited by configured tolerances from the lowest energy
            sample in the current sample set.

        Examples:
            >>> sampleset = dimod.ExactSolver().sample_ising({'a': .001},
            ...                                              {('a', 'b'): -1})
            >>> print(sampleset.lowest())
               a  b energy num_oc.
            0 -1 -1 -1.001       1
            ['SPIN', 1 rows, 1 samples, 2 variables]
            >>> print(sampleset.lowest(atol=.1))
               a  b energy num_oc.
            0 -1 -1 -1.001       1
            1 +1 +1 -0.999       1
            ['SPIN', 2 rows, 2 samples, 2 variables]

        Note:
            "Lowest energy" is the lowest energy in the sample set. This is not
            always the "ground energy" which is the lowest energy possible
            for a binary quadratic model.

        """

        if len(self) == 0:
            # empty so all are lowest
            return self.copy()

        record = self.record

        # want all the rows within tolerance of the minimal energy
        close = np.isclose(record.energy,
                           np.min(record.energy),
                           rtol=rtol, atol=atol)
        record = record[close]

        return type(self)(record, self.variables, copy.deepcopy(self.info),
                          self.vartype)

    def truncate(self, n, sorted_by='energy'):
        """Create a new sample set with up to n rows.

        Args:
            n (int):
                Maximum number of rows in the returned sample set. Does not return
                any rows above this limit in the original sample set.

            sorted_by (str/None, optional, default='energy'):
                Selects the record field used to sort the samples before
                truncating. Note that this sort order is maintained in the
                returned sample set.

        Returns:
            :obj:`.SampleSet`

        Examples:

            >>> import numpy as np
            ...
            >>> sampleset = dimod.SampleSet.from_samples(np.ones((5, 5)), dimod.SPIN, energy=5)
            >>> print(sampleset)
               0  1  2  3  4 energy num_oc.
            0 +1 +1 +1 +1 +1      5       1
            1 +1 +1 +1 +1 +1      5       1
            2 +1 +1 +1 +1 +1      5       1
            3 +1 +1 +1 +1 +1      5       1
            4 +1 +1 +1 +1 +1      5       1
            ['SPIN', 5 rows, 5 samples, 5 variables]
            >>> print(sampleset.truncate(2))
               0  1  2  3  4 energy num_oc.
            0 +1 +1 +1 +1 +1      5       1
            1 +1 +1 +1 +1 +1      5       1
            ['SPIN', 2 rows, 2 samples, 5 variables]

        See:
            :meth:`SampleSet.slice`

        """
        return self.slice(n, sorted_by=sorted_by)

    def slice(self, *slice_args, **kwargs):
        """Create a new sample set with rows sliced according to standard Python
        slicing syntax.

        Args:
            start (int, optional, default=None):
                Start index for `slice`.

            stop (int):
                Stop index for `slice`.

            step (int, optional, default=None):
                Step value for `slice`.

            sorted_by (str/None, optional, default='energy'):
                Selects the record field used to sort the samples before
                slicing. Note that `sorted_by` determines the sample order in
                the returned sample set.

        Returns:
            :obj:`.SampleSet`

        Examples:

            >>> import numpy as np
            ...
            >>> sampleset = dimod.SampleSet.from_samples(np.diag(range(1, 11)),
            ...                   dimod.BINARY, energy=range(10))
            >>> print(sampleset)
               0  1  2  3  4  5  6  7  8  9 energy num_oc.
            0  1  0  0  0  0  0  0  0  0  0      0       1
            1  0  1  0  0  0  0  0  0  0  0      1       1
            2  0  0  1  0  0  0  0  0  0  0      2       1
            3  0  0  0  1  0  0  0  0  0  0      3       1
            4  0  0  0  0  1  0  0  0  0  0      4       1
            5  0  0  0  0  0  1  0  0  0  0      5       1
            6  0  0  0  0  0  0  1  0  0  0      6       1
            7  0  0  0  0  0  0  0  1  0  0      7       1
            8  0  0  0  0  0  0  0  0  1  0      8       1
            9  0  0  0  0  0  0  0  0  0  1      9       1
            ['BINARY', 10 rows, 10 samples, 10 variables]

            The above example's first 3 samples by energy == truncate(3):

            >>> print(sampleset.slice(3))
               0  1  2  3  4  5  6  7  8  9 energy num_oc.
            0  1  0  0  0  0  0  0  0  0  0      0       1
            1  0  1  0  0  0  0  0  0  0  0      1       1
            2  0  0  1  0  0  0  0  0  0  0      2       1
            ['BINARY', 3 rows, 3 samples, 10 variables]

            The last 3 samples by energy:

            >>> print(sampleset.slice(-3, None))
               0  1  2  3  4  5  6  7  8  9 energy num_oc.
            0  0  0  0  0  0  0  0  1  0  0      7       1
            1  0  0  0  0  0  0  0  0  1  0      8       1
            2  0  0  0  0  0  0  0  0  0  1      9       1
            ['BINARY', 3 rows, 3 samples, 10 variables]

            Every second sample in between, skipping top and bottom 3:

            >>> print(sampleset.slice(3, -3, 2))
               0  1  2  3  4  5  6  7  8  9 energy num_oc.
            0  0  0  0  1  0  0  0  0  0  0      3       1
            1  0  0  0  0  0  1  0  0  0  0      5       1
            ['BINARY', 2 rows, 2 samples, 10 variables]

        """
        # handle `sorted_by` kwarg with a default value in a python2-compatible way
        sorted_by = kwargs.pop('sorted_by', 'energy')
        if kwargs:
            # be strict about allowed kwargs: throw the same error as python3 would
            raise TypeError('slice got an unexpected '
                            'keyword argument {!r}'.format(kwargs.popitem()[0]))

        # follow Python's slice syntax
        if slice_args:
            selector = slice(*slice_args)
        else:
            selector = slice(None)

        if sorted_by is None:
            record = self.record[selector]
        else:
            sort_indices = np.argsort(self.record[sorted_by])
            record = self.record[sort_indices[selector]]

        return type(self)(record, self.variables, copy.deepcopy(self.info),
                          self.vartype)


    ###############################################################################################
    # Serialization
    ###############################################################################################

    def to_serializable(self, use_bytes=False, bytes_type=bytes,
                        pack_samples=True):
        """Convert a :class:`SampleSet` to a serializable object.

        Note that the contents of the :attr:`.SampleSet.info` field are assumed
        to be serializable.

        Args:
            use_bytes (bool, optional, default=False):
                If True, a compact representation of the biases as bytes is used.

            bytes_type (class, optional, default=bytes):
                If `use_bytes` is True, this class is used to wrap the bytes
                objects in the serialization. Useful for Python 2 using BSON
                encoding, which does not accept the raw `bytes` type;
                `bson.Binary` can be used instead.

            pack_samples (bool, optional, default=True):
                Pack the samples using 1 bit per sample. Samples are never
                packed when :attr:`SampleSet.vartype` is
                `~ExtendedVartype.DISCRETE`.

        Returns:
            dict: Object that can be serialized.

        Examples:
            This example encodes using JSON.

            >>> import json
            ...
            >>> samples = dimod.SampleSet.from_samples([-1, 1, -1], dimod.SPIN, energy=-.5)
            >>> s = json.dumps(samples.to_serializable())

        See also:
            :meth:`~.SampleSet.from_serializable`

        """
        schema_version = "3.1.0"

        # developer note: numpy's record array stores the samples, energies,
        # num_occ. etc as a struct array. If we dumped that array directly to
        # bytes we could avoid a copy when undoing the serialization. However,
        # we want to pack the samples, so that means that we're storing the
        # arrays individually.
        vectors = {name: serialize_ndarray(data, use_bytes=use_bytes,
                                           bytes_type=bytes_type)
                   for name, data in self.data_vectors.items()}

        # we never pack DISCRETE samplesets
        pack_samples = pack_samples and self.vartype is not DISCRETE

        if pack_samples:
            # we could just do self.record.sample > 0 for all of these, but to
            # save on the copy if we are already binary and bool/integer we
            # check and just pass through in that case
            samples = self.record.sample
            if (self.vartype is Vartype.BINARY and
                    (np.issubdtype(samples.dtype, np.integer) or
                     np.issubdtype(samples.dtype, np.bool_))):
                packed = _pack_samples(samples)
            else:
                packed = _pack_samples(samples > 0)

            sample_data = serialize_ndarray(packed,
                                            use_bytes=use_bytes,
                                            bytes_type=bytes_type)
        else:
            sample_data = serialize_ndarray(self.record.sample,
                                            use_bytes=use_bytes,
                                            bytes_type=bytes_type)

        return {
            # metadata
            "type": type(self).__name__,
            "version": {"sampleset_schema": schema_version},

            # samples
            "num_variables": len(self.variables),
            "num_rows": len(self),
            "sample_data": sample_data,
            "sample_type": self.record.sample.dtype.name,
            "sample_packed": bool(pack_samples),  # 3.1.0+, default=True

            # vectors
            "vectors": vectors,

            # other
            "variable_labels": self.variables.to_serializable(),
            "variable_type": self.vartype.name,
            "info": serialize_ndarrays(self.info, use_bytes=use_bytes,
                                       bytes_type=bytes_type),
            }

    def _asdict(self):
        # support simplejson encoding
        return self.to_serializable()

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

            >>> import json
            ...
            >>> samples = dimod.SampleSet.from_samples([-1, 1, -1], dimod.SPIN, energy=-.5)
            >>> s = json.dumps(samples.to_serializable())
            >>> new_samples = dimod.SampleSet.from_serializable(json.loads(s))

        See also:
            :meth:`~.SampleSet.to_serializable`

        """

        version = obj["version"]["sampleset_schema"]
        if version < "3.0.0":
            raise ValueError("No longer supported serialization format")

        # assume we're working with v3

        # other data
        vartype = str(obj['variable_type'])  # cast to str for python2
        num_variables = obj['num_variables']
        variables = list(iter_deserialize_variables(obj['variable_labels']))
        info = deserialize_ndarrays(obj['info'])

        # vectors
        vectors = {name: deserialize_ndarray(data)
                   for name, data in obj['vectors'].items()}

        sample = deserialize_ndarray(obj['sample_data'])
        if obj.get('sample_packed', True):  # 3.1.0
            sample = unpack_samples(sample,
                                    n=num_variables,
                                    dtype=obj['sample_type'])

            if vartype == 'SPIN':
                sample *= 2
                sample -= 1

        return cls.from_samples((sample, variables), vartype, info=info,
                                **vectors)

    ###############################################################################################
    # Export to dataframe
    ###############################################################################################

    def to_pandas_dataframe(self, sample_column=False):
        """Convert a sample set to a Pandas DataFrame

        Args:
            sample_column(bool, optional, default=False): If True, samples are
            represented as a column of type dict.

        Returns:
            :obj:`pandas.DataFrame`

        Examples:
            >>> samples = dimod.SampleSet.from_samples([{'a': -1, 'b': +1, 'c': -1},
            ...                                         {'a': -1, 'b': -1, 'c': +1}],
            ...                                        dimod.SPIN, energy=-.5)
            >>> samples.to_pandas_dataframe()    # doctest: +SKIP
               a  b  c  energy  num_occurrences
            0 -1  1 -1    -0.5                1
            1 -1 -1  1    -0.5                1
            >>> samples.to_pandas_dataframe(sample_column=True)    # doctest: +SKIP
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
