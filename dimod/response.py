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

from collections import namedtuple

import numpy as np

from dimod.decorators import vartype_argument
from dimod.sampleset import SampleSet, SampleView
from dimod.vartypes import Vartype

__all__ = ['Response']


class Response(SampleSet):
    """Samples and any other data returned by dimod samplers.

    Args:
        record (:obj:`numpy.recarray`)
            A numpy record array. Must have 'sample', 'energy' and 'num_occurrences' as fields.
            The 'sample' field should be a 2D numpy int8 array where each row is a sample and each
            column represents the value of a variable.

        labels (list):
            A list of variable labels.

        info (dict):
            Information about the response as a whole formatted as a dict.

        vartype (:class:`.Vartype`/str/set):
            Variable type for the response. Accepted input values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

    Examples:
        >>> import dimod
        ...
        >>> sampler = dimod.ExactSolver()
        >>> response = sampler.sample_ising({'a': -0.5, 'b': -0.5}, {('a', 'b'): -1.0})
        >>> response.record.sample
        array([[-1, -1],
               [ 1, -1],
               [ 1,  1],
               [-1,  1]], dtype=int8)
        >>> response.record.energy
        array([ 0.,  1., -2.,  1.])
        >>> response.variable_labels # doctest: +SKIP
        ['a', 'b']
        >>> response.labels_to_idx['b'] # doctest: +SKIP
        1
        >>> response.vartype is dimod.SPIN
        True
        >>> for sample, energy in response.data(['sample', 'energy']):  # doctest: +SKIP
        ...     print(sample, energy)
        {'a': 1, 'b': 1} -2.0
        {'a': -1, 'b': -1} 0.0
        {'a': 1, 'b': -1} 1.0
        {'a': -1, 'b': 1} 1.0

    """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super(Response, self).__init__(*args, **kwargs)

        import warnings

        warnings.warn("dimod.Response is deprecated, please use dimod.SampleSet instead.",
                      DeprecationWarning)

    ###############################################################################################
    # Properties
    ###############################################################################################

    @property
    def variable_labels(self):
        """list: Variable labels of the samples.

        Corresponds to the columns of the sample field of :attr:`.Response.record`.
        """
        return self.variables

    @property
    def label_to_idx(self):
        """dict: Maps the variable labels to the column in :attr:`.Response.record`."""
        return self.variables.index

    ###############################################################################################
    # Constructors
    ###############################################################################################

    @classmethod
    def from_samples(cls, samples_like, vectors, info, vartype, variable_labels=None):
        """Build a response from samples.

        Args:
            samples_like:
                A collection of samples. 'samples_like' is an extension of NumPy's array_like
                to include an iterable of sample dictionaries (as returned by
                :meth:`.Response.samples`).

            data_vectors (dict[field, :obj:`numpy.array`/list]):
                Additional per-sample data as a dict of vectors. Each vector is the
                same length as `samples_matrix`. The key 'energy' and it's vector is required.

            info (dict):
                Information about the response as a whole formatted as a dict.

            vartype (:class:`.Vartype`/str/set):
                Variable type for the response. Accepted input values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            variable_labels (list, optional):
                Determines the variable labels if samples_like is not an iterable of dictionaries.
                If samples_like is not an iterable of dictionaries and if variable_labels is not
                provided then index labels are used.

        Returns:
            :obj:`.Response`

        Examples:
            From dicts

            >>> import dimod
            ...
            >>> samples = [{'a': -1, 'b': +1}, {'a': -1, 'b': -1}]
            >>> response = dimod.Response.from_samples(samples, {'energy': [-1, 0]}, {}, dimod.SPIN)

            From an array

            >>> import dimod
            >>> import numpy as np
            ...
            >>> samples = np.ones((2, 3), dtype='int8')  # 2 samples, 3 variables
            >>> response = dimod.Response.from_samples(samples, {'energy': [-1.0, -1.0]}, {},
            ...                                        dimod.SPIN, variable_labels=['a', 'b', 'c'])


        """

        # there is no np.is_array_like so we use a try-except block
        try:
            # trying to cast it to int8 rules out list of dictionaries. If we didn't try to cast
            # then it would just create a vector of np.object
            samples = np.asarray(samples_like, dtype=np.int8)
        except TypeError:
            # if labels are None, they are set here
            samples, variable_labels = _samples_dicts_to_array(samples_like, variable_labels)

        assert samples.dtype == np.int8, 'sanity check'

        record = data_struct_array(samples, **vectors)

        # if labels are still None, set them here. We could do this in an else in the try-except
        # block, but the samples-array might not have the correct shape
        if variable_labels is None:
            __, num_variables = record.sample.shape
            variable_labels = list(range(num_variables))

        return cls(record, variable_labels, info, vartype)


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
        dtype = float if kwarg == 'energy' else None
        datavectors[kwarg] = vector = np.asarray(vector, dtype)

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
