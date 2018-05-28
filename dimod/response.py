"""
dimod samplers respond with a consistent :class:`.Response` object that is
:class:`~collections.Iterable` (over samples, from lowest energy to highest) and
:class:`~collections.Sized` (number of samples).

Examples
--------
    This example shows the response of the dimod ExactSolver sampler.

>>> import dimod
>>> response = dimod.ExactSolver().sample_ising({'a': -0.5}, {})
>>> len(response)
2
>>> for sample in response:
...     print(sample)
{'a': 1}
{'a': -1}

"""
from collections import Mapping, Iterable, Sized, namedtuple, ValuesView, ItemsView
import itertools
import concurrent.futures

import numpy as np
from six import itervalues, iteritems

from dimod.decorators import vartype_argument
from dimod.utilities import resolve_label_conflict
from dimod.vartypes import Vartype

__all__ = ['Response']


class Response(Iterable, Sized):
    """A container for samples and any other data returned by dimod samplers.

    Args:
        samples_matrix (:obj:`numpy.matrix`):
            Samples as a NumPy matrix where each row is a sample.

        data_vectors (dict[field, :obj:`numpy.array`/list]):
            Additional per-sample data as a dict of vectors. Each vector is of the
            same length as `samples_matrix`. The key 'energy' and its vector are required.

        vartype (:class:`.Vartype`):
            Vartype of the samples.

        info (dict, optional, default=None):
            Information about the response as a whole formatted as a dict.

        variable_labels (list, optional, default=None):
            Variable labels mappped by index to columns of the samples matrix.

    Attributes:
        vartype (:class:`.Vartype`): Vartype of the samples.

        info (dict): Information about the response as a whole formatted as a dict.

        variable_labels (list/None): Variable labels. Each column in the samples matrix is the
            values returned for one variable. If None, column indices are the labels.

        label_to_idx (dict): Map of variable labels to columns in samples matrix.

    Examples:
        This example shows some attributes of the response for the sampler
        of dimod package's random_sampler.py reference example.

        >>> from dimod.reference.samplers.random_sampler import RandomSampler
        >>> sampler = RandomSampler()
        >>> bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 1.0}, {(0, 1): 0.5}, -0.5, dimod.SPIN)
        >>> response = sampler.sample(bqm)
        >>> response.vartype  # doctest: +SKIP
        <Vartype.SPIN: frozenset([1, -1])>  # doctest: +SKIP
        >>> response.variable_labels
        [0, 1]

    """

    @vartype_argument('vartype')
    def __init__(self, samples_matrix, data_vectors, vartype, info=None, variable_labels=None):
        # Constructor is opinionated about the samples_matrix type, it should be a numpy matrix
        if not isinstance(samples_matrix, np.matrix):
            raise TypeError("expected 'samples_matrix' to be a NumPy matrix")
        elif samples_matrix.dtype != np.int8:
            # cast to int8
            samples_matrix = samples_matrix.astype(np.int8)

        self._samples_matrix = samples_matrix
        num_samples, num_variables = samples_matrix.shape

        if not isinstance(data_vectors, Mapping):
            raise TypeError("expected 'data_vectors' to be a dict")
        if 'energy' not in data_vectors:
            raise ValueError("energy must be provided")
        else:
            data_vectors = dict(data_vectors)  # shallow copy

        for key, vector in iteritems(data_vectors):
            try:
                data_vectors[key] = vector = np.asarray(vector)
            except (ValueError, TypeError):
                raise TypeError("expected data vector {} to be array-like".format(key))

            shape = vector.shape
            if not shape or shape[0] != num_samples:
                raise ValueError(("expected data vector {} (shape {}) to have {} rows"
                                  "").format(key, vector.shape, num_samples))

        self._data_vectors = data_vectors

        # vartype is checked by the decorator
        self.vartype = vartype

        if info is None:
            info = {}
        elif not isinstance(info, dict):
            raise TypeError("expected 'info' to be a dict.")
        else:
            info = dict(info)  # make a shallow copy
        self.info = info

        if variable_labels is None:
            self.variable_labels = None
            self.label_to_idx = None
        else:
            self.variable_labels = variable_labels = list(variable_labels)
            if len(variable_labels) != num_variables:
                msg = ("variable_labels' length must match the number of columns in "
                       "samples_matrix, {} labels, matrix has {} columns".format(len(variable_labels), num_variables))
                raise ValueError(msg)

            self.label_to_idx = {v: idx for idx, v in enumerate(variable_labels)}

        # will store any pending Future objects and data about them
        self._futures = {}

    def __len__(self):
        """The number of samples."""
        num_samples, num_variables = self.samples_matrix.shape
        return num_samples

    def __iter__(self):
        """Iterate over the samples, low energy to high."""
        return self.samples(sorted_by='energy')

    def __str__(self):
        # developer note: it would be nice if the variable labels (if present could be printed)
        return self.samples_matrix.__str__()

    ##############################################################################################
    # Properties
    ##############################################################################################

    @property
    def samples_matrix(self):
        """:obj:`numpy.matrix`: Samples as a NumPy matrix of data type int8.

        Examples:
            This example shows the samples of dimod package's ExactSolver reference sampler
            formatted as a NumPy matrix.

            >>> import dimod
            >>> response = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> response.samples_matrix
            matrix([[-1, -1],
                    [ 1, -1],
                    [ 1,  1],
                    [-1,  1]])

        """
        if self._futures:
            self._resolve_futures(**self._futures)

        return self._samples_matrix

    @samples_matrix.setter
    def samples_matrix(self, mat):
        self._samples_matrix = mat

    @property
    def data_vectors(self):
        """dict[field, :obj:`numpy.array`/list]: Per-sample data as a dict, where keys are the
        data labels and values are each a vector of the same length as sample_matrix.

        Examples:
            This example shows the returned energies of dimod package's ExactSolver
            reference sampler.

            >>> import dimod
            >>> response = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> response.data_vectors['energy']
            array([-1.5, -0.5, -0.5,  2.5])

        """
        if self._futures:
            self._resolve_futures(**self._futures)

        return self._data_vectors

    def done(self):
        """True if all loaded futures are done or if there are no futures.

        Only relevant when the response is constructed with :meth:`Response.from_futures`.

        Examples:
            This example checks whether futures are done before and after a `set_result` call.

            >>> from concurrent.futures import Future
            >>> future = Future()
            >>> response = dimod.Response.from_futures((future,), dimod.BINARY, 3)
            >>> future.done()
            False
            >>> future.set_result({'samples': [0, 1, 0], 'energy': [1]})
            >>> future.done()
            True

        """
        return all(future.done() for future in self._futures)

    ##############################################################################################
    # Construction and updates
    ##############################################################################################

    @classmethod
    def from_matrix(cls, samples, data_vectors, vartype=None, info=None, variable_labels=None):
        """Build a response from a NumPy array-like object.

        Args:
            samples (array_like/str):
                Samples as a :class:`numpy.matrix` or NumPy array-like object. See Notes.

            data_vectors (dict[field, :obj:`numpy.array`/list]):
                Additional per-sample data as a dict of vectors. Each vector is the same length as
                `samples_matrix`. The key 'energy' and its vector are required.

            vartype (:class:`.Vartype`, optional, default=None):
                Vartype of the response. If not provided, vartype is inferred from the
                samples matrix if possible or a ValueError is raised.

            info (dict, optional, default=None):
                Information about the response as a whole formatted as a dict.

            variable_labels (list, optional, default=None):
                Maps (by index) variable labels to the columns of the samples matrix.

        Returns:
            :obj:`.Response`: A `dimod` :obj:`.Response` object based on the input
            NumPy array-like object.

        Raises:
            :exc:`ValueError`: If vartype is not provided and samples are all 1s, have more
                than two unique values, or have values of an unknown vartype.

        Examples:
            This example code snippet builds a response from a NumPy matrix.

            .. code-block:: python

                samples = np.matrix([[0, 1], [1, 0]])
                energies = [0.0, 1.0]
                response = Response.from_matrix(samples, {'energy': energies})

            This example code snippet builds a response from a NumPy array-like object (a Python list).

            .. code-block:: python

                samples = [[0, 1], [1, 0]]
                energies = [0.0, 1.0]
                response = Response.from_matrix(samples, {'energy': energies})

        Notes:
            SciPy defines array_like in the following way: "In general, numerical data arranged in
            an array-like structure in Python can be converted to arrays through the use of the
            array() function. The most obvious examples are lists and tuples. See the documentation
            for array() for details for its use. Some objects may support the array-protocol and
            allow conversion to arrays this way. A simple way to find out if the object can be
            converted to a numpy array using array() is simply to try it interactively and see if it
            works! (The Python Way)." [array_like]_

        References:
            .. [array_like] Docs.scipy.org. (2018). Array creation - NumPy v1.14 Manual. [online]
                Available at: https://docs.scipy.org/doc/numpy/user/basics.creation.html
                [Accessed 16 Feb. 2018].

        """
        samples_matrix = np.matrix(samples, dtype=np.int8)

        if vartype is None:
            vartype = infer_vartype(samples_matrix)

        response = cls(samples_matrix, data_vectors=data_vectors,
                       vartype=vartype, info=info, variable_labels=variable_labels)

        return response

    @classmethod
    def from_dicts(cls, samples, data_vectors, vartype=None, info=None):
        """Build a response from an iterable of dicts.

        Args:
            samples (iterable[dict]):
                Iterable of samples where each sample is a dictionary (or Mapping).

            data_vectors (dict[field, :obj:`numpy.array`/list]):
                Additional per-sample data as a dict of vectors. Each vector is the
                same length as `samples_matrix`. The key 'energy' and its vector are required.

            vartype (:class:`.Vartype`, optional, default=None):
                Vartype of the response. If not provided, vartype is inferred from the
                samples matrix if possible or a ValueError is raised.

            info (dict, optional, default=None):
                Information about the response as a whole formatted as a dict.

        Returns:
            :obj:`.Response`: A `dimod` :obj:`.Response` object based on the input dicts.

        Raises:
            :exc:`ValueError`: If vartype is not provided and samples are all 1s, have more
                than two unique values, or have values of an unknown vartype.

        Examples:
            This example code snippet builds a response from an interable of samples and
            its corresponding dict of energies.

            .. code-block:: python

                samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1}]
                energies = [-1.0, -1.0]
                response = Response.from_dicts(samples, {'energy': energies})

        """
        samples = iter(samples)

        # get the first sample
        first_sample = next(samples)

        try:
            variable_labels = sorted(first_sample)
        except TypeError:
            # unlike types cannot be sorted in python3
            variable_labels = list(first_sample)
        num_variables = len(variable_labels)

        def _iter_samples():
            yield np.fromiter((first_sample[v] for v in variable_labels),
                              count=num_variables, dtype=np.int8)

            try:
                for sample in samples:
                    yield np.fromiter((sample[v] for v in variable_labels),
                                      count=num_variables, dtype=np.int8)
            except KeyError:
                msg = ("Each dict in 'samples' must have the same keys.")
                raise ValueError(msg)

        samples_matrix = np.matrix(np.stack(list(_iter_samples())))

        return cls.from_matrix(samples_matrix, data_vectors=data_vectors, vartype=vartype,
                               info=info, variable_labels=variable_labels)

    @classmethod
    def from_pandas(cls, samples_df, data_vectors, vartype=None, info=None):
        """Build a response from a pandas DataFrame.

        Args:
            samples (:obj:`pandas.DataFrame`):
                A pandas DataFrame of samples where each row is a sample.

            data_vectors (dict[field, :obj:`numpy.array`/list]):
                Additional per-sample data as a dict of vectors. Each vector is the
                same length as `samples_matrix`. The key 'energy' and its vector are required.

            vartype (:class:`.Vartype`, optional, default=None):
                Vartype of the response. If not provided, vartype is inferred from the
                samples matrix if possible or a ValueError is raised.

            info (dict, optional, default=None):
                Information about the response as a whole formatted as a dict.

        Returns:
            :obj:`.Response`: A `dimod` :obj:`.Response` object based on the input DataFrame.

        Raises:
            :exc:`ValueError`: If vartype is not provided and samples are all 1s, have more
                than two unique values, or have values of an unknown vartype.

        Examples:
            These example code snippets build a response from a pandas DataFrame.

            .. code-block:: python

                import pandas as pd

                samples = pd.DataFrame([{'a': 1, 'b': 0}, {'a': 0, 'b': 0}], dtype='int8')
                response = Response.from_pandas(samples, {energy: [1, 0]})

            .. code-block:: python

                import pandas as pd

                samples = pd.DataFrame([[+1, -1]], dtype='int8', columns=['v1', 'v2'])
                response = Response.from_pandas(samples, {energy: [1]})

        """
        import pandas as pd

        variable_labels = list(samples_df.columns)
        samples_matrix = samples_df.as_matrix(columns=variable_labels)

        if isinstance(data_vectors, pd.DataFrame):
            raise NotImplementedError("support for DataFrame data_vectors is forthcoming")

        return cls.from_matrix(samples_matrix, data_vectors, vartype=vartype, info=info,
                               variable_labels=variable_labels)

    @classmethod
    def empty(cls, vartype):
        empty_samples_matrix = np.matrix(np.empty((0, 0), dtype=np.int8))
        empty_data_vectors = {'energy': []}
        return cls(samples_matrix=empty_samples_matrix, data_vectors=empty_data_vectors,
                   vartype=vartype)

    @classmethod
    def from_futures(cls, futures, vartype, num_variables,
                     samples_key='samples', data_vector_keys=None,
                     info_keys=None, variable_labels=None, active_variables=None,
                     ignore_extra_keys=True):
        """Build a response from :obj:`~concurrent.futures.Future`-like objects.

        Args:
            futures (iterable):
                Iterable :obj:`~concurrent.futures.Future` or :obj:`~concurrent.futures.Future`-like
                objects (Python objects with similar structure).
                :meth:`~concurrent.futures.Future.result` returns a dict.

            vartype (:class:`.Vartype`):
                Vartype of the response.

            num_variables (int):
                Number of variables for each sample.

            samples_key (hashable, optional, default='samples'):
                Key of the result dict containing the samples. Samples are array-like.

            data_vector_keys (iterable/mapping, optional, default=None):
                A mapping from the keys of the result dict to :attr:`Response.data_vectors`. If
                None, ['energy'] is assumed to be a key in the result dict and the
                'energy' data vector is mapped.

            info_keys (iterable/mapping, optional, default=None):
                A mapping from the keys of the result dict to :attr:`Response.info`.
                If None, info is empty.

            variable_labels (list, optional, default=None):
                Maps (by index) variable labels to columns of the samples matrix.

            active_variables (array-like, optional, default=None):
                Selects which columns of the result's samples are used. If `variable_labels` is
                not provided, `variable_labels` is set to match `active_variables`.

            ignore_extra_keys (bool, optional, default=True):
                If True, keys given in `data_vector_keys` and `info_keys` but that are not in
                :meth:`~concurrent.futures.Future.result` are ignored. If False, extra keys
                cause a ValueError.

        Returns:
            :obj:`.Response`: A `dimod` :obj:`.Response` object based on the input
            Future-like objects.

        Notes:
            :obj:`~concurrent.futures.Future` objects are read on the first read
            of :attr:`.Response.samples_matrix` or :attr:`.Response.data_vectors`.

        Examples:
            These example code snippets build responses from :obj:`~concurrent.futures.Future` objects.

            .. code-block:: python

                from concurrent.futures import Future

                future = Future()

                # load the future into response
                response = dimod.Response.from_futures((future,), dimod.BINARY, 3)

                future.set_result({'samples': [0, 1, 0], 'energy': [1]})

                # now read from the response
                matrix = response.samples_matrix

            .. code-block:: python

                from concurrent.futures import Future

                future = Future()

                # load the future into response
                response = dimod.Response.from_futures((future,), dimod.BINARY, 3,
                                                       active_variables=[0, 1, 3])

                future.set_result({'samples': [0, 1, 3, 0], 'energy': [1]})

                # now read from the response, this matrix
                matrix = response.samples_matrix

            .. code-block:: python

                from concurrent.futures import Future

                future = Future()

                # load the future into response
                response = dimod.Response.from_futures((future,), dimod.BINARY, 3,
                                                       data_vector_keys={'en': 'energy'})

                future.set_result({'samples': [0, 1, 0], 'en': [1]})

                # now read from the response
                matrix = response.samples_matrix

        """

        if data_vector_keys is None:
            data_vector_keys = {'energy': 'energy'}
        elif isinstance(data_vector_keys, Mapping):
            data_vector_keys = dict(data_vector_keys)
        else:
            data_vector_keys = {key: key for key in data_vector_keys}  # identity mapping

        if info_keys is None:
            info_keys = {}
        elif isinstance(info_keys, Mapping):
            info_keys = dict(info_keys)
        else:
            info_keys = {key: key for key in info_keys}

        if active_variables is not None:
            if variable_labels is None:
                variable_labels = active_variables
            elif len(variable_labels) != len(active_variables):
                raise ValueError("active_variables and variable_labels should have the same length")

        response = cls.empty(vartype)

        # now dump all of the remaining information into the _futures
        response._futures = {'futures': futures,
                             'samples_key': samples_key,
                             'data_vector_keys': data_vector_keys,
                             'info_keys': info_keys,
                             'variable_labels': variable_labels,
                             'active_variables': active_variables,
                             'ignore_extra_keys': ignore_extra_keys}

        return response

    def _resolve_futures(self, futures, samples_key, data_vector_keys, info_keys,
                         variable_labels, active_variables, ignore_extra_keys):

        # first reset the futures to avoid recursion errors
        self._futures = {}

        # `dwave.cloud.qpu.computation.Future` is not yet interchangeable with
        # `concurrent.futures.Future`, so we need to detect the kind of future
        # we're dealing with.
        futures = list(futures)  # if generator
        if hasattr(futures[0], 'as_completed'):
            as_completed = futures[0].as_completed
        else:
            as_completed = concurrent.futures.as_completed

        # combine all samples from all futures into a single response
        for future in as_completed(futures):
            result = dict(future.result())  # create a shallow copy

            # first get the samples matrix and filter out any inactive variables
            samples = np.matrix(result.pop(samples_key), dtype=np.int8)
            if active_variables is not None:
                samples = samples[:, active_variables]

            # next get the data vectors
            if ignore_extra_keys:
                data_vectors = {}
                for source_key, key in iteritems(data_vector_keys):
                    try:
                        data_vectors[key] = result[source_key]
                    except KeyError:
                        pass
                info = {}
                for source_key, key in iteritems(info_keys):
                    try:
                        info[key] = result[source_key]
                    except KeyError:
                        pass
            else:
                data_vectors = {}
                for source_key, key in iteritems(data_vector_keys):
                    try:
                        data_vectors[key] = result[source_key]
                    except KeyError:
                        raise ValueError("data vector key '{}' not in Future.result()".format(key))
                info = {}
                for source_key, key in iteritems(info_keys):
                    try:
                        info[key] = result[source_key]
                    except KeyError:
                        raise ValueError("info key '{}' not in Future.result()".format(key))

            # now get the appropriate response
            response = self.__class__.from_matrix(samples, data_vectors=data_vectors, info=info,
                                                  vartype=self.vartype, variable_labels=variable_labels)
            self.update(response)

    def update(self, *other_responses):
        """Add values of other responses to the response.

        Args:
            *other_responses: (:obj:`.Response`):
                Additional responses from which to add values. Any number of additional response objects,
                separated by commas, can be specified. Responses must have matching `sample_matrix`
                dimensions, `data_vector` keys, and variable labels.

        Examples:
            This example updates a response with values from two other responses.

            >>> import dimod
            >>> samples = [[0, 1], [1, 0]]
            >>> energies = [0.0, 1.0]
            >>> response = dimod.Response.from_matrix(samples, {'energy': energies})
            >>> samples1 = [[0, 0], [1, 1]]
            >>> energies1 = [0.25, 1.25]
            >>> response1 = dimod.Response.from_matrix(samples1, {'energy': energies1})
            >>> samples2 = [[1, 0], [0, 1]]
            >>> energies2 = [0.5, 1.75]
            >>> response2 = dimod.Response.from_matrix(samples2, {'energy': energies2})
            >>> len(response)
            2
            >>> for i in response.data():         # doctest: +SKIP
            ...     print(i)
            ...
            Sample(sample={0: 0, 1: 1}, energy=0.0)
            Sample(sample={0: 1, 1: 0}, energy=1.0)
            >>> response.update(response1, response2)
            >>> len(response)
            6
            >>> for energy in response.data(fields=['energy'], name='UpdatedEnergy'):
            ...     print(energy)
            ...
            UpdatedEnergy(energy=0.0)
            UpdatedEnergy(energy=0.25)
            UpdatedEnergy(energy=0.5)
            UpdatedEnergy(energy=1.0)
            UpdatedEnergy(energy=1.25)
            UpdatedEnergy(energy=1.75)

        """
        # make sure all of the other responses are the appropriate vartype. We could cast them but
        # that would effect the energies so it is best to happen outside of this function.
        vartype = self.vartype
        for response in other_responses:
            if vartype is not response.vartype:
                raise ValueError("can only update with responses of matching vartype base")

        # if self is empty, then we are done
        if not self:
            other_responses = list(other_responses)

            response = other_responses.pop()

            self.samples_matrix = response.samples_matrix
            self.data_vectors.update(response.data_vectors)
            self.info.update(response.info)
            self.variable_labels = response.variable_labels
            self.label_to_idx = response.label_to_idx

            if other_responses:
                self.update(*other_responses)

            return

        # make sure that the variable labels are consistent
        variable_labels = self.variable_labels
        if variable_labels is None:
            __, num_variables = self.samples_matrix.shape
            variable_labels = list(range(num_variables))
            # in this case we need to allow for either None or variable_labels
            if not all(response.variable_labels is None or response.variable_labels == variable_labels
                       for response in other_responses):
                raise ValueError("cannot update responses with unlike variable labels")
        else:
            if not all(response.variable_labels == variable_labels for response in other_responses):
                raise ValueError("cannot update responses with unlike variable labels")

        # concatenate all of the matrices
        matrices = [self.samples_matrix]
        matrices.extend([response.samples_matrix for response in other_responses])
        self.samples_matrix = np.concatenate(matrices)

        # group all of the data vectors
        for key in self.data_vectors:
            vectors = [self.data_vectors[key]]
            vectors.extend(response.data_vectors[key] for response in other_responses)
            self.data_vectors[key] = np.concatenate(vectors)

        # finally update the response info
        for response in other_responses:
            self.info.update(response.info)

    ###############################################################################################
    # Transformations and Copies
    ###############################################################################################

    def copy(self):
        """Creates a shallow copy of a response.

        Examples:
            This example copies a response.

            >>> import dimod
            >>> samples = [[0, 1], [1, 0]]
            >>> energies = [0.0, 1.0]
            >>> response = dimod.Response.from_matrix(samples, {'energy': energies})
            >>> copied_response = response.copy()

        """
        return self.from_matrix(self.samples_matrix, self.data_vectors,
                                vartype=self.vartype, info=self.info,
                                variable_labels=self.variable_labels)

    @vartype_argument('vartype')
    def change_vartype(self, vartype, data_vector_offsets=None, inplace=True):
        """Create a new response with the given vartype.

        Args:
            vartype (:class:`.Vartype`/str/set):
                Variable type to use for the new response. Accepted input values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            data_vector_offsets (dict[field, :obj:`numpy.array`/list], optional, default=None):
                Offsets to add to `data_vectors` of the response formatted as a dict containing
                per-sample offsets in vectors. Each vector is the same length as `samples_matrix`.

            inplace (bool, optional, default=True):
                If True, the response is updated in-place, otherwise a new response is returned.

        Returns:
            :obj:`.Response`. New response with vartype matching input 'vartype'.

        Examples:
            This example converts the response of the dimod package's ExactSolver sampler
            to binary and adds offsets.

            >>> import dimod
            >>> response = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> response_binary = response.change_vartype('BINARY',
            ...                   data_vector_offsets={'energy': [0, 0.1, 0.2, 0.3]},
            ...                   inplace=False)
            >>> response_binary.vartype
            <Vartype.BINARY: frozenset([0, 1])>
            >>> for datum in response_binary.data():    # doctest: +SKIP
            ...    print(datum)
            ...
            Sample(sample={'a': 0, 'b': 0}, energy=-1.5)
            Sample(sample={'a': 1, 'b': 0}, energy=-0.4)
            Sample(sample={'a': 1, 'b': 1}, energy=-0.3)
            Sample(sample={'a': 0, 'b': 1}, energy=2.8)

            This example code snippet creates a response with spin variables from a response
            with binary variables while adding energy offsets to the new response.

            .. code-block:: python

                import pandas as pd
                samples = [[0, 1], [1, 0]]
                energies = [0.0, 1.0]
                response = dimod.Response.from_matrix(samples, {'energy': energies})
                offsets = {'energy': [0.25, 0.75]}
                response.change_vartype('SPIN',
                                         data_vector_offsets = offsets)
        """
        if not inplace:
            return self.copy().change_vartype(vartype, data_vector_offsets=data_vector_offsets, inplace=True)

        if data_vector_offsets is not None:
            for key in data_vector_offsets:
                self.data_vectors[key] += data_vector_offsets[key]

        if vartype is self.vartype:
            return self

        if vartype is Vartype.SPIN and self.vartype is Vartype.BINARY:
            self.samples_matrix = 2 * self.samples_matrix - 1
            self.vartype = vartype
        elif vartype is Vartype.BINARY and self.vartype is Vartype.SPIN:
            self.samples_matrix = (self.samples_matrix + 1) // 2
            self.vartype = vartype
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
            This example relabels the response of the dimod package's ExactSolver sampler and
            saves it as a new response.

            >>> import dimod
            >>> response = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> new_response = response.relabel_variables({'a': 0, 'b': 1}, inplace=False)
            >>> [next(new_response.samples())[x] for x in [0, 1]]
            [-1, -1]            


            This example code snippet relabels variables in a response.

            .. code-block:: python

                response = dimod.Response.from_dicts([{'a': -1}, {'a': +1}], {'energy': [-1, 1]})
                response.relabel_variables({'a': 0})

        """
        if not inplace:
            return self.copy().relabel_variables(mapping, inplace=True)

        # we need labels
        if self.variable_labels is None:
            __, num_variables = self.samples_matrix.shape
            self.variable_labels = list(range(num_variables))

        try:
            old_labels = set(mapping)
            new_labels = set(itervalues(mapping))
        except TypeError:
            raise ValueError("mapping targets must be hashable objects")

        for v in new_labels:
            if v in self.variable_labels and v not in old_labels:
                raise ValueError(('A variable cannot be relabeled "{}" without also relabeling '
                                  "the existing variable of the same name").format(v))

        shared = old_labels & new_labels
        if shared:
            old_to_intermediate, intermediate_to_new = resolve_label_conflict(mapping, old_labels, new_labels)

            self.relabel_variables(old_to_intermediate, inplace=True)
            self.relabel_variables(intermediate_to_new, inplace=True)
            return self

        self.variable_labels = variable_labels = [mapping.get(v, v) for v in self.variable_labels]
        self.label_to_idx = {v: idx for idx, v in enumerate(variable_labels)}
        return self

    ###############################################################################################
    # Viewing a Response
    ###############################################################################################

    def samples(self, n=None, sorted_by='energy'):
        """Iterate over the samples in the response.

        Args:
            n (int, optional, default=None):
                The maximum number of samples to provide. If None, all are provided.

            sorted_by (str/None, optional, default='energy'):
                Selects the `data_vector` used to sort the samples. If None, the samples are yielded in
                the order given by the samples matrix.

        Yields:
            :obj:`.SampleView`: A view object mapping the variable labels to their values. Acts like
            a read-only dict.

        Examples:
            This example iterates over the response samples of the dimod ExactSolver sampler.

            >>> import dimod
            >>> response = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> for sample in response.samples():    # sorted_by='energy'
            ...     print(sample['a']==sample['b'])
            ...
            True
            False
            True
            False
            >>> for sample in response.samples(sorted_by=None):   # doctest: +SKIP
            ...     print(sample)
            ...
            {'a': -1, 'b': -1}
            {'a': 1, 'b': -1}
            {'a': 1, 'b': 1}
            {'a': -1, 'b': 1}

        """
        num_samples = len(self)

        if n is not None:
            for sample in itertools.islice(self.samples(n=None, sorted_by=sorted_by), n):
                yield sample
            return

        if sorted_by is None:
            order = np.arange(num_samples)
        else:
            order = np.argsort(self.data_vectors[sorted_by])

        samples = self.samples_matrix
        label_mapping = self.label_to_idx

        for idx in order:
            yield SampleView(idx, self)

    def data(self, fields=None, sorted_by='energy', name='Sample'):
        """Iterate over the data in the response.

        Args:
            fields (list, optional, default=None):
                If specified, only these fields' values are included in the yielded tuples.
                The special field name 'sample' can be used to view the samples.

            sorted_by (str/None, optional, default='energy'):
                Selects the data_vector used to sort the samples. If None, the samples are yielded
                in the order given by the samples matrix.

            name (str/None, optional, default='Sample'):
                Name of the yielded namedtuples or None to yield regular tuples.

        Yields:
            namedtuple/tuple: The data in the response, in the order specified by the input
            `fields`.

        Examples:
            This example iterates over the response data of the dimod ExactSolver sampler.

            >>> import dimod
            >>> response = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> for datum in response.data():   # doctest: +SKIP
            ...     print(datum)
            ...
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
        if fields is None:
            fields = ['sample']
            fields.extend(self.data_vectors)

        if sorted_by is None:
            order = np.arange(len(self))
        else:
            order = np.argsort(self.data_vectors[sorted_by])

        if name is None:
            # yielding a tuple
            def _pack(values):
                return tuple(values)
        else:
            # yielding a named tuple
            SampleTuple = namedtuple(name, fields)

            def _pack(values):
                return SampleTuple(*values)

        samples = self.samples_matrix
        label_mapping = self.label_to_idx
        data_vectors = self.data_vectors

        def _values(idx):
            for field in fields:
                if field == 'sample':
                    yield SampleView(idx, self)
                else:
                    yield data_vectors[field][idx]

        for idx in order:
            yield _pack(_values(idx))


class SampleView(Mapping):
    """View each row of the samples matrix as if it was a dict."""
    def __init__(self, idx, response):
        self._idx = idx  # row of response._samples_matrix
        self._response = response

    def __getitem__(self, key):
        label_mapping = self._response.label_to_idx
        if label_mapping is not None:
            key = label_mapping[key]
        return int(self._response.samples_matrix[self._idx, key])

    def __iter__(self):
        # iterate over the variables
        label_mapping = self._response.label_to_idx
        if label_mapping is None:
            __, num_variables = self._response.samples_matrix.shape
            return iter(range(num_variables))
        return label_mapping.__iter__()

    def __len__(self):
        __, num_variables = self._response.samples_matrix.shape
        return num_variables

    def __repr__(self):
        """Represents itself as as a dictionary"""
        return dict(self).__repr__()

    def values(self):
        return SampleValuesView(self)

    def items(self):
        return SampleItemsView(self)


class SampleItemsView(ItemsView):
    """Faster read access to the numpy matrix"""
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        variable_labels = self._mapping._response.variable_labels
        samples_matrix = self._mapping._response.samples_matrix
        idx = self._mapping._idx
        if variable_labels is None:
            for v, val in enumerate(np.nditer(samples_matrix[idx, :], order='C', op_flags=['readonly'])):
                yield (v, int(val))
        else:
            for v, val in zip(variable_labels, np.nditer(samples_matrix[idx, :], order='C', op_flags=['readonly'])):
                yield (v, int(val))


class SampleValuesView(ValuesView):
    """Faster read access to the numpy matrix"""
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        samples_matrix = self._mapping._response.samples_matrix
        for val in np.nditer(samples_matrix[self._mapping._idx, :], op_flags=['readonly']):
            yield int(val)


def infer_vartype(samples_matrix):
    """Try to determine the Vartype of the samples matrix based on its values.

    Args:
        samples_matrix (:object:`numpy.ndarray`):
            An array or matrix of samples.

    Returns:
        :class:`.Vartype`

    Raises:
        ValueError: If the matrix is all ones, contains values other than -1, 1, 0 or contains more
        than two unique values.

    """
    ones_matrix = samples_matrix == 1

    if np.all(ones_matrix):
        msg = ("ambiguous vartype - an empty samples_matrix or one where all the values "
               "are all 1 must have the vartype specified by setting vartype=dimod.SPIN or "
               "vartype=dimod.BINARY.")
        raise ValueError(msg)

    if np.all(ones_matrix + (samples_matrix == 0)):
        return Vartype.BINARY
    elif np.all(ones_matrix + (samples_matrix == -1)):
        return Vartype.SPIN
    else:
        sample_vals = set(int(v) for v in np.nditer(samples_matrix)) - {-1, 1, 0}
        if sample_vals:
            msg = ("samples_matrix includes unknown values {}").format(sample_vals)
        else:
            msg = ("samples_matrix includes both -1 and 0 values")
        raise ValueError(msg)
