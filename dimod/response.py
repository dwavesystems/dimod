"""
The dimod Response object allows dimod Samplers to respond consistently.

The Response is :class:`~collections.Iterable` (over the samples from lowest energy to highest) and
:class:`~collections.Sized` (the number of samples in the response).

Examples
--------

>>> response = dimod.ExactSolver().sample_ising({'a': -0.5}, {})
>>> len(response)
2
>>> for sample in response:
...     print(sample)
{'a': 1}
{'a': -1}

"""
from collections import Mapping, Iterable, Sized, namedtuple
import itertools

import numpy as np
from six import itervalues

from dimod.decorators import vartype_argument
from dimod.vartypes import Vartype

__all__ = ['Response']


class Response(Iterable, Sized):
    """A container for the samples and any other data returned by a dimod Sampler.

    Args:
        samples_matrix (:obj:`numpy.matrix`):
            A numpy matrix where each row is a sample.

        data_vectors (dict[field, :obj:`numpy.array`/list]):
            A dict containing additional per-sample data in vectors. Each vector should be the
            same length as samples_matrix. The key 'energy' and its vector are required.

        vartype (:class:`.Vartype`):
            The vartype of the samples.

        info (dict, optional, default=None):
            A dict containing information about the response as a whole.

        variable_labels (list, optional, default=None):
            Maps (by index) variable labels to the columns of the samples matrix.

    Attributes:
        vartype (:class:`.Vartype`): The vartype of the samples.

        info (dict): A dictionary containing information about the response as a whole.

        variable_labels (list/None): The variable labels. Each column in the samples matrix is the
            values assigned to one variable. If None then the column indices are the labels.

        label_to_idx (dict): A mapping from the variable labels to their columns in samples matrix.

    """

    @vartype_argument('vartype')
    def __init__(self, samples_matrix, data_vectors, vartype, info=None, variable_labels=None):
        # Constructor is opinionated about the samples_matrix type, it should be a numpy matrix
        if not isinstance(samples_matrix, np.matrix):
            raise TypeError("expected 'samples_matrix' to be a numpy matrix")
        elif samples_matrix.dtype != np.int8:
            # cast to int8
            samples_matrix = samples_matrix.astype(np.int8)

        self._samples_matrix = samples_matrix
        num_samples, num_variables = samples_matrix.shape

        if not isinstance(data_vectors, dict):
            raise TypeError("expected 'data_vectors' to be a dict")
        if 'energy' not in data_vectors:
            raise ValueError("energy must be provided")
        else:
            data_vectors = data_vectors.copy()  # shallow copy
            data_vectors['energy'] = np.asarray(data_vectors['energy'])
        for vector in data_vectors.values():
            # todo - check that is a vector and that has the right length
            if isinstance(vector, (np.ndarray, list)):
                if len(vector) != num_samples:
                    raise ValueError(("expected data vector {} to be a vector of length {}"
                                      "").format(vector, num_samples))
            else:
                raise TypeError("expected data vector {} to be a list of numpy array".format(vector))
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
                raise ValueError("variable_labels' length must match the number of variables in samples_matrix")

            self.label_to_idx = {v: idx for idx, v in enumerate(variable_labels)}

        self._futures = []

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
        """:obj:`numpy.matrix`: The numpy int8 matrix containing the samples."""
        if self._futures:
            self._from_futures()

        return self._samples_matrix

    @samples_matrix.setter
    def samples_matrix(self, mat):
        self._samples_matrix = mat

    @property
    def data_vectors(self):
        """dict[field, :obj:`numpy.array`/list]: The per-sample data. The keys should be the
        data labels and the values should each be a vector of the same length as sample_matrix.
        """
        if self._futures:
            self._from_futures()

        return self._data_vectors

    def done(self):
        """True if all loaded futures are done or if there are no futures.

        Only relevant when the response is constructed with :meth:`Response.from_futures`.
        """
        return all(future.done() for future in self._futures)

    ##############################################################################################
    # Construction and updates
    ##############################################################################################

    @classmethod
    def from_matrix(cls, samples, data_vectors, vartype=None, info=None, variable_labels=None):
        """Build a Response from an array-like object.

        Args:
            samples (array_like/str):
                As for :class:`numpy.matrix`. See Notes.

            data_vectors (dict[field, :obj:`numpy.array`/list]):
                A dict containing additional per-sample data in vectors. Each vector should be the
                same length as samples_matrix. The key 'energy' and its vector are required.

            vartype (:class:`.Vartype`, optional, default=None):
                The vartype of the response. If not provided, the vartype will be inferred from the
                samples matrix if possible or a ValueError will be raised.

            info (dict, optional, default=None):
                A dict containing information about the response as a whole.

            variable_labels (list, optional, default=None):
                Maps (by index) variable labels to the columns of the samples matrix.

        Returns:
            :obj:`.Response`

        Raises:
            :exc:`ValueError`: If vartype is not provided and samples are either all 1s or have more
                than two unique values or if those values are not a know vartype.

        Examples:
            .. code-block:: python

                samples = np.matrix([[0, 1], [1, 0]])
                energies = [0.0, 1.0]
                response = Response.from_matrix(samples, {'energy': energies})

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
        """Build a Response from an iterable of dicts.

        Args:
            samples (iterable[dict]):
                An iterable of samples where each sample is a dictionary (or Mapping).

            data_vectors (dict[field, :obj:`numpy.array`/list]):
                A dict containing additional per-sample data in vectors. Each vector should be the
                same length as samples_matrix. The key 'energy' and its vector are required.

            vartype (:class:`.Vartype`, optional, default=None):
                The vartype of the response. If not provided, the vartype will be inferred from the
                samples matrix if possible or a ValueError will be raised.

            info (dict, optional, default=None):
                A dict containing information about the response as a whole.

        Returns:
            :obj:`.Response`

        Raises:
            :exc:`ValueError`: If vartype is not provided and samples are either all 1s or have more
                than two unique values or if those values are not a know vartype.

        Examples:
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
        """Build a Response from a pandas DataFrame.

        Args:
            samples (:obj:`pandas.DataFrame`):
                A pandas DataFrame of samples where each row is a sample.

            data_vectors (dict[field, :obj:`numpy.array`/list]):
                A dict containing additional per-sample data in vectors. Each vector should be the
                same length as samples_matrix. The key 'energy' and its vector are required.

            vartype (:class:`.Vartype`, optional, default=None):
                The vartype of the response. If not provided, the vartype will be inferred from the
                samples matrix if possible or a ValueError will be raised.

            info (dict, optional, default=None):
                A dict containing information about the response as a whole.

        Returns:
            :obj:`.Response`

        Raises:
            :exc:`ValueError`: If vartype is not provided and samples are either all 1s or have more
                than two unique values or if those values are not a know vartype.

        Examples:
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
    def from_futures(cls):
        """Build a response from :obj:`~concurrent.futures.Future` objects.

        Note:
            Not yet implemented.

        """
        # concurrent.futures.as_completed
        raise NotImplementedError("support for python Future objects is forthcoming")

    def update(self, *other_responses):
        """Add other responses' values to the response.

        Args:
            *other_responses: (:obj:`.Response`):
                Any number of additional response objects. Must have matching sample_matrix
                dimensions, matching data_vector keys and identical variable labels.

        """

        # make sure all of the other responses are the appropriate vartype. We could cast them but
        # that would effect the energies so it is best to happen outside of this function.
        vartype = self.vartype
        for response in other_responses:
            if vartype is not response.vartype:
                raise ValueError("can only update with responses of matching vartype")

        # make sure that the variable labels are consistent
        variable_labels = self.variable_labels
        if variable_labels is None:
            __, num_variables = self.samples_matrix.shape
            variable_labels = list(range(num_variables))
            # in this case we need to allow for either None or variable_labels
            if not all(response.variable_labels is None or response.variable_labels == variable_labels):
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
        """Creates a shallow copy of the response."""
        return self.from_matrix(self.samples_matrix, self.data_vectors,
                                vartype=self.vartype, info=self.info,
                                variable_labels=self.variable_labels)

    @vartype_argument('vartype')
    def change_vartype(self, vartype, data_vector_offsets=None, inplace=True):
        """Creates a new response with the given vartype.

        Args:
            vartype (:class:`.Vartype`/str/set, optional):
                The variable type desired for the penalty model. Accepted input values:
                :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            inplace (bool, optional, default=True):
                If True, the response is updated in-place, otherwise a new response is returned.

        Returns:
            :obj:`.Response`. A new Response with vartype matching input 'vartype'.

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
        """Relabel the variables according to the given mapping.

        Args:
            mapping (dict):
                A dict mapping the current variable labels to new ones. If an incomplete mapping is
                provided unmapped variables will keep their labels

            inplace (bool, optional, default=True):
                If True, the response is updated in-place, otherwise a new response is returned.

        Returns:
            :class:`.Response`: A response with the variables relabeled. If inplace=True, returns
            itself.

        Examples:
            .. code-block:: python

                response = dimod.Response.from_dicts([{'a': -1}, {'a': +1}], {'energy': [-1, 1]})
                response.relabel_variables({'a': 0})

            .. code-block:: python

                response = dimod.Response.from_dicts([{'a': -1}, {'a': +1}], {'energy': [-1, 1]})
                new_response = response.relabel_variables({'a': 0}, inplace=False)

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

    def samples(self, sorted_by='energy'):
        """Iterate over the samples in the response.

        Args:
            sorted_by (str/None, optional, default='energy'):
                Over what data_vector to sort the samples. If None, the samples are yielded in
                the order given by the samples matrix.

        Yields:
            :obj:`.SampleView`: A view object mapping the variable labels to their values. Acts like
            a read-only dict.

        Examples:
            >>> response = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> response.samples_matrix
            matrix([[-1, -1],
                    [ 1, -1],
                    [ 1,  1],
                    [-1,  1]])
            >>> for sample in response.samples(sorted_by=None):
            ...     print(sample)
            {'a': -1, 'b': -1}
            {'a': 1, 'b': -1}
            {'a': 1, 'b': 1}
            {'a': -1, 'b': 1}
            >>> for sample in response.samples():  # sorted_by='energy'
            ...     print(sample)
            {'a': -1, 'b': -1}
            {'a': 1, 'b': -1}
            {'a': 1, 'b': 1}
            {'a': -1, 'b': 1}

        """
        if sorted_by is None:
            order = np.arange(len(self))
        else:
            order = np.argsort(self.data_vectors[sorted_by])

        samples = self.samples_matrix
        label_mapping = self.label_to_idx
        for idx in order:
            yield SampleView(idx, samples, label_mapping)

    def data(self, fields=None, sorted_by='energy', name='Sample'):
        """Iterate over the data in the response.

        Args:
            fields (list, optional, default=None):
                If specified, the yielded tuples will only include the values in fields.
                A special field name 'sample' can be used to view the samples.

            sorted_by (str/None, optional, default='energy'):
                Over what data_vector to sort the samples. If None, the samples are yielded in
                the order given by the samples matrix.

            name (str/None, optional, default='Sample'):
                The name of the yielded namedtuples or None to yield regular tuples.

        Yields:
            namedtuple/tuple: The data in the response, in the order specified by the input
            'fields'.

        Examples:
            >>> response = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
            >>> for datum in response.data():
            ...     print(datum)
            Sample(sample={'a': -1, 'b': -1}, energy=0.0)
            Sample(sample={'a': 1, 'b': -1}, energy=1.0)
            Sample(sample={'a': 1, 'b': 1}, energy=1.0)
            Sample(sample={'a': -1, 'b': 1}, energy=4.0)
            >>> for sample, energy in response.data():
            ...     print(energy)
            0.0
            1.0
            1.0
            4.0
            >>> for energy, in response.data(['energy']):
            ...     print(energy)
            0.0
            1.0
            1.0
            4.0

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
                    yield SampleView(idx, samples, label_mapping)
                else:
                    yield data_vectors[field][idx]

        for idx in order:
            yield _pack(_values(idx))


class SampleView(Mapping):
    """View each row of the samples matrix as if it was a dict."""
    def __init__(self, idx, samples, label_mapping=None):
        self.label_mapping = label_mapping
        self.idx = idx
        self.samples = samples

    def __getitem__(self, key):
        if self.label_mapping is not None:
            key = self.label_mapping[key]

        return int(self.samples[self.idx, key])

    def __iter__(self):
        # iterate over the variables
        return self.label_mapping.__iter__()

    def __len__(self):
        num_samples, num_variables = self.samples.shape
        return num_variables

    def __repr__(self):
        """Represents itself as as a dictionary"""
        return dict(self).__repr__()


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
