"""todo"""
from collections import Mapping, Iterable, Sized, namedtuple
import itertools

import numpy as np

from dimod.compatibility23 import itervalues
from dimod.decorators import vartype_argument
from dimod.vartypes import Vartype

__all__ = ['Response']


class Response(Iterable, Sized):
    """todo"""

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
        if self.variable_labels is None:
            return self.samples_matrix.__str__()
        else:
            raise NotImplementedError

    ##############################################################################################
    # Properties
    ##############################################################################################

    @property
    def samples_matrix(self):
        """todo"""
        if self._futures:
            self._from_futures()

        return self._samples_matrix

    @samples_matrix.setter
    def samples_matrix(self, mat):
        self._samples_matrix = mat

    @property
    def data_vectors(self):
        """todo"""
        if self._futures:
            self._from_futures()

        return self._data_vectors

    def done(self):
        """todo"""
        return all(future.done() for future in self._futures)

    ##############################################################################################
    # Construction and updates
    ##############################################################################################

    @classmethod
    def from_matrix(cls, samples, data_vectors, vartype=None, info=None, variable_labels=None):
        """Build a Response from an array-like object.

        Args:
            samples (array_like/string):
                As for :func:`numpy.matrix`. See Notes.

            data_vectors (dict[str, array_like]):
                Misc data about the object

            todo

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

            todo

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
        """todo
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
        """NotImplemented"""
        # concurrent.futures.as_completed
        raise NotImplementedError

    def update(self, *other_responses):
        """todo"""

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
        return self.from_matrix(self.samples_matrix, self.data_vectors,
                                vartype=self.vartype, info=self.info,
                                variable_labels=self.variable_labels)

    @vartype_argument('vartype')
    def change_vartype(self, vartype, data_vector_offsets=None, inplace=True):
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
        """todo"""
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

            sorted_by:
                todo

            name (str/None, optional, default='Sample'):
                The name of the yielded namedtuples or None to yield regular tuples.

        Yields:
            namedtuple/tuple: The data in the response, in the order specified by the input
            'fields'.

        Examples:
            .. code-block:: python
                :linenos:

                samples = [{'a': 0, 'b': 1}, {'a': 1, 'b': 0}, {'a': 0, 'b': 0}]
                energies = [-1.0, 0.0, 1.0]
                response = dimod.Response.from_dicts(samples,
                                                     {'energy': energies},
                                                     vartype=dimod.BINARY)
                for datum in response.data():
                    print(datum)

            .. code-block:: python
                :linenos:

                samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1}, {'a': -1, 'b': -1}]
                energies = [-1.0, 0.0, 1.0]
                num_spin_up = [1, 1, 0]
                response = dimod.Response.from_dicts(samples,
                                                    {'energy': energies, 'num_spin_up': num_spin_up},
                                                    vartype=dimod.SPIN)
                for datum in response.data():
                    print(datum)

                for datum in response.data(['num_spin_up', 'energy']):
                    print(datum)

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

    def __str__(self):
        if self.label_mapping is None:
            return ' '.join(str(v) for v in self.samples[self.idx].tolist()[0])
        else:
            raise NotImplementedError

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
