"""TODO:
    - module level docstring
    - examples in Response
"""
from collections import namedtuple

import pandas as pd

from dimod.compatibility23 import iteritems
from dimod.vartypes import Vartype

try:
    import dwave_micro_client as microclient
except ImportError:  # pragma: no cover
    microclient = None

__all__ = ['Response']


class Response(object):
    """Encodes a response from a dimod sampler.

    Args:
        vartype (:class:`.Vartype`/str/set):
            The variable type desired for the response. Accepted input values:
            :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

    Attributes:
        vartype (:class:`.Vartype`): The variable type of the response.

    """

    def __init__(self, vartype):
        # the response object keeps two dataframes, these are kept index-linked so that they can
        # be joined
        self.df_samples = pd.DataFrame(dtype='int8')
        self.df_data = pd.DataFrame(columns=['energy'])

        try:    # pragma: no cover
            if isinstance(vartype, str):
                vartype = Vartype[vartype]
            else:
                vartype = Vartype(vartype)

            if not (vartype is Vartype.SPIN or vartype is Vartype.BINARY):
                raise ValueError  # this gets caught
        except (ValueError, KeyError):    # pragma: no cover
            raise TypeError(("expected input vartype to be one of: "
                             "Vartype.SPIN, 'SPIN', {-1, 1}, "
                             "Vartype.BINARY, 'BINARY', or {0, 1}."))
        self.vartype = vartype

        # we also store a list of futures that are awaiting read
        self._futures = []

    def __len__(self):
        """The number of samples."""
        return self.df_samples.shape[0]

    def __iter__(self):
        """Iterate over the samples as dicts from low energy to high."""
        return self.samples()

    def __str__(self):
        return self.merged(sorted_by_energy=False).__str__()  # pragma: no cover

    @property
    def df_samples(self):
        """:class:`pandas.DataFrame`: The samples. Should be treated as read-only."""
        # if there are any waiting futures, process them now
        if self._futures:
            self._add_samples_future()

        return self._samples

    @df_samples.setter
    def df_samples(self, df_samples):
        self._samples = df_samples

    @property
    def df_data(self):
        """:class:`pandas.DataFrame`: The data. Should be treated as read-only."""
        # if there are any waiting futures, process them now
        if self._futures:
            self._add_samples_future()

        return self._data

    @df_data.setter
    def df_data(self, df_data):
        self._data = df_data

    def merged(self, sorted_by_energy=True):
        """Returns the dataframe created by joining `.df_samples` and `.df_data`.

        Args:
            sorted_by_energy (bool, optional, default=True):
                Whether the samples should be returned in order of increasing energy or in the order
                they were added to the response.

        Returns:
            :class:`pandas.DataFrame`: The joined `.df_samples` and `.df_data`. If there are name
            conflicts, the data column will have '_data' prepended to it.

        """
        if sorted_by_energy:
            return self.df_samples.merge(self.df_data, how='right',
                                         left_index=True, right_index=True,
                                         suffixes=['', '_data'])
        else:
            return self.df_samples.merge(self.df_data, how='left',
                                         left_index=True, right_index=True,
                                         suffixes=['', '_data'])

    def done(self):
        """True if all of the futures added to the response have arrived."""
        return all(future.done() for future in self._futures)

    def add_sample(self, sample, energy, **kwargs):
        """Add a sample to the response.

        Args:
            sample (dict/:class:`pandas.Series`/list):
                A single sample as a dict or a pandas Series. If a dict, the keys should be the
                variables and the values are their value. If a Series or list, the index should be
                the variables and the data should be their value.

            energy (number):
                The energy of the given sample.

            **kwargs:
                Additional keywords will store additional data about the sample. See examples.

        Examples:
            >>> response = dimod.Response(dimod.SPIN)
            >>> response.add_sample({'a': -1, 'b': +1}, 1.)
            >>> print(response)
               a  b energy
            0 -1  1      1
            >>> response.add_sample({'a': +1, 'b': +1}, -1., num_spin_up=2)
            >>> print(response)
               a  b  energy  num_spin_up
            0 -1  1     1.0          NaN
            1  1  1    -1.0          2.0

            The sample can also be a :class:`pandas.Series` or a list.

            >>> response = dimod.Response(dimod.SPIN)
            >>> response.add_sample(pd.Series([-1, +1]), 1)
            >>> print(response)
               0  1 energy
            0  0  1      1
            >>> response.add_sample([+1, +1], -1)
            >>> print(response)
               0  1 energy
            0 -1  1      1
            1  1  1     -1

        See also:
            add_samples_from

        Notes:
            Very little input checking is performed in the interests of speed. It is up to the
            dimod sampler or composite that is populating the response to ensure correct
            variable labels and correct variable types.

        """
        self.add_samples_from([sample], [energy], **{key: [val] for key, val in iteritems(kwargs)})

    def add_samples_from(self, samples, energy, **kwargs):
        """Add a collection of samples to the response.

        Args:
            samples (list[dict]/:class:`pandas.DataFrame`/:class:`numpy.ndarray`/list[list]):
                A collection of samples.
                A single sample as a dict, row of a pandas DataFrame, row of numpy array or a list.
                If a dict, the keys should be the variables and the values are their value.
                If a row or list, the index should be the variables and the data should be their
                value.

            energy (iterable):
                An iterable of energies, one for each sample.

            **kwargs:
                Additional keywords will store additional data about the sample. See examples.

        Examples:
            >>> response = dimod.Response(dimod.BINARY)
            >>> samples = [{'a': 0, 'b': 1}, {'a': 1, 'b': 0}, {'a': 0, 'b': 0}]
            >>> energies = [1, 0, 0]
            >>> response.add_samples_from(samples, energies)
            >>> print(response)
               a  b energy
            0  0  1      1
            1  1  0      0
            2  0  0      0
            >>> samples_df = pd.DataFrame(samples)
            >>> response.add_samples_from(samples_df, energies)
               a  b energy
            0  0  1      1
            1  1  0      0
            2  0  0      0
            3  0  1      1
            4  1  0      0
            5  0  0      0

        See also:
            add_sample

        Notes:
            Very little input checking is performed in the interests of speed. It is up to the
            dimod sampler or composite that is populating the response to ensure correct
            variable labels and correct variable types.

        """
        # determine indices for the new data
        num_samples = self._samples.shape[0]
        new_indices = list(range(num_samples, num_samples + len(samples)))

        # create new samples dataframe from the given samples
        if isinstance(samples, pd.DataFrame):
            new_df_samples = samples.rename(index={idx: v for idx, v in enumerate(new_indices)})
        else:
            new_df_samples = pd.DataFrame(samples, index=new_indices, dtype='int8')

        # create the new data dataframe from energies and kwargs
        kwargs['energy'] = energy
        new_df_data = pd.DataFrame(kwargs, index=new_indices)

        # append the new dataframe to our existing samples, act on the actual objects, not their
        # getter versions
        self._samples = self._samples.append(new_df_samples)
        self._data = self._data.append(new_df_data)

        # we keep df_data sorted by energy
        self._data.sort_values('energy', inplace=True)

    def add_samples_future(self, future):
        """Add samples from a micro client Future.

        Args:
            future (:class:`dwave_micro_client.Future`):
                A Future from the dwave_micro_client.

        """
        self._futures.append(future)

    def _add_samples_future(self):
        """The main logic of add_samples_future. However, the samples only get loaded the first time
        the response is read, at which case this method is invoked.
        """
        futures = self._futures

        while futures:
            # wait for at least one future to be done
            microclient.Future.wait_multiple(futures, min_done=1)
            waiting = []

            for future in futures:
                if future.done():
                    # we have a response! add it to datalist
                    self._add_future(future)
                else:
                    waiting.append(future)

            futures = waiting

        self._futures = futures  # reset to 0

    def _add_future(self, future):
        """Add the samples from a single future. Note that future is expected to be done."""
        # construct a dataframe from the future
        samples = pd.DataFrame(future.samples, columns=future.solver.nodes, dtype='int8')

        self.add_samples_from(samples, future.energies, num_occurrences=future.occurrences)

    def samples(self, sample_type=dict, sorted_by_energy=True):
        """Iterate over the samples in the response.

        Args:
            sample_type (type, optional, default=dict):
                The requested type for the returned sample. Either dict or :class:`pd.Series`.

            sorted_by_energy (bool, optional, default=True):
                Whether the samples should be returned in order of increasing energy or in the order
                they were added to the response.

        Yields:
            A sample from the response. The type is determined by sample_type.

        Examples:
            >>> response = dimod.Response(dimod.BINARY)
            >>> samples = [{'a': 0, 'b': 1}, {'a': 1, 'b': 0}, {'a': 0, 'b': 0}]
            >>> energies = [-1, 0, 1]
            >>> list(response.samples())
            [{'a': 0, 'b': 1}, {'a': 1, 'b': 0}, {'a': 0, 'b': 0}]

        """
        variables = self.df_samples.columns
        if sample_type is dict:
            for idx, sample_row in self.merged(sorted_by_energy=sorted_by_energy).iterrows():
                yield sample_row.loc[variables].to_dict()
        elif sample_type is pd.Series:
            for idx, sample_row in self.merged(sorted_by_energy=sorted_by_energy).iterrows():
                yield sample_row.loc[variables]
        else:
            raise ValueError("sample_type should be dict or pandas.Series")  # pragma: no cover

    def data(self, fields=None, sample_type=dict, sorted_by_energy=True, name='Sample'):
        """Iterate over the data in the response.

        Args:
            fields (list, optional, default=None):
                If specified, the yielded tuples will only include the values in fields.
                A special field name 'sample' can be used to aggregate the samples.

            sample_type (type, optional, default=dict):
                If 'sample' is a requested field, specifies the requested type for the returned
                sample. Either dict or :class:`pd.Series`.

            sorted_by_energy (bool, optional, default=True):
                Whether the data should be returned in order of increasing energy or in the order
                it was added to the response.

            name (str/None, optional, default='Sample'):
                The name of the yielded namedtuples or None to yield regular tuples.

        Yields:
            namedtuple/tuple: The data in the response, in the order specified by the input
            'fields'.

        Examples:
            >>> response = dimod.Response(dimod.BINARY)
            >>> samples = [{'a': 0, 'b': 1}, {'a': 1, 'b': 0}, {'a': 0, 'b': 0}]
            >>> energies = [-1, 0, 1]
            >>> response.add_samples_from(samples, energies)
            >>> for datum in response.data():
            ...     print(datum)
            ...
            Sample(sample={'b': 1, 'a': 0}, energy=-1)
            Sample(sample={'b': 0, 'a': 1}, energy=0)
            Sample(sample={'b': 0, 'a': 0}, energy=1)

            >>> response = dimod.Response(dimod.BINARY)
            >>> response.add_sample({'a': +1, 'b': +1}, -1, num_spin_up=2)
            >>> for datum in response.data():
            ...     print(datum)
            ...
            Sample(sample={'a': 1, 'b': 1}, energy=-1, num_spin_up=2.0)

            >>> response = dimod.Response(dimod.BINARY)
            >>> response.add_sample({'a': +1, 'b': +1}, -1, num_spin_up=2)
            >>> for datum in response.data(['sample', 'num_spin_up'], name=None):
            ...     print(datum)
            ...
            ({'a': 1, 'b': 1}, 2)

        """

        if fields is None:
            fields = ['sample']
            fields.extend(self.df_data.columns)

        variables = self.df_samples.columns
        df_merged = self.merged(sorted_by_energy=sorted_by_energy)

        # first we handle the possible yield types
        if name is None:
            # yielding a tuple
            def _pack(values):
                return tuple([*values])
        else:
            # yielding a named tuple
            SampleTuple = namedtuple(name, fields)

            def _pack(values):
                return SampleTuple(*values)

        # we also create a function to parse the rows into the form we want.
        def _values(row):
            for field in fields:
                if field == 'sample':
                    # if 'sample' is requested, dump the full sample into the appropriate type
                    if sample_type is dict:
                        yield row[variables].to_dict()
                    elif sample_type is pd.Series:
                        yield row[variables]
                    else:  # pragma: no cover
                        raise ValueError("sample_type should be dict or pandas.Series")
                else:
                    # if not 'sample', just return the value as-is (this also works for variables)
                    yield row[field]

        # finally the main loop
        for idx, row in df_merged.iterrows():
            yield _pack(_values(row))

    def change_vartype(self, vartype, offset=0.0):
        """Change the response's vartype in-place.

        Args:
            vartype (:class:`.Vartype`/str/set):
                The variable type desired for the response. Accepted input values:
                :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            offset (float, optional, default=0.0):
                The constant offset that is added to the energies.

        """
        try:    # pragma: no cover
            if isinstance(vartype, str):
                vartype = Vartype[vartype]
            else:
                vartype = Vartype(vartype)

            if not (vartype is Vartype.SPIN or vartype is Vartype.BINARY):
                raise ValueError  # this gets caught
        except (ValueError, KeyError):    # pragma: no cover
            raise TypeError(("expected input vartype to be one of: "
                             "Vartype.SPIN, 'SPIN', {-1, 1}, "
                             "Vartype.BINARY, 'BINARY', or {0, 1}."))

        if vartype is self.vartype:
            # don't need to do anything
            return

        if vartype is Vartype.SPIN and self.vartype is Vartype.BINARY:
            self.df_samples = 2 * self.df_samples - 1
            self.vartype = vartype
        elif vartype is Vartype.BINARY and self.vartype is Vartype.SPIN:
            self.df_samples = (self.df_samples + 1) // 2
            self.vartype = vartype
        else:  # pragma: no cover
            raise ValueError("Cannot convert from {} to {}".format(self.vartype, vartype))

        if offset:
            self.df_data['energy'] += offset
