import sys
import itertools
import bisect

from dimod.decorators import ising, qubo
from dimod import ising_energy, qubo_energy

PY2 = sys.version_info[0] == 2
if PY2:
    range = xrange
    zip = itertools.izip
    iteritems = lambda d: d.iteritems()
    itervalues = lambda d: d.itervalues()
else:
    iteritems = lambda d: d.items()
    itervalues = lambda d: d.values()


class DiscreteModelResponse(object):
    """Serves as a superclass for response objects.

    Args:
        data (dict, optional): Data about the response as a whole
        as a dictionary. Default {}.

    Examples:
        >>> response = DiscreteModelResponse({'name', 'example'})
        >>> print(response.data)
        '{'name', 'example'}'

    """

    def __init__(self, data={}):
        self._samples = []
        self._energies = []
        self._sample_data = []
        self.data = data

    def __iter__(self):
        """Iterate over the samples. Use the expression 'for sample in
        response'.

        Returns:
            iterator: An iterator over all samples in the response,
            in order of increasing energy.

        Examples:
            >>> response = DiscreteModelResponse()
            >>> response.add_samples_from([{0: -1}, {0: 1}], [1, -1])
            >>> [s for s in response]
            [{0: 1}, {0: -1}]

        """
        return iter(self._samples)

    def samples(self, data=False):
        """Iterator over the samples.

        Args:
            data (bool, optional): If True, return an iterator
            over the the samples in a 2-tuple (sample, data).
            If False return an iterator over the samples.
            Default False.

        Returns:
            iterator: If data is False, return an iterator over
            all samples in response, in order of increasing energy.
            If data is True, return a 2-tuple (sample, data) in order
            of increasing sample energy.

        Examples:
            >>> response = DiscreteModelResponse()
            >>> response.add_sample({0: -1}, 1, data={'n': 5})
            >>> response.add_sample({0: 1}, -1, data={'n': 1})
            >>> list(response.samples())
            [{0: 1}, {0: -1}]
            >>> list(response.samples(data=True))
            [({0: 1}, {'n': 1}), ({0: -1}, {'n': 5})]

        """
        if data:
            # in PY2, we have overloaded zip with izip
            return zip(self._samples, self._sample_data)
        return iter(self._samples)

    def energies(self, data=False):
        """Iterator over the energies.

        Args:
            data (bool, optional): If True, return an iterator
            over the the energies in a 2-tuple (energy, data).
            If False return an iterator over the energies.
            Default False.

        Returns:
            iterator: If data is False, return an iterator over
            all energies in response, in increasing order.
            If data is True, return a 2-tuple (energy, data) in
            order of increasing energy.

        Examples:
            >>> response = DiscreteModelResponse()
            >>> response.add_sample({0: -1}, 1, data={'n': 5})
            >>> response.add_sample({0: 1}, -1, data={'n': 1})
            >>> list(response.energies())
            [-1, 1]
            >>> list(response.energies(data=True))
            [(-1, {'n': 1}), (1, {'n': 5})]

        """
        if data:
            # in PY2, we have overloaded zip with izip
            return zip(self._energies, self._sample_data)
        return iter(self._energies)

    def items(self, data=False):
        """Iterator over the samples and energies.

        Args:
            data (bool, optional): If True, return an iterator
            of 3-tuples (sample, energy, data). If False return
            an iterator of 2-tuples (sample, energy) over all of
            the samples and energies. Default False.

        Returns:
            iterator: If data is False, return an iterator of 2-tuples
            (sample, energy) over all samples and energies in response
            in order of increasing energy. If data is True, return an
            iterator of 3-tuples (sample, energy, data) in order of
            increasing energy.

        Examples:
            >>> response = DiscreteModelResponse()
            >>> response.add_sample({0: -1}, 1, data={'n': 5})
            >>> response.add_sample({0: 1}, -1, data={'n': 1})
            >>> list(response.items())
            [({0: 1}, -1), ({0: -1}, 1)]
            >>> list(response.items(data=True))
            [({0: 1}, -1, {'n': 1}), ({0: -1}, 1, {'n': 5})]

        """
        if data:
            return zip(self._samples, self._energies, self._sample_data)
        return zip(self._samples, self._energies)

    def add_sample(self, sample, energy, data={}):
        """Loads a sample and associated energy into the response.

        Args:
            sample (dict): A sample as would be returned by a discrete
            model solver. Should be a dict of the form
            {var: value, ...}.
            energy (float/int): The energy associated with the given
            sample.
            data (dict, optional): A dict containing any additional
            data about the sample. Default empty.

        Notes:
            Solutions are stored in order of energy, lowest first.

        Raises:
            TypeError: If `sample` is not a dict.
            TypeError: If `energy` is not an int or float.
            TypeError: If `data` is not a dict.

        Examples:
            >>> response = DiscreteModelResponse()
            >>> response.add_sample({0: -1}, 1)
            >>> response.add_sample({0: 1}, -1, data={'n': 1})

        """

        if not isinstance(sample, dict):
            raise TypeError("expected 'sample' to be a dict")
        if not isinstance(energy, (float, int)):
            raise TypeError("expected 'energy' to be numeric")
        if not isinstance(data, dict):
            raise TypeError("expected 'data' to be a dict")

        idx = bisect.bisect(self._energies, energy)
        self._samples.insert(idx, sample)
        self._energies.insert(idx, energy)
        self._sample_data.insert(idx, data)

    def add_samples_from(self, samples, energies, sample_data=None):
        """Loads samples and associated energies from iterators.

        Args:
            samples (iterator): An iterable object that yields
            samples. Each sample should be a dict of the form
            {var: value, ...}.
            energies (iterator): An iterable object that yields
            energies associated with each sample.
            sample_data (iterator, optional): An iterable object
            that yields data about each sample as  dict. Default
            empty dicts.

        Notes:
            Solutions are stored in order of energy, lowest first.

        Raises:
            TypeError: If any `sample` in `samples` is not a dict.
            TypeError: If any `energy`  in `energies` is not an int
            or float.
            TypeError: If any `data` in `sample_data` is not a dict.

        Examples:
            >>> samples = [{0: -1}, {0: 1}, {0: -1}]
            >>> energies = [1, -1, 1]
            >>> sample_data = [{'t': .2}, {'t': .5}, {'t': .1}]

            >>> response = DiscreteModelResponse()
            >>> response.add_samples_from(samples, energies)
            >>> list(response.samples())
            [{0: 1}, {0: -1}, {0: -1}]

            >>> response = DiscreteModelResponse()
            >>> response.add_samples_from(samples, energies, sample_data)
            >>> list(response.samples())
            [{0: 1}, {0: -1}, {0: -1}]

            >>> items = [({0: -1}, -1), ({0: -1}, 1)]
            >>> response = DiscreteModelResponse()
            >>> response.add_samples_from(*zip(*items))
            >>> list(response.samples())
            [{0: 1}, {0: -1}]

        """

        if sample_data is None:
            # if no sample data is provided, we want to yield a unique dict
            # for each sample added to the system
            def _sample_data():
                while True:
                    yield {}  # faster than dict()
            sample_data = _sample_data()

        # load them into self
        for sample, energy, data in zip(samples, energies, sample_data):
            self.add_sample(sample, energy, data)

    def __str__(self):
        """Return a string representation of the response.

        Returns:
            str: A string representation of the graph.

        """

        lines = [self.__repr__(), 'data: {}'.format(self.data)]

        item_n = 0
        total_n = len(self)
        for sample, energy, data in self.items(data=True):
            if item_n > 9 and item_n < total_n - 1:
                if item_n == 10:
                    lines.append('...')
                item_n += 1
                continue

            lines.append('Item {}:'.format(item_n))
            lines.append('  sample: {}'.format(sample))
            lines.append('  energy: {}'.format(energy))
            lines.append('  data: {}'.format(data))

            item_n += 1

        return '\n'.join(lines)

    def __getitem__(self, sample):
        """Get the energy for the given sample.

        Args:
            sample (dict): A sample in response.

        Return:
            float/int: The energy associated with sample.

        Raises:
            KeyError: If the sample is not in response.

        Notes:
            dicts are matched by contents, not by reference.

        """
        try:
            idx = self._samples.index(sample)
        except ValueError as e:
            raise KeyError(e.message)

        return self._energies[idx]

    def __len__(self):
        """The number of samples in response."""
        return self._samples.__len__()

    def relabel_samples(self, mapping, copy=True):
        """Relabels the variable in the samples.

        Args:
            mapping (dict): A dictionary with the old labels as keys
            and the new labels as values. A partial mapping is
            allowed.
            copy (optional): If True, return a copy or if False
            relabel the samples in place.

        Examples:
            >>> response = DiscreteModelResponse()
            >>> response.add_sample({'a': -1, 'b': 1}, 1)
            >>> response.add_sample({'a': 1, 'b': -1}, -1)
            >>> mapping = {'a': 1, 'b': 0}

            >>> new_response = response.relabel_samples(mapping)
            >>> list(new_response.samples())
            [{0: -1, 1: 1}, {0: 1, 1: -1}]

            >>> response.relabel_samples(mapping, copy=False)
            >>> list(response.samples())
            [{0: -1, 1: 1}, {0: 1, 1: -1}]

        """

        try:
            if copy:
                return _relabel_copy(self, mapping)
            else:
                return _relabel_inplace(self, mapping)
        except MappingError:
            raise ValueError('given mapping does not have unique values.')


class MappingError(Exception):
    """mapping causes conflicting values in samples"""


def _relabel_copy(response, mapping):
    # make a a new response of the same class
    rl_response = response.__class__()

    # copy over the data
    rl_response.data = response.data

    # for each sample, energy, data in self, relabel the sample
    # and add to the new response. Missing labels are kept the
    # same.
    for sample, energy, data in response.items(data=True):
        rl_sample = {}
        for v, val in iteritems(sample):
            if v in mapping:
                new_v = mapping[v]
                if new_v in rl_sample:
                    raise MappingError
                rl_sample[mapping[v]] = val
            if v not in mapping:
                rl_sample[v] = val
        rl_response.add_sample(rl_sample, energy, data)

    # return the new object
    return rl_response


def _relabel_inplace(response, mapping):
    response = _relabel_copy(response, mapping)
    return


class BinaryResponse(DiscreteModelResponse):
    def add_sample(self, sample, energy=None, data={}, Q=None):
        """Loads a sample and associated energy into the response.

        Args:
            sample (dict): A sample as would be returned by a discrete
            model solver. Should be a dict of the form
            {var: value, ...}. The values should be spin-valued, that is
            -1 or 1.
            energy (float/int, optional): The energy associated with the
            given sample.
            data (dict, optional): A dict containing any additional
            data about the sample. Default empty.
            Q (dict): Defines a Quadratic Unconstrained Binary
            Optimization problem that can be used to calculate the energy
            associated with `sample`.

        Notes:
            Solutions are stored in order of energy, lowest first.

        Raises:
            TypeError: If `sample` is not a dict.
            TypeError: If `energy` is not an int or float.
            TypeError: If `data` is not a dict.
            ValueError: If any of the values in `sample` are not -1
            or 1.
            TypeError: If energy is not provided, Q must be.

        Examples:
            >>> sample = {0: 1, 1: 0, 2: 0}
            >>> energy = -1
            >>> response = BinaryResponse()
            >>> response.add_sample(sample, energy)
            >>> list(response.samples())
            [{0: 1, 1: 0, 2: 0}]
            >>> response.add_sample(sample, Q={(0, 0): -1, (1, 1): 0, (2, 2): 0})
            >>> list(response.items())
            [({0: 1, 1: 0, 2: 0}, -1), ({0: 1, 1: 0, 2: 0}, -1)]
            >>> response.add_sample(sample, data={'n': 10},
                                    Q={(0, 0): -1, (1, 1): 0, (2, 2): 0})

        """
        # check that the sample is sp]n-valued
        if any(val not in (0, 1) for val in itervalues(sample)):
            raise ValueError('given sample is not binary. Values should be 0 or 1')

        # if energy is not provided, but Q is, then we can calculate
        # the energy for the sample.
        if energy is None:
            if Q is None:
                raise TypeError("most provide 'energy' or 'Q'")
            energy = qubo_energy(Q, sample)

        DiscreteModelResponse.add_sample(self, sample, energy, data)

    def add_samples_from(self, samples, energies=None, sample_data=None, Q=None):
        """Loads samples and associated energies from iterators.

        Args:
            samples (iterator): An iterable object that yields
            samples. Each sample should be a dict of the form
            {var: value, ...}.
            energies (iterator): An iterable object that yields
            energies associated with each sample.
            sample_data (iterator, optional): An iterable object
            that yields data about each sample as  dict. Default
            empty dicts.
            Q (dict): Defines a Quadratic Unconstrained Binary
            Optimization problem that can be used to calculate the energy
            associated with `sample`.

        Notes:
            Solutions are stored in order of energy, lowest first.

        Raises:
            TypeError: If any `sample` in `samples` is not a dict.
            TypeError: If any `energy`  in `energies` is not an int
            or float.
            TypeError: If any `data` in `sample_data` is not a dict.
            ValueError: If any of the values in `sample` are not 0
            or 1.
            TypeError: If energy is not provided, Q must be.

        Examples:
            >>> samples = [{0: 0}, {0: 1}, {0: 0}]
            >>> energies = [1, -1, 1]
            >>> sample_data = [{'t': .2}, {'t': .5}, {'t': .1}]

            >>> response = BinaryResponse()
            >>> response.add_samples_from(samples, energies)
            >>> list(response.samples())
            [{0: 1}, {0: 0}, {0: 0}]

            >>> response = BinaryResponse()
            >>> response.add_samples_from(samples, energies, sample_data)
            >>> list(response.samples())
            [{0: 1}, {0: 0}, {0: 0}]

            >>> items = [({0: -1}, -1), ({0: -1}, 1)]
            >>> response = BinaryResponse()
            >>> response.add_samples_from(*zip(*items))
            >>> list(response.samples())
            [{0: 1}, {0: 0}]

            >>> response = BinaryResponse()
            >>> response.add_samples_from(samples, h={0: -1}, J={}})
            >>> list(response.energies())
            [-1, 1, 1]

        """

        if energies is None:
            energies = itertools.repeat(None)

        if sample_data is None:
            # if no sample data is provided, we want to yield a unique dict
            # for each sample added to the system
            def _sample_data():
                while True:
                    yield {}  # faster than dict()
            sample_data = _sample_data()

        for sample, energy, sample_data in zip(samples, energies, sample_data):
            self.add_sample(sample, energy, sample_data, Q=Q)

    def as_spin(self, offset):
        """Converts a BinaryResponse to a SpinResponse.

        Args:
            offset (float/int, optional): The energy offset as would
            be returned by `ising_to_qubo`. The energy offset is
            applied to each energy in the response.

        Returns:
            SpinResponse: A SpinResponse with the samples converted
            from spin to binary, the energies updated with `offset` and
            all of the data transferred directly.

        Notes:
            Only information stored in `data` property and as would be
            returned by `samples(data=True)` is transferred.

        """

        spin_response = SpinResponse()
        spin_response.data = self.data.copy()

        for sample, energy, data in self.items(data=True):
            spin_sample = {v: 2 * val - 1 for v, val in iteritems(sample)}
            spin_response.add_sample(spin_sample, energy + offset, data)

        return spin_response


class SpinResponse(DiscreteModelResponse):
    def add_sample(self, sample, energy=None, data={}, h=None, J=None):
        """Loads a sample and associated energy into the response.

        Args:
            sample (dict): A sample as would be returned by a discrete
            model solver. Should be a dict of the form
            {var: value, ...}. The values should be spin-valued, that is
            -1 or 1.
            energy (float/int, optional): The energy associated with the
            given sample.
            data (dict, optional): A dict containing any additional
            data about the sample. Default empty.
            h (dict) and J (dict): Define an Ising problem that can be
            used to calculate the energy associated with `sample`.

        Notes:
            Solutions are stored in order of energy, lowest first.

        Raises:
            TypeError: If `sample` is not a dict.
            TypeError: If `energy` is not an int or float.
            TypeError: If `data` is not a dict.
            ValueError: If any of the values in `sample` are not -1
            or 1.
            TypeError: If energy is not provided, h and J must be.

        Examples:
            >>> response = SpinResponse()
            >>> response.add_sample({0: -1}, 1)
            >>> response.add_sample({0: 1}, -1, data={'n': 1})
            >>> response.add_sample({0: 1}, h={0: -1}, J={})
            >>> list(response.energies())
            [-1, -1]

        """
        # check that the sample is sp]n-valued
        if any(val not in (-1, 1) for val in itervalues(sample)):
            raise ValueError('given sample is not spin-valued. Values should be -1 or 1')

        # if energy is not provided, but h, J are, then we can calculate
        # the energy for the sample.
        if energy is None:
            if h is None or J is None:
                raise TypeError("most provide 'energy' or 'h' and 'J'")
            energy = ising_energy(h, J, sample)

        DiscreteModelResponse.add_sample(self, sample, energy, data)

    def add_samples_from(self, samples, energies=None, sample_data=None, h=None, J=None):
        """Loads samples and associated energies from iterators.

        Args:
            samples (iterator): An iterable object that yields
            samples. Each sample should be a dict of the form
            {var: value, ...}.
            energies (iterator): An iterable object that yields
            energies associated with each sample.
            sample_data (iterator, optional): An iterable object
            that yields data about each sample as  dict. Default
            empty dicts.
            h (dict) and J (dict): Define an Ising problem that can be
            used to calculate the energy associated with `sample`.

        Notes:
            Solutions are stored in order of energy, lowest first.

        Raises:
            TypeError: If any `sample` in `samples` is not a dict.
            TypeError: If any `energy`  in `energies` is not an int
            or float.
            TypeError: If any `data` in `sample_data` is not a dict.
            ValueError: If any of the values in `sample` are not -1
            or 1.
            TypeError: If energy is not provided, h and J must be.

        Examples:
            >>> samples = [{0: -1}, {0: 1}, {0: -1}]
            >>> energies = [1, -1, 1]
            >>> sample_data = [{'t': .2}, {'t': .5}, {'t': .1}]

            >>> response = SpinResponse()
            >>> response.add_samples_from(samples, energies)
            >>> list(response.samples())
            [{0: 1}, {0: -1}, {0: -1}]

            >>> response = SpinResponse()
            >>> response.add_samples_from(samples, energies, sample_data)
            >>> list(response.samples())
            [{0: 1}, {0: -1}, {0: -1}]

            >>> items = [({0: -1}, -1), ({0: -1}, 1)]
            >>> response = SpinResponse()
            >>> response.add_samples_from(*zip(*items))
            >>> list(response.samples())
            [{0: 1}, {0: -1}]

            >>> response = SpinResponse()
            >>> response.add_samples_from(samples, h={0: -1}, J={}})
            >>> list(response.energies())
            [-1, 1, 1]

        """

        if energies is None:
            energies = itertools.repeat(None)

        if sample_data is None:
            # if no sample data is provided, we want to yield a unique dict
            # for each sample added to the system
            def _sample_data():
                while True:
                    yield {}  # faster than dict()
            sample_data = _sample_data()

        for sample, energy, sample_data in zip(samples, energies, sample_data):
            self.add_sample(sample, energy, sample_data, h=h, J=J)

    def as_binary(self, offset=0):
        """Converts a SpinResponse to a BinaryResponse.

        Args:
            offset (float/int, optional): The energy offset as would
            be returned by `ising_to_qubo`. The energy offset is
            applied to each energy in the response.

        Returns:
            BinaryResponse: A BinaryResponse with the samples converted
            from spin to binary, the energies updated with `offset` and
            all of the data transferred directly.

        Notes:
            Only information stored in `data` property and as would be
            returned by `samples(data=True)` is transferred.

        """

        bin_response = BinaryResponse()
        bin_response.data = self.data.copy()

        for sample, energy, data in self.items(data=True):
            bin_sample = {v: (val + 1) / 2 for v, val in iteritems(sample)}
            bin_response.add_sample(bin_sample, energy + offset, data)

        return bin_response
