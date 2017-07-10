import sys
import itertools
import bisect

from dimodsampler.decorators import ising, qubo
from dimodsampler import ising_energy, qubo_energy

# Python 2/3 compatibility
if sys.version_info[0] == 2:
    range = xrange
    zip = itertools.izip


class DiscreteModelResponse(object):
    def __init__(self):
        """Constructor. See __doc__ for DiscreteModelResponse"""
        self._samples = []
        self._energies = []
        self._sample_data = []
        self.data = {}

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
                    yield {}
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


class BinaryResponse(DiscreteModelResponse):

    # @solve_qubo_aspi(3)
    def add_sample(self, sample, energy=None, Q={}):
        """Adds a single sample to the response.

        Args:
            sample (dict): A single Discrete Model sample of the
            form {var: b, ...} where `var` is any hashable object and
            b is either 0 or 1.
            energy (float/int, optional): The energy indiced by each
            sample. Default is NaN.
            Q (dict): Dictionary of QUBO coefficients. Takes the form
            {(var0, var1): coeff, ...}. If not provided, the given
            energy is used. If energy is not provided, but Q is then
            the energy is calculated from Q. If both are provided then
            the energy is checked against Q.

        """

        # if no Q provided, just use the inherited method
        if Q is None:
            DiscreteModelResponse.add_sample(self, sample, energy)
            return

        # ok, so we have a Q to play with. So let's calculate the induced energy
        # from Q.
        calculated_energy = 0
        for v0, v1 in Q:
            calculated_energy += sample[v0] * sample[v1] * Q[(v0, v1)]

        # if both Q and energy were provided, let's check that they are equal
        if not math.isnan(energy) and energy != calculated_energy:
            raise ValueError("given energy ({}) and energy induced by Q ({}) do not agree"
                             .format(energy, calculated_energy))

        # finally add the sample
        DiscreteModelResponse.add_sample(self, sample, calculated_energy)

    # @solve_qubo_api(3)
    def add_samples_from(self, samples, energies=None, Q={}):
        if Q is None and not energies:
            raise TypeError("Either 'energies' or 'Q' must be provided")

        if Q is not None:
            calculated_energies = (qubo_energy(Q, soln) for soln in samples)

        raise NotImplementedError

    def as_spins(self, offset):
        raise NotImplementedError


class SpinResponse(DiscreteModelResponse):

    # @solve_ising_api(3, 4)
    def add_sample(self, sample, energy=None, h={}, J={}):
        if any(spin not in (-1, 1) for spin in sample.values()):
            raise ValueError("sample values must be spin (-1 or 1)")

        # the base case
        if not h and not J:
            DiscreteModelResponse.add_sample(self, sample, energy)
            return

        # if h, J are provided, we can use them to determine/check the energy
        if not h:
            raise TypeError("input 'h' defined but not 'J'")
        if not J:
            raise TypeError("input 'J' defined but not 'h'")

        # now calculate the energy
        energy = 0

        # first the linear biases
        for var in h:
            energy += sample[var] * h[var]

        # then the quadratic baises
        for var0, var1 in J:
            energy += sample[var0] * sample[var1] * J[(var0, var1)]

        # finally add the sample
        DiscreteModelResponse.add_sample(self, sample, energy)

    def add_samples_from(self, samples, energies=None, h=None, J=None):
        raise NotImplementedError

    def as_binary(self, offset, copy=True):

        b_response = BinaryResponse()

        # create iterators over the stored data
        binary_samples = ({v: (sample[v] + 1) / 2 for v in sample}
                            for sample in self.samples_iter())
        binary_energies = (energy + offset for energy in self.energies_iter())

        b_response.add_samples_from(binary_samples, binary_energies)

        b_response = self.data

        if copy:
            return b_response
            return

        self = b_response
