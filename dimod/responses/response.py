"""
.. _responses:

Responses
=========

The :class:`BinaryResponse` and :class:`SpinResponse` are the main response
types for dimod samplers. Both are subclasses of the :class:`.TemplateResponse`.

Response Classes
----------------
"""

import itertools

from dimod import _PY2
from dimod.template_response import TemplateResponse
from dimod.utilities import qubo_energy, ising_energy

__all__ = ['BinaryResponse', 'SpinResponse']

if _PY2:
    iteritems = lambda d: d.iteritems()
    itervalues = lambda d: d.itervalues()
else:
    iteritems = lambda d: d.items()
    itervalues = lambda d: d.values()


class BinaryResponse(TemplateResponse):
    """Response object that encodes binary samples.

    Args:
        data (dict, optional): Data about the response as a whole
            as a dictionary. Default {}.

    """
    def add_sample(self, sample, energy=None, data=None, Q=None):
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
        if not isinstance(sample, dict):
            raise TypeError("expected 'sample' to be a dict")

        # check that the sample is sp]n-valued
        if any(val not in (0, 1) for val in itervalues(sample)):
            raise ValueError('given sample is not binary. Values should be 0 or 1')

        # if energy is not provided, but Q is, then we can calculate
        # the energy for the sample.
        if energy is None:
            if Q is None:
                raise TypeError("most provide 'energy' or 'Q'")
            energy = qubo_energy(Q, sample)

        if data is None:
            data = {}

        TemplateResponse.add_sample(self, sample, energy, data)

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

        samples = list(samples)

        if not all(isinstance(sample, dict) for sample in samples):
            raise TypeError("expected each sample in 'samples' to be a dict")

        if any(any(val not in (0, 1) for val in itervalues(sample)) for sample in samples):
            raise ValueError('given samples are not binary. Values should be 0 or 1')

        if energies is None:
            if Q is None:
                raise TypeError("most provide 'energy' or 'Q'")
            energies = [qubo_energy(Q, sample) for sample in samples]

        TemplateResponse.add_samples_from(self, samples, energies, sample_data)

    def as_spin(self, offset=0.0, data_copy=False):
        """Converts a BinaryResponse to a SpinResponse.

        Args:
            offset (float/int, optional): The energy offset as would
                be returned by `ising_to_qubo`. The energy offset is
                applied to each energy in the response.
            data_copy (bool, optional): Whether to create a copy
                of each data dict. Default False.

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

        # the energies are offset by a constant, so the order stays the same. Thus we can
        # transfer directly.
        spin_response._samples = [{v: 2 * val - 1 for v, val in iteritems(sample)}
                                  for sample in self.samples()]
        spin_response._energies = [energy + offset for energy in self.energies()]

        # whether we copy each data, or simply pass the same variable
        if data_copy:
            spin_response._sample_data = [data.copy() for __, data in self.samples(data=True)]
        else:
            spin_response._sample_data = [data for __, data in self.samples(data=True)]

        return spin_response


class SpinResponse(TemplateResponse):
    """Response object that encodes spin-valued samples.

    Args:
        data (dict, optional): Data about the response as a whole
            as a dictionary. Default {}.

    """

    def add_sample(self, sample, energy=None, data=None, h=None, J=None):
        """Loads a sample and associated energy into the response.

        Args:
            sample (dict): A sample as would be returned by a discrete
                model solver. Should be a dict of the form
                {var: value, ...}. The values should be spin-valued, that is
                -1 or 1.
            energy (float/int, optional): The energy associated with the
                given sample.
            data (dict, optional): A dict containing any additional
                data about the sample. Default {}.
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
        if not isinstance(sample, dict):
            raise TypeError("expected 'sample' to be a dict")

        # check that the sample is spin-valued
        if any(val not in (-1, 1) for val in itervalues(sample)):
            raise ValueError('given sample is not spin-valued. Values should be -1 or 1')

        # if energy is not provided, but h, J are, then we can calculate
        # the energy for the sample.
        if energy is None:
            if h is None or J is None:
                raise TypeError("most provide 'energy' or 'h' and 'J'")
            energy = ising_energy(h, J, sample)

        if data is None:
            data = {}

        TemplateResponse.add_sample(self, sample, energy, data)

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

        samples = list(samples)

        if not all(isinstance(sample, dict) for sample in samples):
            raise TypeError("expected each sample in 'samples' to be a dict")

        if any(any(val not in (-1, 1) for val in itervalues(sample)) for sample in samples):
            raise ValueError('given sample is not spin-valued. Values should be -1 or 1')

        if energies is None:
            if h is None or J is None:
                raise TypeError("most provide 'energy' or 'h' and 'J'")
            energies = [ising_energy(h, J, sample) for sample in samples]

        TemplateResponse.add_samples_from(self, samples, energies, sample_data)

    def as_binary(self, offset=0.0, data_copy=False):
        """Converts a SpinResponse to a BinaryResponse.

        Args:
            offset (float/int, optional): The energy offset as would
                be returned by `ising_to_qubo`. The energy offset is
                applied to each energy in the response.
            data_copy (bool, optional): Whether to create a copy
                of each data dict. Default False.

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

        bin_response._samples = [{v: (val + 1) // 2 for v, val in iteritems(sample)}
                                 for sample in self.samples()]
        bin_response._energies = [energy + offset for energy in self.energies()]

        if data_copy:
            bin_response._sample_data = [data.copy() for __, data in self.samples(data=True)]
        else:
            bin_response._sample_data = [data for __, data in self.samples(data=True)]

        return bin_response
