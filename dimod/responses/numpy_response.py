"""
.. _numpy_responses:

Numpy Responses
===============

While dimod does not require `numpy <http://www.numpy.org/>`_, in practice many
implemented samplers use
numpy to speed up their calculation. These samplers can use :class:`.NumpyResponse`,
:class:`.NumpySpinResponse`, or :class:`.NumpyBinaryResponse` response types.

The numpy response types have all of the same methods and behaviors as
:class:`.SpinResponse` and :class:`.BinaryResponse`, but also include
methods that can access numpy arrays directly.

Numpy Response Classes
----------------------
"""
import itertools

from dimod import _PY2
from dimod.responses.response import BinaryResponse, SpinResponse, TemplateResponse

__all__ = ['NumpyResponse', 'NumpySpinResponse', 'NumpyBinaryResponse']


if _PY2:
    range = xrange
    zip = itertools.izip


class NumpyResponse(TemplateResponse):
    """Serves as a superclass for numpy response objects.

    Differs from the TemplateResponse by storing samples and energies
    internally in a numpy ndarray.

    All of the access and construction methods work the same as for
    TemplateResponse derived responses.

    Args:
        data (dict, optional): Data about the response as a whole
            as a dictionary. Default {}.

    """
    def __init__(self, data=None):
        # NumpyResponse stores the samples in a 2d int array, energies in a 1d float array
        # and the sample_data in a list of dicts
        self._samples = None
        self._energies = None
        self._sample_data = []

        if data is None:
            self.data = {}
        elif not isinstance(data, dict):
            raise TypeError('expected input "data" to be None or a dict')
        else:
            self.data = data

    def samples(self, data=False):
        """Iterator over the samples.

        Args:
            data (bool, optional): If True, return an iterator
                over the the samples in a 2-tuple `(sample, data)`.
                If False return an iterator over the samples.
                Default False.

        Returns:
            iterator: If data is False, return an iterator over
            all samples in response, in order of increasing energy.
            If data is True, return a 2-tuple (sample, data) in order
            of increasing sample energy.

        """
        if data:
            return zip(self.samples(), self._sample_data)

        if self._samples is None:
            return iter([])

        return iter({idx: val for idx, val in enumerate(row)} for row in self._samples)

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

        """
        if data:
            return zip(self.energies(), self._sample_data)

        if self._samples is None:
            return iter([])

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

        """
        if data:
            return zip(self.samples(), self.energies(), self._sample_data)

        if self._samples is None:
            return iter([])

        return zip(self.samples(), self.energies())

    def add_sample(self, sample, energy, data=None):
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

        Note:
            For NumpyResponse objects, it is more efficient to use
            `add_samples_from_array` method.

        """
        import numpy as np

        if not isinstance(sample, dict):
            raise TypeError(("expected 'sample' to be a dict, to add samples from an"
                             " ndarray use add_samples_from_array"))
        elif not all(idx in sample for idx in range(len(sample))):
            raise ValueError("all variables in 'sample' must be integer labeled")

        if data is None:
            data = {}
        elif not isinstance(data, dict):
            raise TypeError("expected input 'data' to be a dict or None.")

        sample = np.asarray([[sample[idx] for idx in range(len(sample))]], dtype=int)
        energy = np.array([energy], dtype=float)
        self.add_samples_from_array(sample, energy, [data])

    def add_samples_from(self, samples, energies, sample_data=None):
        """Loads samples and associated energies from iterators.

        Args:
            samples (iterator): An iterable object that yields
                samples. Each sample should be a dict of the form
                {var: value, ...}.
            energies (iterator): An iterable object that yields
                energies associated with each sample.
            sample_data (iterator, optional): An iterable object
                that yields data about each sample as  dict. If
                None, then each data will be an empty dict. Default
                None.

        Notes:
            Solutions are stored in order of energy, lowest first.

        Note:
            For NumpyResponse objects, it is more efficient to use
            `add_samples_from_array` method.

        """
        import numpy as np

        samples = list(samples)
        energies = list(energies)

        if not all(isinstance(sample, dict) for sample in samples):
            raise TypeError(("expected each sample in 'samples' to be a dict, "
                             "to add samples from an ndarray use add_samples_from_array"))
        elif not all(all(idx in sample for idx in range(len(sample))) for sample in samples):
            raise ValueError("all variables in 'sample' must be integer labeled")

        if sample_data is None:
            sample_data = [{} for __ in range(len(samples))]
        else:
            sample_data = list(sample_data)
            if not all(isinstance(data, dict) for data in sample_data):
                raise TypeError("expected input 'data' to be a dict or None.")

        sample = np.asarray([[sample[idx] for idx in range(len(sample))] for sample in samples],
                            dtype=int)
        energy = np.asarray(energies, dtype=float)
        self.add_samples_from_array(sample, energy, sample_data)

    def __len__(self):
        """The number of samples in response."""
        if self._samples is None:
            return 0
        num_samples, __ = self._samples.shape
        return num_samples

    def relabel_samples(self, mapping, copy=True):
        raise NotImplementedError("NumpyResponse does not support arbitrarily labeled variables")

    def add_samples_from_array(self, samples, energies, sample_data=None, sorted_by_energy=False):
        """Loads samples and associated energies from numpy arrays.

        Args:
            samples (:obj:`numpy.ndarray`): An two dimensional numpy array
                in which each row is a sample.
            energies (:obj:`numpy.ndarray`): A one dimensional numpy array
                in which each value is an energy. Must be the same length
                as `samples`.
            sample_data (iterator, optional): An iterable object
                that yields data about each sample as  dict. If
                None, then each data will be an empty dict. Default
                None.
            sorted_by_energy (bool): If True, then the user asserts that
                `samples` and `energies` are sorted by energy from low to
                high. This is not checked.

        Notes:
            Solutions are stored in order of energy, lowest first.

        """
        import numpy as np

        # check samples
        if not isinstance(samples, np.ndarray):
            raise TypeError("expected 'samples' to be a two dimensional ndarray")
        if samples.ndim != 2:
            raise ValueError("expected 'samples' to be a two dimensional ndarray")

        if not isinstance(energies, np.ndarray):
            raise TypeError("expected 'energies' to be a two dimensional ndarray")
        if energies.ndim != 1:
            raise ValueError("expected 'energies' to be a two dimensional ndarray")

        # assumes dimension 2
        num_samples, num_variables = samples.shape

        if len(energies) != num_samples:
            raise ValueError('length of energies does not match number of samples')

        if sample_data is None:
            sample_data = [{} for __ in range(num_samples)]
        else:
            sample_data = list(sample_data)
            if len(sample_data) != num_samples:
                raise ValueError('length of data and does not match number of samples')

        if sorted_by_energy and self._samples is None:
            # samples are sorted by energy and response is empty
            self._samples = samples
            self._energies = energies
            self._sample_data = sample_data
        else:

            if self._samples is not None:
                samples = np.concatenate((self._samples, samples), axis=0)
                energies = np.concatenate((self._energies, energies))

            sample_data = self._sample_data + sample_data

            idxs = np.argsort(energies)

            self._samples = samples[idxs, :]
            self._energies = energies[idxs]
            self._sample_data = [sample_data[i] for i in idxs]

    def samples_array(self):
        """Returns the :obj:`numpy.ndarray` containing the samples."""
        import numpy as np

        samples = self._samples
        if samples is None:
            return np.empty((0, 0), dtype=int)
        else:
            return samples

    def energies_array(self):
        """Returns the :obj:`numpy.ndarray` containing the energies."""
        import numpy as np

        energies = self._energies
        if energies is None:
            return np.empty((0,), dtype=float)
        else:
            return energies


class NumpySpinResponse(NumpyResponse):
    """Subclass of :class:`.NumpyResponse` which encodes spin-valued samples.

    Differs from the :class:`.SpinResponse` by storing samples and energies
    internally in a :obj:`numpy.ndarray`.

    Args:
        data (dict, optional): Data about the response as a whole
            as a dictionary. Default {}.

    Examples:
        >>> import numpy as np
        >>> response = NumpySpinResponse()
        >>> samples = np.asarray([[-1, -1, 1], [1, 1, -1]], dtype=int)
        >>> energies = np.asarray([2., -2.], dtype=float)
        >>> response.add_samples_from_array(samples, energies)
        >>> response.samples_array()
        array([[1, 1, -1], [-1, -1, 1]])
        >>> for sample in response:
        ...     # still works like a normal response object
        ...     pass

    """
    def __init__(self, data=None):
        NumpyResponse.__init__(self, data)

    def add_samples_from_array(self, samples, energies, sample_data=None, sorted_by_energy=False):
        """Loads samples and associated energies from spin-valued numpy arrays.

        Args:
            samples (:obj:`numpy.ndarray`): An two dimensional numpy array
                in which each row is a sample. Values should be -1 or 1.
            energies (:obj:`numpy.ndarray`): A one dimensional numpy array
                in which each value is an energy. Must be the same length
                as `samples`.
            sample_data (iterator, optional): An iterable object
                that yields data about each sample as  dict. If
                None, then each data will be an empty dict. Default
                None.
            sorted_by_energy (bool): If True, then the user asserts that
                `samples` and `energies` are sorted by energy from low to
                high. This is not checked.

        Raises:
            ValueError: If any values in the samples are not -1 or 1.

        Notes:
            Solutions are stored in order of energy, lowest first.

        """
        if any(s not in {-1, 1} for s in samples.flat):
            raise ValueError("All values in samples should be -1 or 1")

        NumpyResponse.add_samples_from_array(self, samples, energies,
                                             sample_data, sorted_by_energy)

    def as_binary(self, offset=0.0, data_copy=False):
        """Returns the :class:`.NumpyBinaryResponse` version of itself.

        Args:
            offset (float/int, optional): The energy offset as would
                be returned by `ising_to_qubo`. The energy offset is
                applied to each energy in the response.
            data_copy (bool, optional): Whether to create a copy
                of each data dict. Default False.

        Returns:
            NumpyBinaryResponse: A BinaryResponse with the samples converted
            from spin to binary, the energies updated with `offset` and
            all of the data transferred directly.

        Notes:
            Only information stored in `data` property and as would be
            returned by `samples(data=True)` is transferred.


        """
        binary_response = NumpyBinaryResponse()

        binary_response._samples = (self._samples + 1) // 2
        binary_response._energies = self._energies + offset

        if data_copy:
            binary_response.data = self.data.copy()
            binary_response._sample_data = [data.copy() for data in self._sample_data]
        else:
            binary_response.data = self.data
            binary_response._sample_data = [data for data in self._sample_data]

        return binary_response

    def as_spin_response(self, data_copy=False):
        """Returns the :class:`.SpinResponse` version of itself.

        Args:
            data_copy (bool, optional): Whether to create a copy
                of each data dict. Default False.

        Returns:
            SpinResponse: A SpinResponse with the same values as the
                SpinNumpyResponse

        Notes:
            Only information stored in `data` property and as would be
            returned by `samples(data=True)` is transferred.


        """
        if data_copy:
            response = SpinResponse(self.data.copy())
            sample_data = [data.copy() for data in self._sample_data]
        else:
            response = SpinResponse(self.data)
            sample_data = list(self._sample_data)

        response.add_samples_from(self.samples(), self.energies(), sample_data)

        return response


class NumpyBinaryResponse(NumpyResponse):
    """Subclass of :class:`.NumpyResponse` which encodes binary-valued samples.

    Differs from the :class:`.BinaryResponse` by storing samples and energies
    internally in a :obj:`numpy.ndarray`.

    Args:
        data (dict, optional): Data about the response as a whole
            as a dictionary. Default {}.

    Examples:
        >>> import numpy as np
        >>> response = NumpyBinaryResponse()
        >>> samples = np.asarray([[0, 0, 0], [1, 1, 1]], dtype=int)
        >>> energies = np.asarray([0., -3.], dtype=float)
        >>> response.add_samples_from_array(samples, energies)
        >>> response.samples_array()
        array([[1, 1, 1], [0, 0, 0]])
        >>> for sample in response:
        ...     # still works like a normal response object
        ...     pass

    """
    def __init__(self, data=None):
        NumpyResponse.__init__(self, data)

    def add_samples_from_array(self, samples, energies, sample_data=None, sorted_by_energy=False):
        """Loads samples and associated energies from binary-valued numpy arrays.

        Args:
            samples (:obj:`numpy.ndarray`): An two dimensional numpy array
                in which each row is a sample. Values should be 0 or 1.
            energies (:obj:`numpy.ndarray`): A one dimensional numpy array
                in which each value is an energy. Must be the same length
                as `samples`.
            sample_data (iterator, optional): An iterable object
                that yields data about each sample as  dict. If
                None, then each data will be an empty dict. Default
                None.
            sorted_by_energy (bool): If True, then the user asserts that
                `samples` and `energies` are sorted by energy from low to
                high. This is not checked.

        Raises:
            ValueError: If any values in the samples are not 0 or 1.

        Notes:
            Solutions are stored in order of energy, lowest first.

        """
        if any(s not in {0, 1} for s in samples.flat):
            raise ValueError("All values in samples should be -1 or 1")

        NumpyResponse.add_samples_from_array(self, samples, energies,
                                             sample_data, sorted_by_energy)

    def as_spin(self, offset=0.0, data_copy=False):
        """Returns the :class:`.NumpySpinResponse` version of itself.

        Args:
            offset (float/int, optional): The energy offset as would
                be returned by `qubo_to_ising`. The energy offset is
                applied to each energy in the response.
            data_copy (bool, optional): Whether to create a copy
                of each data dict. Default False.

        Returns:
            NumpySpinResponse: A SpinResponse with the samples converted
            from binary to spin, the energies updated with `offset` and
            all of the data transferred directly.

        Notes:
            Only information stored in `data` property and as would be
            returned by `samples(data=True)` is transferred.


        """
        spin_response = NumpySpinResponse()

        spin_response._samples = 2 * self._samples - 1
        spin_response._energies = self._energies + offset

        if data_copy:
            spin_response.data = self.data.copy()
            spin_response._sample_data = [data.copy() for data in self._sample_data]
        else:
            spin_response.data = self.data
            spin_response._sample_data = [data for data in self._sample_data]

        return spin_response

    def as_binary_response(self, data_copy=False):
        """Returns the :class:`.BinaryResponse` version of itself.

        Args:
            data_copy (bool, optional): Whether to create a copy
                of each data dict. Default False.

        Returns:
            BinaryResponse: A BinaryResponse with the same values as the
                BinaryNumpyResponse

        Notes:
            Only information stored in `data` property and as would be
            returned by `samples(data=True)` is transferred.


        """
        if data_copy:
            response = BinaryResponse(self.data.copy())
            sample_data = [data.copy() for data in self._sample_data]
        else:
            response = BinaryResponse(self.data)
            sample_data = list(self._sample_data)

        response.add_samples_from(self.samples(), self.energies(), sample_data)

        return response
