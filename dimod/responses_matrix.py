"""
TODO
"""
import sys
import itertools

from dimod import BinaryResponse, SpinResponse

PY2 = sys.version_info[0] == 2
if PY2:
    range = xrange
    zip = itertools.izip
    iteritems = lambda d: d.iteritems()
    itervalues = lambda d: d.itervalues()
else:
    iteritems = lambda d: d.items()
    itervalues = lambda d: d.values()


class NumpyResponse(object):
    """TODO"""
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
        if data:
            return zip(self.samples(), self._sample_data)

        return iter({idx: val for idx, val in enumerate(row)} for row in self._samples)

    def energies(self, data=False):
        if data:
            return zip(self.energies(), self._sample_data)

        return iter(self._energies)

    def items(self, data=False):
        if data:
            return zip(self.samples(), self.energies(), self._sample_data)

        return zip(self.samples(), self.energies())

    def add_sample(self, sample, energy, data=None):
        raise NotImplementedError(("To add samples to NumpyResponse, "
                                   "please use add_samples_from_array"))

    def add_samples_from(self, samples, energies, sample_data=None):
        raise NotImplementedError(("To add samples to NumpyResponse, "
                                   "please use add_samples_from_array"))

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

    def __len__(self):
        num_samples, __ = self._samples.shape
        return num_samples

    def relabel_samples(self, mapping, copy=True):
        raise NotImplementedError("NumpyResponse does not support arbitrarily labeled variables")

    def add_samples_from_array(self, samples, energies, sample_data=None, sorted_by_energy=False):
        import numpy as np

        # # Input checking
        # #   expect samples to be 2darray
        # #   expect energies to be 1darray
        # #   expect sample_data to be None or an iterable of dicts
        # if not isinstance(samples, np.ndarray):
        #     raise NotImplementedError  # TODO
        # if samples.ndim != 2:
        #     raise ValueError('expected samples to by a numpy ndarray with 2 dimensions')

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


class NumpySpinResponse(NumpyResponse):
    def __init__(self, data=None):
        NumpyResponse.__init__(self, data)

    def add_samples_from_array(self, samples, energies, sample_data=None, sorted_by_energy=False):

        if any(s not in (-1, 1) for s in row for row in samples):
            raise ValueError("All values in samples should be -1 or 1")

        NumpyResponse.add_samples_from_array(self, samples, energies,
                                             sample_data, sorted_by_energy)

    def as_binary(self, offset=0.0, data_copy=False):
        binary_response = NumpyBinaryResponse()

        binary_response._samples = (self._samples + 1) // 2
        binary_response._energies = self._energies + offset

        if data_copy:
            binary_response.data = self.data.copy()
            binary_response._sample_data = [data.copy() for data in self._sample_data]
        else:
            binary_response.data = self.data
            binary_response._sample_data = [data for data in self._sample_data]

    def as_dimod_response(self, data_copy=False):

        if data_copy:
            response = SpinResponse(self.data.copy())
            sample_data = [data.copy() for data in self._sample_data]
        else:
            response = SpinResponse(self.data)
            sample_data = list(self._sample_data)

        response.add_samples_from(self.samples(), self.energies(), sample_data)

        return response


class NumpyBinaryResponse(NumpyResponse):
    def __init__(self, data=None):
        NumpyResponse.__init__(self, data)

    def add_samples_from_array(self, samples, energies, sample_data=None, sorted_by_energy=False):

        if any(s not in (0, 1) for s in row for row in samples):
            raise ValueError("All values in samples should be -1 or 1")

        NumpyResponse.add_samples_from_array(self, samples, energies,
                                             sample_data, sorted_by_energy)

    def as_spin(self, offset=0.0, data_copy=False):
        spin_response = NumpySpinResponse()

        spin_response._samples = 2 * self._samples - 1
        spin_response._energies = self._energies + offset

        if data_copy:
            spin_response.data = self.data.copy()
            spin_response._sample_data = [data.copy() for data in self._sample_data]
        else:
            spin_response.data = self.data
            spin_response._sample_data = [data for data in self._sample_data]

    def as_dimod_response(self, data_copy=False):

        if data_copy:
            response = BinaryResponse(self.data.copy())
            sample_data = [data.copy() for data in self._sample_data]
        else:
            response = BinaryResponse(self.data)
            sample_data = list(self._sample_data)

        response.add_samples_from(self.samples(), self.energies(), sample_data)

        return response
