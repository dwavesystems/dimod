"""
TODO
"""

# from dimod.responses import TemplateResponse


class NumpyResponse():
    """TODO"""
    def __init__(self, data=None):
        import numpy as np

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
        raise NotImplementedError

    def energies(self, data=False):
        raise NotImplementedError

    def items(self, data=False):
        raise NotImplementedError

    def add_sample(self, sample, energy, data=None):
        raise NotImplementedError

    def add_samples_from(self, samples, energies, sample_data=None):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __getitem__(self, sample):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def relabel_samples(self, mapping, copy=True):
        raise NotImplementedError

    def add_samples_from_array(self, samples, energies, sample_data=None, sorted_by_energy=False):
        import numpy as np

        # # Input checking
        # #   expect samples to be 2darray
        # #   expect energies to be 1darray or 2d vector
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
        elif sorted_by_energy:
            # some samples already in response and the new samples are sorted
            raise NotImplementedError()
        elif self._samples is None:
            # response is empty and new samples are unsorted
            raise NotImplementedError
        else:
            # some samples already in response and new are unsorted
            raise NotImplementedError
