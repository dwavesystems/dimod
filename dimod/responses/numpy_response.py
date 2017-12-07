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

from numbers import Number

from dimod import _PY2
from dimod.exceptions import MappingError
from dimod.responses.template_response import TemplateResponse
from dimod.responses.type_response import BinaryResponse, SpinResponse
from dimod.vartypes import VARTYPES

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
        todo

    """
    def __init__(self, info=None, vartype=VARTYPES.UNDEFINED):
        import numpy as np

        TemplateResponse.__init__(self, info=info, vartype=vartype)
        # self.datalist = []  # we overwrite this property with our own getter/setter
        # self.sorted_datalist = []  # this is a @property inherited from TemplateResponse

        # we store 'sample', 'energy' and 'num_occurences' in numpy arrays,
        # all other keys are stored in datalist like normal
        self.sample_array = np.empty((0, 0), dtype=int)
        self.energy_array = np.empty((0,), dtype=float)
        self.num_occurences_array = np.empty((0,), dtype=int)

        # variable_labels assigns a label to each column of the sample_array
        self.variable_labels = None

    @property
    def datalist(self):
        """todo"""

        if len(self) != len(self._datalist):
            self._datalist.extend(({} for __ in range(len(self._datalist), len(self))))

        for idx, datum in enumerate(self._datalist):
            if 'sample' not in datum:
                if self.variable_labels is None:
                    self.variable_labels = list(range(len(self.sample_array[idx, :])))
                datum['sample'] = dict(zip(self.variable_labels, self.sample_array[idx, :]))

            if 'energy' not in datum:
                datum['energy'] = self.energy_array[idx]

            if 'num_occurences' not in datum:
                datum['num_occurences'] = int(self.num_occurences_array[idx])

        return self._datalist

    @datalist.setter
    def datalist(self, datalist):
        """setter for datalist, see the getter for documentation. Should
        only be used by __init__ to set an empty list."""
        self._datalist = datalist

    def __len__(self):
        """The number of samples in response."""
        num_samples, __ = self.sample_array.shape
        return num_samples

    def add_samples_from_array(self, sample_array, energy_array, num_occurences_array=None, datalist=None):
        """Loads samples and associated energies from numpy arrays.

        todo

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
            todo

        Notes:
            Solutions are stored in order of energy, lowest first.

        """
        import numpy as np

        valid_sample_bias = self.vartype.value

        if not isinstance(sample_array, np.ndarray):
            raise TypeError("expected 'sample_array' to be a two dimensional ndarray")
        if sample_array.ndim != 2:
            raise ValueError("expected 'sample_array' to be a two dimensional ndarray")
        if valid_sample_bias is not None:
            if any(s not in valid_sample_bias for s in sample_array.flat):
                ValueError("invalid sample bias, expected to be in {}".format(valid_sample_bias))

        num_samples, num_variables = sample_array.shape

        if not isinstance(energy_array, np.ndarray):
            raise TypeError("expected 'energy_array' to be a one dimensional ndarray")
        if energy_array.ndim != 1:
            raise ValueError("expected 'energy_array' to be a one dimensional ndarray")
        if len(energy_array) != num_samples:
            raise ValueError("length of 'sample_array' and 'energy_array' do not match")

        if num_occurences_array is None:
            num_occurences_array = np.ones(num_samples, dtype=np.int)
        else:
            if not isinstance(num_occurences_array, np.ndarray):
                raise TypeError("expected 'num_occurences_array' to be a one dimensional ndarray")
            if num_occurences_array.ndim != 1:
                raise ValueError("expected 'num_occurences_array' to be a one dimensional ndarray")
            if len(num_occurences_array) != num_samples:
                raise ValueError("length of 'sample_array' and 'num_occurences_array' do not match")

        if datalist is None:
            datalist = [{} for __ in range(num_samples)]
        else:
            if not isinstance(datalist, list):
                datalist = list(datalist)
            if len(datalist) != num_samples:
                raise ValueError("length of 'datalist' and does not match number of samples")
            if not all(isinstance(datum, dict) for datum in datalist):
                raise TypeError("expected each datum in 'datalist' to be a dict")

        # finally save everything note that we extend or overwrite the protected
        # form of datalist because otherwise the getter would extend it before
        # we overwrite it
        if len(self):
            self.sample_array = np.concatenate((self.sample_array, sample_array), axis=0)
            self.energy_array = np.concatenate((self.energy_array, energy_array))
            self.num_occurences_array = np.concatenate((self.num_occurences_array, num_occurences_array))
            self._datalist.extend(datalist)  # all the additional information
        else:
            self.sample_array = sample_array
            self.energy_array = energy_array
            self.num_occurences_array = num_occurences_array
            self._datalist = datalist  # all the additional information

    def add_data_from(self, data):
        """todo"""
        import numpy as np

        sampleslist = []
        energieslist = []
        num_occurenceslist = []
        datalist = []

        variable_labels = self.variable_labels

        for datum in data:
            # iterate over each datum and do type checking
            if not isinstance(datum, dict):
                raise TypeError("each datum in 'data' should be a dict")

            # sample
            if 'sample' not in datum:
                raise ValueError("each datum in 'data' must include a 'sample' key with a dict value")
            sample = datum['sample']
            if not isinstance(sample, dict):
                raise TypeError("expected 'sample' to be a dict")
            # checking against vartype is done in add_samples_from_array

            # energy
            if 'energy' not in datum:
                raise ValueError("each datum in 'data' must include an 'energy' key with a numeric value")
            if not isinstance(datum['energy'], Number):
                raise TypeError("expected 'energy' to be a number")

            # num_occurences, default = 1
            if 'num_occurences' in datum:
                if not isinstance(datum['num_occurences'], int):
                    raise TypeError("expected 'num_occurences' to be a positive int")
                if datum['num_occurences'] <= 0:
                    raise ValueError("expected 'num_occurences' to be a positive int")
            else:
                datum['num_occurences'] = 1

            # add sample to a list of lists
            if variable_labels is None:
                if len(self) != 0:
                    raise RuntimeError("internal error - variable_labels should only be unset with no samples")
                try:
                    self.variable_labels = variable_labels = sorted(list(sample))
                except TypeError:
                    # in python3 unlike types cannot be sorted
                    self.variable_labels = variable_labels = list(sample)
            else:
                if any(v not in sample for v in self.variable_labels):
                    raise TypeError("NumpResponse requires consistent variable labels")

            sampleslist.append([sample[v] for v in variable_labels])

            energieslist.append(datum['energy'])
            num_occurenceslist.append(datum['num_occurences'])
            datalist.append(datum)

        sample_array = np.asarray(sampleslist, dtype=int)
        energy_array = np.array(energieslist, dtype=float)
        num_occurences_array = np.array(num_occurenceslist, dtype=int)

        self.add_samples_from_array(sample_array, energy_array, num_occurences_array=num_occurences_array,
                                    datalist=datalist)

    def relabel_samples(self, mapping, copy=True):
        """todo"""
        if len(set(mapping.values())) != len(mapping):
            raise MappingError("mapping contains repeated keys")
        if copy:
            response = NumpyResponse(self.info)
            response.add_samples_from_array(self.sample_array.copy(),
                                            self.energy_array.copy(),
                                            self.num_occurences_array.copy(),
                                            datalist=[datum.copy() for datum in self.datalist])
            if self.variable_labels is None:
                response.variable_labels = [mapping[v] for v in range(len(mapping))]
            else:
                response.variable_labels = [mapping[v] for v in self.variable_labels]
            return response
        else:
            if self.variable_labels is None:
                self.variable_labels = [mapping[v] for v in range(len(mapping))]
            else:
                self.variable_labels = [mapping[v] for v in self.variable_labels]
            return self


class NumpySpinResponse(NumpyResponse):
    def __init__(self, info=None, vartype=VARTYPES.SPIN):
        NumpyResponse.__init__(self, info=info, vartype=vartype)

    def as_binary(self, offset):
        raise NotImplementedError


class NumpyBinaryResponse(NumpyResponse):
    def __init__(self, info=None, vartype=VARTYPES.BINARY):
        NumpyResponse.__init__(self, info=info, vartype=vartype)

    def as_spin(self, offset, copy=True):
        return self.cast(SpinResponse, varmap={0: -1, 1: 1}, offset=offset, copy=copy)

# class NumpySpinResponse(NumpyResponse):
#     """Subclass of :class:`.NumpyResponse` which encodes spin-valued samples.

#     Differs from the :class:`.SpinResponse` by storing samples and energies
#     internally in a :obj:`numpy.ndarray`.

#     Args:
#         data (dict, optional): Data about the response as a whole
#             as a dictionary. Default {}.

#     Examples:
#         >>> import numpy as np
#         >>> response = NumpySpinResponse()
#         >>> samples = np.asarray([[-1, -1, 1], [1, 1, -1]], dtype=int)
#         >>> energies = np.asarray([2., -2.], dtype=float)
#         >>> response.add_samples_from_array(samples, energies)
#         >>> response.samples_array()
#         array([[1, 1, -1], [-1, -1, 1]])
#         >>> for sample in response:
#         ...     # still works like a normal response object
#         ...     pass

#     """
#     def __init__(self, data=None):
#         raise NotImplementedError
#         NumpyResponse.__init__(self, data)

#     def add_samples_from_array(self, samples, energies, sample_data=None, sorted_by_energy=False):
#         """Loads samples and associated energies from spin-valued numpy arrays.

#         Args:
#             samples (:obj:`numpy.ndarray`): An two dimensional numpy array
#                 in which each row is a sample. Values should be -1 or 1.
#             energies (:obj:`numpy.ndarray`): A one dimensional numpy array
#                 in which each value is an energy. Must be the same length
#                 as `samples`.
#             sample_data (iterator, optional): An iterable object
#                 that yields data about each sample as  dict. If
#                 None, then each data will be an empty dict. Default
#                 None.
#             sorted_by_energy (bool): If True, then the user asserts that
#                 `samples` and `energies` are sorted by energy from low to
#                 high. This is not checked.

#         Raises:
#             ValueError: If any values in the samples are not -1 or 1.

#         Notes:
#             Solutions are stored in order of energy, lowest first.

#         """
#         if any(s not in {-1, 1} for s in samples.flat):
#             raise ValueError("All values in samples should be -1 or 1")

#         NumpyResponse.add_samples_from_array(self, samples, energies,
#                                              sample_data, sorted_by_energy)

#     def as_binary(self, offset=0.0, data_copy=False):
#         """Returns the :class:`.NumpyBinaryResponse` version of itself.

#         Args:
#             offset (float/int, optional): The energy offset as would
#                 be returned by `ising_to_qubo`. The energy offset is
#                 applied to each energy in the response.
#             data_copy (bool, optional): Whether to create a copy
#                 of each data dict. Default False.

#         Returns:
#             NumpyBinaryResponse: A BinaryResponse with the samples converted
#             from spin to binary, the energies updated with `offset` and
#             all of the data transferred directly.

#         Notes:
#             Only information stored in `data` property and as would be
#             returned by `samples(data=True)` is transferred.


#         """
#         binary_response = NumpyBinaryResponse()

#         binary_response._samples = (self._samples + 1) // 2
#         binary_response._energies = self._energies + offset

#         if data_copy:
#             binary_response.data = self.data.copy()
#             binary_response._sample_data = [data.copy() for data in self._sample_data]
#         else:
#             binary_response.data = self.data
#             binary_response._sample_data = [data for data in self._sample_data]

#         return binary_response

#     def as_spin_response(self, data_copy=False):
#         """Returns the :class:`.SpinResponse` version of itself.

#         Args:
#             data_copy (bool, optional): Whether to create a copy
#                 of each data dict. Default False.

#         Returns:
#             SpinResponse: A SpinResponse with the same values as the
#                 SpinNumpyResponse

#         Notes:
#             Only information stored in `data` property and as would be
#             returned by `samples(data=True)` is transferred.


#         """
#         if data_copy:
#             response = SpinResponse(self.data.copy())
#             sample_data = [data.copy() for data in self._sample_data]
#         else:
#             response = SpinResponse(self.data)
#             sample_data = list(self._sample_data)

#         response.add_samples_from(self.samples(), self.energies(), sample_data)

#         return response


# class NumpyBinaryResponse(NumpyResponse):
#     """Subclass of :class:`.NumpyResponse` which encodes binary-valued samples.

#     Differs from the :class:`.BinaryResponse` by storing samples and energies
#     internally in a :obj:`numpy.ndarray`.

#     Args:
#         data (dict, optional): Data about the response as a whole
#             as a dictionary. Default {}.

#     Examples:
#         >>> import numpy as np
#         >>> response = NumpyBinaryResponse()
#         >>> samples = np.asarray([[0, 0, 0], [1, 1, 1]], dtype=int)
#         >>> energies = np.asarray([0., -3.], dtype=float)
#         >>> response.add_samples_from_array(samples, energies)
#         >>> response.samples_array()
#         array([[1, 1, 1], [0, 0, 0]])
#         >>> for sample in response:
#         ...     # still works like a normal response object
#         ...     pass

#     """
#     def __init__(self, data=None):
#         raise NotImplementedError
#         NumpyResponse.__init__(self, data)

#     def add_samples_from_array(self, samples, energies, sample_data=None, sorted_by_energy=False):
#         """Loads samples and associated energies from binary-valued numpy arrays.

#         Args:
#             samples (:obj:`numpy.ndarray`): An two dimensional numpy array
#                 in which each row is a sample. Values should be 0 or 1.
#             energies (:obj:`numpy.ndarray`): A one dimensional numpy array
#                 in which each value is an energy. Must be the same length
#                 as `samples`.
#             sample_data (iterator, optional): An iterable object
#                 that yields data about each sample as  dict. If
#                 None, then each data will be an empty dict. Default
#                 None.
#             sorted_by_energy (bool): If True, then the user asserts that
#                 `samples` and `energies` are sorted by energy from low to
#                 high. This is not checked.

#         Raises:
#             ValueError: If any values in the samples are not 0 or 1.

#         Notes:
#             Solutions are stored in order of energy, lowest first.

#         """
#         if any(s not in {0, 1} for s in samples.flat):
#             raise ValueError("All values in samples should be -1 or 1")

#         NumpyResponse.add_samples_from_array(self, samples, energies,
#                                              sample_data, sorted_by_energy)

#     def as_spin(self, offset=0.0, data_copy=False):
#         """Returns the :class:`.NumpySpinResponse` version of itself.

#         Args:
#             offset (float/int, optional): The energy offset as would
#                 be returned by `qubo_to_ising`. The energy offset is
#                 applied to each energy in the response.
#             data_copy (bool, optional): Whether to create a copy
#                 of each data dict. Default False.

#         Returns:
#             NumpySpinResponse: A SpinResponse with the samples converted
#             from binary to spin, the energies updated with `offset` and
#             all of the data transferred directly.

#         Notes:
#             Only information stored in `data` property and as would be
#             returned by `samples(data=True)` is transferred.


#         """
#         spin_response = NumpySpinResponse()

#         spin_response._samples = 2 * self._samples - 1
#         spin_response._energies = self._energies + offset

#         if data_copy:
#             spin_response.data = self.data.copy()
#             spin_response._sample_data = [data.copy() for data in self._sample_data]
#         else:
#             spin_response.data = self.data
#             spin_response._sample_data = [data for data in self._sample_data]

#         return spin_response

#     def as_binary_response(self, data_copy=False):
#         """Returns the :class:`.BinaryResponse` version of itself.

#         Args:
#             data_copy (bool, optional): Whether to create a copy
#                 of each data dict. Default False.

#         Returns:
#             BinaryResponse: A BinaryResponse with the same values as the
#                 BinaryNumpyResponse

#         Notes:
#             Only information stored in `data` property and as would be
#             returned by `samples(data=True)` is transferred.


#         """
#         if data_copy:
#             response = BinaryResponse(self.data.copy())
#             sample_data = [data.copy() for data in self._sample_data]
#         else:
#             response = BinaryResponse(self.data)
#             sample_data = list(self._sample_data)

#         response.add_samples_from(self.samples(), self.energies(), sample_data)

#         return response
