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
from dimod.vartypes import Vartype

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
        info (dict): Information about the response as a whole.
        vartype (:class:`.Vartype`): The values that the variables in
            each sample can take. See :class:`.Vartype`.

    Examples:
        >>> response = dimod.NumpyResponse({'name': 'example'})
        >>> response.info
        {'name': 'example'}

    """
    def __init__(self, info=None, vartype=Vartype.UNDEFINED):
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
        """list[dict]: The data in order of insertion. Each datum
            in data is a dict containing 'sample', 'energy', and
            'num_occurences' keys as well an any other information added
            on insert. The list is constructed from the internal
            data arrays on first read.
        """

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
            sample_array (:obj:`numpy.ndarray`): An two dimensional numpy array
                in which each row is a sample.
            energy_array (:obj:`numpy.ndarray`): A one dimensional numpy array
                in which each value is an energy. Must be the same length
                as `samples`.
            num_occurences_array (:obj:`numpy.ndarray`, optional): A one dimensional
                numpy array giving the number of occurances of each sample. Defaults
                to one for each sample.
            datalist (list[dict], optional): Default None. If given, the information
                in each datum in datalist is stored with its associated sample.

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
        """Loads data into the response.

        Args:
            data (iterable[dict]): An iterable of datum. Each datum is a
                dict. The datum must contain 'sample' and 'energy' keys
                with dict and number values respectively.

        Examples:
            >>> response = dimod.TemplateResponse
            >>> response.add_data_from([{'energy': -1, 'num_occurences': 1, 'sample': {0: 1}},
                                        {'energy': 1, 'num_occurences': 1, 'sample': {0: -1}}])

        """
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

    def relabel_samples(self, mapping):
        """Return a new response object with the samples relabeled.

        Args:
            mapping (dict[hashable, hashable]): A dictionary with the old labels as keys
                and the new labels as values. A partial mapping is
                allowed.

        Examples:
            >>> response = NumpyResponse()
            >>> response.add_sample({'a': -1, 'b': 1}, 1)
            >>> response.add_sample({'a': 1, 'b': -1}, -1)
            >>> mapping = {'a': 1, 'b': 0}

            >>> new_response = response.relabel_samples(mapping)
            >>> list(new_response.samples())
            [{0: -1, 1: 1}, {0: 1, 1: -1}]

        """
        if len(set(mapping.values())) != len(mapping):
            raise MappingError("mapping contains repeated keys")

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


class NumpySpinResponse(NumpyResponse):
    """Subclass of :class:`.NumpyResponse` which encodes spin-valued samples.

    Args:
        info (dict): Information about the response as a whole.
        vartype (:class:`.Vartype`): The values that the variables in
            each sample can take. See :class:`.Vartype`.
    """
    def __init__(self, info=None, vartype=Vartype.SPIN):
        NumpyResponse.__init__(self, info=info, vartype=vartype)

    def as_binary(self, offset):
        """Casts self to :class:`.BinaryResponse`.

        Args:
            offset (number, optional): Default 0.0. The energy offset
                to apply to all of the energies in the response.
        """
        return self.cast(BinaryResponse, varmap={-1: 0, 1: 1}, offset=offset)


class NumpyBinaryResponse(NumpyResponse):
    """Subclass of :class:`.NumpyResponse` which encodes binary-valued samples.

    Args:
        info (dict): Information about the response as a whole.
        vartype (:class:`.Vartype`): The values that the variables in
            each sample can take. See :class:`.Vartype`.
    """
    def __init__(self, info=None, vartype=Vartype.BINARY):
        NumpyResponse.__init__(self, info=info, vartype=vartype)

    def as_spin(self, offset):
        """Casts self to :class:`.SpinResponse`.

        Args:
            offset (number, optional): Default 0.0. The energy offset
                to apply to all of the energies in the response.
        """
        return self.cast(SpinResponse, varmap={0: -1, 1: 1}, offset=offset)
