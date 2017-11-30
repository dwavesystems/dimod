"""
Response API
============

For dimod samplers to be uniform, it is important that they all respond
in the same way.

Using a response
----------------

Let us say that an application uses a the :class:`.ExactSolver` sampler to
minimize the following Ising problem:

>>> h = {'a': 1., 'b': -.5}
>>> J = {('a', 'b'): -1.}

The application would include a line of code like:

>>> response = dimod.ExactSolver().sample_ising(h, J)

There are now several ways to iterate over the samples provided by
the sampler.

The simplest is to simply iterate:

>>> for sample in response:
...     pass

The samples can also be accessed using the `samples` method:

>>> list(response.samples())
[{'a': -1, 'b': -1}, {'a': 1, 'b': 1}, {'a': -1, 'b': 1}, {'a': 1, 'b': -1}]

Note that the samples are returned in order of increasing energy. The energies
can also be queried

>>> list(response.energies())
[-1.5, -0.5, -0.5, 2.5]

Or both can be iterated over together

>>> list(response.items())
[({'a': -1, 'b': -1}, -1.5), ({'a': 1, 'b': 1}, -0.5), ({'a': -1, 'b': 1}, -0.5), ...]

Finally, if there is a data associated with any of the samples, it can accessed through
the same methods. The data is returned as a dict.

>>> for sample, data in response.samples(data=True):
...     pass
>>> for energy, data in response.energies(data=True):
...     pass
>>> for sample, energy, data in response.items(data=True):
...     pass

To access the lowest energy sample

>>> next(iter(response))
{'a': -1, 'b': -1}

Finally the response's length is the number of samples

>>> len(response)
4


Instantiating a response
------------------------

Define an example QUBO. This QUBO is minimized when variable
'v0'=1, 'v1'=0, 'v2'=1, 'v3'=0.

>>> Q = {('v0', 'v0'): -2, ('v0', 'v1'): 2, ('v1', 'v1'): -2,
...      ('v1', 'v2'): 2, ('v2', 'v2'): -2, ('v2', 'v3'): 2,
...      ('v3', 'v3'): -1}

Let's say that we draw three binary samples from some process and
calculate their corresponding energies.

>>> sample0 = {'v0': 0, 'v1': 1, 'v2': 0, 'v3': 1}
>>> sample1 = {'v0': 1, 'v1': 0, 'v2': 1, 'v3': 0}  # the minimum
>>> sample2 = {'v0': 1, 'v1': 1, 'v2': 1, 'v3': 1}
>>> energy0 = -3.
>>> energy1 = -4.
>>> energy2 = -1.

We can now add them to the response either one at a time or in
groups. In general adding in batches with `add_samples_from` is
faster.

>>> response = dimod.BinaryResponse()
>>> response.add_sample(sample0, energy0)
>>> response.add_samples_from([sample1, sample2], [energy1, energy2])


Template Response Class
-----------------------

The :obj:`.TemplateResponse` can be subclassed to make dimod compliant
response objects.

"""
from __future__ import division

import itertools
import bisect

from dimod import _PY2
from dimod.decorators import ising, qubo
from dimod.utilities import ising_energy, qubo_energy

__all__ = ['TemplateResponse']

if _PY2:
    range = xrange
    zip = itertools.izip

    def iteritems(d):
        return d.iteritems()

    def itervalues(d):
        return d.itervalues()
else:
    def iteritems(d):
        return d.items()

    def itervalues(d):
        return d.values()


class TemplateResponse(object):
    """Serves as a superclass for response objects.

    Args:
        data (dict, optional): Data about the response as a whole
            as a dictionary. Default {}.

    Examples:
        >>> response = TemplateResponse({'name': 'example'})
        >>> print(response.data)
        '{'name': 'example'}'

    """

    def __init__(self, data=None):
        self._samples = []
        self._energies = []
        self._sample_data = []
        if data is None:
            self.data = {}
        elif not isinstance(data, dict):
            raise TypeError('expected input "data" to be None or a dict')
        else:
            self.data = data

    def __iter__(self):
        """Iterate over the samples. Use the expression 'for sample in
        response'.

        Returns:
            iterator: An iterator over all samples in the response,
            in order of increasing energy.

        Examples:
            >>> response = TemplateResponse()
            >>> response.add_samples_from([{0: -1}, {0: 1}], [1, -1])
            >>> [s for s in response]
            [{0: 1}, {0: -1}]

        """
        return self.samples()

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

        Examples:
            >>> response = TemplateResponse()
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
            >>> response = TemplateResponse()
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
            >>> response = TemplateResponse()
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

        Raises:
            TypeError: If `sample` is not a dict.
            TypeError: If `energy` is not an int or float.
            TypeError: If `data` is not a dict.

        Examples:
            >>> response = TemplateResponse()
            >>> response.add_sample({0: -1}, 1)
            >>> response.add_sample({0: 1}, -1, data={'n': 1})

        """

        # create a new empty dict
        if data is None:
            data = {}

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
                that yields data about each sample as  dict. If
                None, then each data will be an empty dict. Default
                None.

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

            >>> response = TemplateResponse()
            >>> response.add_samples_from(samples, energies)
            >>> list(response.samples())
            [{0: 1}, {0: -1}, {0: -1}]

            >>> response = TemplateResponse()
            >>> response.add_samples_from(samples, energies, sample_data)
            >>> list(response.samples())
            [{0: 1}, {0: -1}, {0: -1}]

            >>> items = [({0: -1}, -1), ({0: -1}, 1)]
            >>> response = TemplateResponse()
            >>> response.add_samples_from(*zip(*items))
            >>> list(response.samples())
            [{0: 1}, {0: -1}]

        """

        N = len(self)  # current number of samples

        samples = list(samples)
        energies = list(energies)

        if not all(isinstance(sample, dict) for sample in samples):
            raise TypeError("expected each sample in 'samples' to be a dict")
        if not all(isinstance(energy, (float, int)) for energy in energies):
            raise TypeError("expected each energy in 'energies' to be numeric")

        if sample_data is None:
            sample_data = [{} for __ in energies]
        else:
            sample_data = list(sample_data)
            if not all(isinstance(data, dict) for data in sample_data):
                raise TypeError("expected sample_data to be an iterator of dicts")

        if N > 0:
            # if we already have samples, concatenate
            samples += self._samples
            energies += self._energies
            sample_data += self._sample_data

        # order the new lists by energy
        order = sorted(range(len(energies)), key=energies.__getitem__)

        # replace the stored samples/energies with the new list
        self._samples = [samples[i] for i in order]
        self._energies = [energies[i] for i in order]
        self._sample_data = [sample_data[i] for i in order]

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
        """The number of samples in response."""
        return self._samples.__len__()

    def relabel_samples(self, mapping):
        """Relabels the variable in the samples.

        Args:
            mapping (dict): A dictionary with the old labels as keys
                and the new labels as values. A partial mapping is
                allowed.

        Examples:
            >>> response = TemplateResponse()
            >>> response.add_sample({'a': -1, 'b': 1}, 1)
            >>> response.add_sample({'a': 1, 'b': -1}, -1)
            >>> mapping = {'a': 1, 'b': 0}

            >>> new_response = response.relabel_samples(mapping)
            >>> list(new_response.samples())
            [{0: -1, 1: 1}, {0: 1, 1: -1}]

        """

        try:
            return _relabel_copy(self, mapping)
        except MappingError:
            raise ValueError('given mapping does not have unique values.')


class MappingError(Exception):
    """mapping causes conflicting values in samples"""


def _relabel_copy(response, mapping):
    """Creates a new response with the variables relabeled according
    to mapping.
    """

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
                rl_sample[new_v] = val
            else:
                rl_sample[v] = val
        rl_response.add_sample(rl_sample, energy, data)

    # return the new object
    return rl_response
