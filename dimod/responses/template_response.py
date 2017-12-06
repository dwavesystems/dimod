"""
Response API
============

It is important that all dimod samplers respond in the same way.

Reading from a dimod response
-----------------------------

Let us say that an application uses the :class:`.ExactSolver` sampler to
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

The :class:`.TemplateResponse` can be subclassed to make dimod compliant
response objects.

"""
from __future__ import division

import warnings
import itertools

from numbers import Number

from dimod import _PY2
from dimod.decorators import ising, qubo
from dimod.exceptions import MappingError
from dimod.utilities import ising_energy, qubo_energy

__all__ = ['TemplateResponse']

if _PY2:
    zip = itertools.izip
    zip_longest = itertools.izip_longest

    def iteritems(d):
        return d.iteritems()
else:
    zip_longest = itertools.zip_longest

    def iteritems(d):
        return d.items()


class TemplateResponse(object):
    """Serves as a superclass for response objects.

    Args:
        todo

    Attributes:
        todo

    Examples:
        >>> response = TemplateResponse({'name': 'example'})
        >>> response.data
        {'name': 'example'}

    """

    def __init__(self, info=None):

        # each sample is stored as a dict in _sample_data
        self.datalist = []
        self._sorted_datalist = []

        if info is None:
            self.info = {}
        elif not isinstance(info, dict):
            raise TypeError("expected 'info' input to be a dict")
        else:
            self.info = info

    @property
    def sorted_datalist(self):
        """todo"""
        if len(self.datalist) != len(self._sorted_datalist):
            # sorting will be faster when the list is partially sorted, so just add the new
            # samples onto the end of our previously sorted list
            data = self._sorted_datalist
            data.extend(self.datalist[len(data):])

            self._sorted_datalist.sort(key=lambda d: d['energy'])
        return self._sorted_datalist

    def __iter__(self):
        """Iterate over the samples.

        Returns:
            iterator: An iterator over all samples in the response,
            in order of increasing energy.

        Examples:
            >>> response = TemplateResponse()
            >>> response.add_samples_from([{0: -1}, {0: 1}], [1, -1])
            >>> [s for s in response]
            [{0: 1}, {0: -1}]

        """
        return iter(self.samples())

    def samples(self, data=False):
        """todo"""
        if data:
            return zip(self.data(keys=['sample']), self.data())
        return self.data(keys=['sample'])

    def energies(self, data=False):
        """todo"""
        if data:
            return zip(self.data(keys=['energy']), self.data())
        return self.data(keys=['energy'])

    def data(self, keys=[], ordered_by_energy=True):
        """todo"""
        # by default samples are ordered by energy low-to-high
        if ordered_by_energy:
            data = self.sorted_datalist
        else:
            data = self.datalist

        if not keys:
            for datum in data:
                yield datum
        else:
            try:
                if len(keys) > 1:
                    for datum in data:
                        yield tuple(datum[key] for key in keys)
                else:
                    key, = keys
                    for datum in data:
                        yield datum[key]
            except KeyError:
                raise KeyError("at least one key in 'keys' is not present in every datum")

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
        warnings.warn("'items' will be depreciated in dimod 1.0.0", DeprecationWarning)
        if data:
            return zip(self.samples(), self.energies(), self.data())
        return zip(self.samples(), self.energies())

    def add_sample(self, sample, energy, num_occurences=1, **kwargs):
        """Loads a sample and associated energy into the response.

        Args:
            sample (dict): A sample as would be returned by a discrete
                model solver. Should be a dict of the form
                {var: value, ...}.
            energy (number): The energy associated with the given
                sample.
            num_occurences (int): The number of times the sample occurred.
            **kwargs

        Examples:
            >>> response = TemplateResponse()
            >>> response.add_sample({0: -1}, 1)
            >>> response.add_sample({0: 1}, -1, num_occurences=1)

        """

        kwargs['sample'] = sample
        kwargs['energy'] = energy
        kwargs['num_occurences'] = num_occurences

        self.add_data_from((kwargs,))

    def add_samples_from(self, samples, energies, num_occurences=1, **kwargs):
        """Loads samples and associated energies from iterators.

        Args:
            samples (iterator): An iterable object that yields
                samples. Each sample should be a dict of the form
                {var: value, ...}.
            energies (iterator): An iterable object that yields
                energies associated with each sample.
            todo


        Examples:
            todo

        """
        stop = object()  # signals that the two iterables are different lengths
        try:
            zipped = zip_longest(samples, energies, fillvalue=stop)
        except TypeError:
            raise TypeError("both 'samples' and 'energies' must be iterable")

        def _iter_datum():
            # construct the data dict by adding sample and energy
            for sample, energy in zipped:
                if energy == stop and sample != stop:
                    raise ValueError("'samples' is longer than 'energies', expected to be the same length")
                elif sample == stop and energy != stop:
                    raise ValueError("'energies' is longer than 'samples', expected to be the same length")

                datum = {'sample': sample, 'energy': energy, 'num_occurences': num_occurences}
                datum.update(**kwargs)
                yield datum

        self.add_data_from(_iter_datum())

    def add_data_from(self, data):
        """todo"""
        def _check_iter():
            for datum in data:
                # iterate over each datum and do type checking
                if not isinstance(datum, dict):
                    raise TypeError("each datum in 'data' should be a dict")

                # sample
                if 'sample' not in datum:
                    raise ValueError("each datum in 'data' must include a 'sample' key with a dict value")
                if not isinstance(datum['sample'], dict):
                    raise TypeError("expected 'sample' to be a dict")

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

                yield datum

        self.datalist.extend(_check_iter())

    def __str__(self):
        """Return a string representation of the response.

        Returns:
            str: A string representation of the graph.

        """
        lines = [self.__repr__(), 'info: {}'.format(self.info)]

        if len(self) > 10:
            data = self.data()
            for __ in range(10):
                lines.append('{}'.format(next(data)))
            lines.append('... ({} total)'.format(len(self)))
        else:
            lines.extend('{}'.format(datum) for datum in self.data())

        return '\n'.join(lines)

    def __len__(self):
        """The number of samples in response."""
        return len(self.datalist)

    def relabel_samples(self, mapping, copy=True):
        """Return a new response object with the samples relabeled.

        Args:
            mapping (dict[hashable, hashable]): A dictionary with the old labels as keys
                and the new labels as values. A partial mapping is
                allowed.
            copy (bool, optional): If copy is True, the datum are copied, if false
                then they will be the same objects.

        Examples:
            >>> response = TemplateResponse()
            >>> response.add_sample({'a': -1, 'b': 1}, 1)
            >>> response.add_sample({'a': 1, 'b': -1}, -1)
            >>> mapping = {'a': 1, 'b': 0}

            >>> new_response = response.relabel_samples(mapping)
            >>> list(new_response.samples())
            [{0: -1, 1: 1}, {0: 1, 1: -1}]

        """

        # get a new response of the same type as self
        if copy:
            response = self.__class__(self.info.copy())
        else:
            response = self.__class__(self.info)

        def _change_sample_iter():
            for self_datum in self.data():
                if copy:
                    # copy the field so they don't point to the same object
                    datum = self_datum.copy()
                else:
                    datum = self_datum

                sample = datum['sample']

                try:
                    new_sample = {mapping[v]: val for v, val in iteritems(sample)}
                except KeyError:
                    for v in sample:
                        if v not in mapping:
                            raise KeyError("no mapping for sample variable '{}'".format(v))

                if len(new_sample) != len(sample):
                    raise MappingError("mapping contains repeated keys")

                datum['sample'] = new_sample

                yield datum

        response.add_data_from(_change_sample_iter())

        if 'relabelings_applied' in response.info:
            response.info['relabelings_applied'].append(mapping)
        else:
            response.info['relabelings_applied'] = [mapping]

        return response

    def cast(self, response_class, varmap=None, offset=0.0, copy=True):
        """TODO"""
        if copy:
            response = response_class(self.info.copy())
        else:
            response = response_class(self.info)

        # the energies are offset by a constant, so the order stays the same. Thus we can
        # transfer directly.
        def _iter_datum():
            for self_datum in self.data():
                if copy:
                    datum = self_datum.copy()
                else:
                    datum = self_datum

                # convert the samples
                if varmap is not None:
                    datum['sample'] = {v: varmap[val] for v, val in iteritems(datum['sample'])}
                if offset != 0.0:
                    datum['energy'] += offset

                yield datum

        response.add_data_from(_iter_datum())

        return response
