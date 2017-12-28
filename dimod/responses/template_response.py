"""
dimod Response
==============

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
from dimod.exceptions import MappingError
from dimod.vartypes import Vartype

__all__ = ['TemplateResponse']

if _PY2:
    zip = itertools.izip
    zip_longest = itertools.izip_longest

    def iteritems(d):
        return d.iteritems()

    def itervalues(d):
        return d.itervalues()
else:
    zip_longest = itertools.zip_longest

    def iteritems(d):
        return d.items()

    def itervalues(d):
        return d.values()


class TemplateResponse(object):
    """Serves as a superclass for response objects.

    Args:
        info (dict): Information about the response as a whole.
        vartype (:class:`.Vartype`): The values that the variables in
            each sample can take. See :class:`.Vartype`.

    Examples:
        >>> response = dimod.TemplateResponse({'name': 'example'})
        >>> response.info
        {'name': 'example'}

    Attributes:
        datalist (list[dict]): The data in order of insertion. Each datum
            in data is a dict containing 'sample', 'energy', and
            'num_occurences' keys as well an any other information added
            on insert. This attribute should be treated as read-only, as
            changing it can break the response's internal logic.

    """

    def __init__(self, info=None, vartype=Vartype.UNDEFINED):

        # each sample is stored as a dict in _sample_data
        self.datalist = []
        self._sorted_datalist = []

        if info is None:
            self.info = {}
        elif not isinstance(info, dict):
            raise TypeError("expected 'info' input to be a dict")
        else:
            self.info = info

        # set the vartype
        if vartype in Vartype:
            self.vartype = vartype
        elif isinstance(vartype, str):
            self.vartype = Vartype[vartype]
        else:
            self.vartype = Vartype(vartype)

    @property
    def sorted_datalist(self):
        """list[dict]: The data in order of energy, low-to-high. The datum
        stored in sorted_datalist are the same as in datalist. This list
        is generated on the first read after an insertion to the response.
        """
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
        """An iterator over the samples.

        Args:
            data (bool, optional): Default False. If True,
                returns an iterator of 2-tuples, (sample, datum).

        Yields:
            dict: The samples, in order of energy low-to-high.

            If data=True, yields 2-tuple (sample, datum). Where
            datum is the data associated with the given sample.

        Examples:
            >>> response = TemplateResponse()
            >>> response.add_samples_from([{0: -1}, {0: 1}], [1, -1])
            >>> list(response.samples())
            [{0: 1}, {0: -1}]

        """
        if data:
            return zip(self.data(keys=['sample']), self.data())
        return self.data(keys=['sample'])

    def energies(self, data=False):
        """An iterator over the energies.

        Args:
            data (bool, optional): Default False. If True,
                returns an iterator of 2-tuples, (sample, datum).

        Yields:
            number: The energies, from low-to-high.

            If data=True, yields 2-tuple (energy, datum). Where
            datum is the data associated with the given energy.

        """
        if data:
            return zip(self.data(keys=['energy']), self.data())
        return self.data(keys=['energy'])

    def data(self, keys=None, ordered_by_energy=True):
        """An iterator over the data.

        Args:
            keys (list, optional). Default None. If keys
                are provided, data yields a tuple of the
                values associated with each key in each
                datum.
            ordered_by_energy (bool, optional): Default True.
                If True, the datum (or tuples) are yielded in
                order energy, low-to-high. If False, they are
                yielded in order of insertion.

        Yields:
            dict: The datum stored in response.

            If keys are provided, returns a tuple (see parameter
            description above and example below).

        Examples:
            >>> response = dimod.TemplateResponse()
            >>> response.add_samples_from([{0: -1}, {0: 1}], [1, -1])
            >>> list(response.data())
            [{'energy': -1, 'num_occurences': 1, 'sample': {0: 1}},
             {'energy': 1, 'num_occurences': 1, 'sample': {0: -1}}]
            >>> for sample in response.data(keys=['sample']):
            ...     pass
            >>> for sample, num_occurences in response.data(keys=['sample', 'num_occurences']):
            ...     pass


        """
        # by default samples are ordered by energy low-to-high
        if ordered_by_energy:
            data = self.sorted_datalist
        else:
            data = self.datalist

        if keys is None:
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

        Note:
            Depreciation Warning: This method of access is being depreciated.
            it can be replaced by `response.data(keys=['sample', 'energy'])`

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
            >>> response = dimod.TemplateResponse()
            >>> response.add_sample({0: -1}, 1)
            >>> response.add_sample({0: 1}, -1, sample_idx=1)
            >>> list(response.data())
            [{'energy': -1, 'num_occurences': 1, 'sample': {0: 1}, 'sample_idx': 1},
             {'energy': 1, 'num_occurences': 1, 'sample': {0: -1}}]

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
            num_occurences (int): Default 1. The number of times the sample
                occurred. This is applied to each sample.
            **kwargs


        Examples:
            >>> response = dimod.TemplateResponse()
            >>> response.add_samples_from([{0: -1}, {0: 1}], [1, -1], dataval='test')
            >>> list(response.data())
            [{'dataval': 'test', 'energy': -1, 'num_occurences': 1, 'sample': {0: 1}},
             {'dataval': 'test', 'energy': 1, 'num_occurences': 1, 'sample': {0: -1}}]

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

        valid_sample_val = self.vartype.value

        def _check_iter():
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
                if valid_sample_val is not None and any(val not in valid_sample_val for val in itervalues(sample)):
                    raise ValueError("expected the biases of 'sample' to be in {}".format(valid_sample_val))

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

    def relabel_samples(self, mapping):
        """Return a new response object with the samples relabeled.

        Args:
            mapping (dict[hashable, hashable]): A dictionary with the old labels as keys
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

        # get a new response of the same type as self
        response = self.__class__(self.info.copy())

        def _change_sample_iter():
            for self_datum in self.data():
                # copy the field so they don't point to the same object
                datum = self_datum.copy()

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

    def cast(self, response_class, varmap=None, offset=0.0):
        """Casts the response to a different type of dimod response.

        Args:
            response_class (type): A dimod response class.
            varmap (dict, optional): A dict mapping a change in sample
                values. If not provided samples are not changed.
            offset (number, optional): Default 0.0. The energy offset
                to apply to all of the energies in the response.

        Returns:
            response_class: A dimod response.

        """
        response = response_class(self.info.copy())

        # the energies are offset by a constant, so the order stays the same. Thus we can
        # transfer directly.
        def _iter_datum():
            for self_datum in self.data():
                datum = self_datum.copy()

                # convert the samples
                if varmap is not None:
                    datum['sample'] = {v: varmap[val] for v, val in iteritems(datum['sample'])}
                if offset != 0.0:
                    datum['energy'] += offset

                yield datum

        response.add_data_from(_iter_datum())

        return response
