# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
The :class:`.Sampler` abstract base class (see :mod:`abc`) helps you create new
dimod samplers.

Any new dimod sampler must define a subclass of :class:`.Sampler` that implements
abstract properties :attr:`~.Sampler.parameters` and :attr:`~.Sampler.properties`
and one of the abstract methods :meth:`~.Sampler.sample`, :meth:`~.Sampler.sample_ising`,
or :meth:`~.Sampler.sample_qubo`. The :class:`.Sampler` class provides the complementary
methods as mixins and ensures consistent responses.

Implemented sample methods must accept, and warn on, unknown keyword arguments
`**kwargs`. This means that all implemented sample methods must have the
`**kwargs` parameter. :meth:`~.Sampler.remove_unknown_kwargs` is a convenience
method provided for this purpose.

For example, the following steps show how to easily create a dimod sampler. It is
sufficient to implement a single method (in this example the :meth:`sample_ising` method)
to create a dimod sampler with the :class:`.Sampler` class.

.. testcode::

    class LinearIsingSampler(dimod.Sampler):

        def sample_ising(self, h, J, **kwargs):
            kwargs = self.remove_unknown_kwargs(**kwargs)
            sample = linear_ising(h, J)
            energy = dimod.ising_energy(sample, h, J)
            return dimod.SampleSet.from_samples([sample], vartype='SPIN', energy=[energy])

        @property
        def properties(self):
            return dict()

        @property
        def parameters(self):
            return dict()

For this example, the implemented sampler :meth:`~.Sampler.sample_ising` can be based on
a simple placeholder function, which returns a sample that minimizes the linear terms:

.. testcode::

    def linear_ising(h, J):
        sample = {}
        for v in h:
            if h[v] < 0:
                sample[v] = +1
            else:
                sample[v] = -1
        return sample


The :class:`.Sampler` ABC provides the other sample methods "for free"
as mixins.

>>> sampler = LinearIsingSampler()
...
... # Implemented by class LinearIsingSampler:
>>> response = sampler.sample_ising({'a': -1}, {})
...
...  # Mixins provided by Sampler class:
>>> response = sampler.sample_qubo({('a', 'a'): 1})  
>>> response = sampler.sample(dimod.BinaryQuadraticModel.from_ising({'a': -1}, {}))

Below is a more complex version of the same sampler, where the :attr:`properties` and
:attr:`parameters` properties return non-empty dicts.

.. testcode::

    class FancyLinearIsingSampler(dimod.Sampler):
        def __init__(self):
            self._properties = {'description': 'a simple sampler that only considers the linear terms'}
            self._parameters = {'verbose': []}

        def sample_ising(self, h, J, verbose=False, **kwargs):
            kwargs = self.remove_unknown_kwargs(**kwargs)
            sample = linear_ising(h, J)
            energy = dimod.ising_energy(sample, h, J)
            if verbose:
                print(sample)
            return dimod.SampleSet.from_samples([sample], energy=[energy])

        @property
        def properties(self):
            return self._properties

        @property
        def parameters(self):
            return self._parameters


"""
import abc
import warnings

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.exceptions import InvalidSampler, SamplerUnknownArgWarning
from dimod.vartypes import Vartype


__all__ = ['Sampler']


class SamplerABCMeta(abc.ABCMeta):
    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = abc.ABCMeta.__new__(mcls, name, bases, namespace, **kwargs)

        samplermixins = {name
                         for name, value in namespace.items()
                         if getattr(value, "__issamplemixin__", False)}
        if len(samplermixins) == 3:
            abstracts = samplermixins
        else:
            abstracts = set()

        for base in bases:
            samplermixins = {name
                             for name in getattr(base, "__abstractmethods__", set())
                             if getattr(getattr(cls, name, None), "__issamplemixin__", False)}
            if len(samplermixins) == 3:
                abstracts.update(samplermixins)

        # if we found any, update abstract methods
        if abstracts:
            cls.__abstractmethods__ = frozenset(abstracts.union(cls.__abstractmethods__))

        return cls


def samplemixinmethod(method):
    """Marks a method as being a mixin.

    Adds the '__issamplemixin__' attribute with value True to the decorated function.

    Examples:
        >>> @samplemixinmethod
        >>> def f():
        ...     pass
        >>> f.__issamplemixin__
        True

    """
    # NB: decorator name was chosen to be consistent with @classmethod and @staticmethod
    method.__issamplemixin__ = True
    return method


class Sampler(metaclass=SamplerABCMeta):
    """Abstract base class for dimod samplers.

    Provides all methods :meth:`~.Sampler.sample`, :meth:`~.Sampler.sample_ising`,
    :meth:`~.Sampler.sample_qubo` assuming at least one is implemented.

    Also includes utility method :meth:`~.Sampler.remove_unknown_kwargs`, which
    may be used in sample methods to handle unknown kwargs.
    """

    @abc.abstractproperty  # for python2 compatibility
    def parameters(self):
        """dict: A dict where keys are the keyword parameters accepted by the sampler
        methods and values are lists of the properties relevent to each parameter.
        """
        pass

    @abc.abstractproperty  # for python2 compatibility
    def properties(self):
        """dict: A dict containing any additional information about the sampler.
        """
        pass

    @samplemixinmethod
    def sample(self, bqm, **parameters):
        """Sample from a binary quadratic model.

        This method is inherited from the :class:`.Sampler` base class.

        Converts the binary quadratic model to either Ising or QUBO format and
        then invokes an implemented sampling method (one of
        :meth:`.sample_ising` or :meth:`.sample_qubo`).

        Args:
            :obj:`.BinaryQuadraticModel`:
                A binary quadratic model.

            **kwargs:
                See the implemented sampling for additional keyword definitions.
                Unknown keywords are accepted but a warning will be raised.

        Returns:
            :obj:`.SampleSet`

        See also:
            :meth:`.sample_ising`, :meth:`.sample_qubo`

        """

        # we try to use the matching sample method if possible
        if bqm.vartype is Vartype.SPIN:
            if not getattr(self.sample_ising, '__issamplemixin__', False):
                # sample_ising is implemented
                h, J, offset = bqm.to_ising()
                sampleset = self.sample_ising(h, J, **parameters)
            else:
                Q, offset = bqm.to_qubo()
                sampleset = self.sample_qubo(Q, **parameters)
        elif bqm.vartype is Vartype.BINARY:
            if not getattr(self.sample_qubo, '__issamplemixin__', False):
                # sample_qubo is implemented
                Q, offset = bqm.to_qubo()
                sampleset = self.sample_qubo(Q, **parameters)
            else:
                h, J, offset = bqm.to_ising()
                sampleset = self.sample_ising(h, J, **parameters)
        else:
            raise RuntimeError("binary quadratic model has an unknown vartype")

        # if the vartype already matches this will just adjust the offset
        return sampleset.change_vartype(bqm.vartype, energy_offset=offset)

    @samplemixinmethod
    def sample_ising(self, h, J, **parameters):
        """Sample from an Ising model using the implemented sample method.

        This method is inherited from the :class:`.Sampler` base class.

        Converts the Ising model into a :obj:`.BinaryQuadraticModel` and then
        calls :meth:`.sample`.

        Args:
            h (dict/list):
                Linear biases of the Ising problem. If a dict, should be of the
                form `{v: bias, ...}` where is a spin-valued variable and `bias`
                is its associated bias. If a list, it is treated as a list of
                biases where the indices are the variable labels.

            J (dict[(variable, variable), bias]):
                Quadratic biases of the Ising problem.

            **kwargs:
                See the implemented sampling for additional keyword definitions.

        Returns:
            :obj:`.SampleSet`

        See also:
            :meth:`.sample`, :meth:`.sample_qubo`

        """
        bqm = BinaryQuadraticModel.from_ising(h, J)
        return self.sample(bqm, **parameters)

    @samplemixinmethod
    def sample_qubo(self, Q, **parameters):
        """Sample from a QUBO using the implemented sample method.

        This method is inherited from the :class:`.Sampler` base class.

        Converts the QUBO into a :obj:`.BinaryQuadraticModel` and then
        calls :meth:`.sample`.

        Args:
            Q (dict):
                Coefficients of a quadratic unconstrained binary optimization
                (QUBO) problem. Should be a dict of the form `{(u, v): bias, ...}`
                where `u`, `v`, are binary-valued variables and `bias` is their
                associated coefficient.

            **kwargs:
                See the implemented sampling for additional keyword definitions.

        Returns:
            :obj:`.SampleSet`

        See also:
            :meth:`.sample`, :meth:`.sample_ising`

        """
        bqm = BinaryQuadraticModel.from_qubo(Q)
        return self.sample(bqm, **parameters)

    def remove_unknown_kwargs(self, **kwargs):
        """Check that all `kwargs` are accepted by the sampler. If a
        keyword is unknown, a warning is raised and the argument is removed.

        Args:
            **kwargs:
                Keyword arguments to be validated.

        Returns:
            dict: Updated `kwargs`

        """
        for kw in [k for k in kwargs if k not in self.parameters]:
            msg = "Ignoring unknown kwarg: {!r}".format(kw)
            warnings.warn(msg, SamplerUnknownArgWarning, stacklevel=3)
            kwargs.pop(kw)

        return kwargs
