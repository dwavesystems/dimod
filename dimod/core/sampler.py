"""
We expect developers will wish to create new dimod samplers. Here we will provide an example of how
to do so.

Imagine that we have a function which returns the sample which minimizes the linear terms of the
Ising problem.

.. code-block:: python

    def linear_ising(h, J):
        sample = {}
        for v in h:
            if h[v] < 0:
                sample[v] = +1
            else:
                sample[v] = -1
        return sample

We decide that this function is useful enough that we wish to create a dimod sampler from it. This
can be done simply by using the :class:`.Sampler` abstract base class (ABC_).

.. _ABC: https://docs.python.org/3.6/library/abc.html#module-abc

.. code-block:: python

    class LinearIsingSampler(dimod.Sampler):

        def sample_ising(self, h, J):
            sample = linear_ising(h, J)
            energy = dimod.ising_energy(sample, h, J)
            return dimod.Response.from_dicts([sample], {'energy': [energy]})

        @property
        def properties(self):
            return dict()

        @property
        def parameters(self):
            return dict()

We know we need to implement :meth:`sample_ising`, :attr:`properties` and :attr:`parameters` by
consulting the above table. The advantage of using the :class:`.Sampler` is that we get the other
sample methods 'for free' as mixins.

>>> sampler = LinearIsingSampler()
>>> response = sampler.sample_ising({'a': -1}, {})  # implemented
>>> response = sampler.sample_qubo({('a', 'a'): 1})  # mixin
>>> response = sampler.sample(BinaryQuadraticModel.from_ising({'a': -1}, {}))  # mixin

In this case, because the sampler is so simple, we chose to have both :attr:`properties` and
:attr:`parameters` return empty dicts, but we could instantiate a more complex version.

.. code-block:: python

    class FancyLinearIsingSampler(dimod.Sampler):
        def __init__(self):
            self._properties = {'description': 'a simple sampler that only considers the linear terms'}
            self._parameters = {'verbose': []}

        def sample_ising(self, h, J, verbose=False):
            sample = linear_ising(h, J)
            energy = dimod.ising_energy(sample, h, J)
            if verbose:
                print(sample)
            return dimod.Response.from_dicts([sample], {'energy': [energy]})

        @property
        def properties(self):
            return self._properties

        @property
        def parameters(self):
            return self._parameters


"""
from six import add_metaclass

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.exceptions import InvalidSampler
from dimod.vartypes import Vartype

import dimod.abc as abc

__all__ = ['Sampler']


@add_metaclass(abc.SamplerABCMeta)
class Sampler:
    """The abstract base class for dimod Samplers.

    Provides the method :meth:`~.Sampler.sample`, :meth:`~.Sampler.sample_ising`,
    :meth:`~.Sampler.sample_qubo` assuming that one has been implemented.

    """

    @abc.abstractproperty  # for python2 compatibility
    def parameters(self):
        """dict: Should be a dict where the keys are the keyword parameters accepted by the sample
        methods. The values should be lists of the properties relevent to the parameter.
        """
        pass

    @abc.abstractproperty  # for python2 compatibility
    def properties(self):
        """dict: Should be a dict containing any additional information about the sampler."""
        pass

    @abc.samplemixinmethod
    def sample(self, bqm, **parameters):
        """Samples from the given bqm using the instantiated sample method."""
        if bqm.vartype is Vartype.SPIN:
            Q, offset = bqm.to_qubo()
            response = self.sample_qubo(Q, **parameters)
            response.change_vartype(Vartype.SPIN, offset)
            return response
        elif bqm.vartype is Vartype.BINARY:
            h, J, offset = bqm.to_ising()
            response = self.sample_ising(h, J, **parameters)
            response.change_vartype(Vartype.BINARY, offset)
            return response
        else:
            raise RuntimeError("binary quadratic model has an unknown vartype")

    @abc.samplemixinmethod
    def sample_ising(self, h, J, **parameters):
        """Samples from the given Ising model using the instantiated sample method."""
        bqm = BinaryQuadraticModel.from_ising(h, J)
        response = self.sample(bqm, **parameters)
        response.change_vartype(Vartype.SPIN)
        return response

    @abc.samplemixinmethod
    def sample_qubo(self, Q, **parameters):
        """Samples from the given QUBO using the instantiated sample method."""
        bqm = BinaryQuadraticModel.from_qubo(Q)
        response = self.sample(bqm, **parameters)
        response.change_vartype(Vartype.BINARY)
        return response
