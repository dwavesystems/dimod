"""

.. list-table::
    :header-rows: 1

    *   - ABC
        - Inherits from
        - Abstract Properties
        - Abstract Methods
        - Mixins
    *   - :class:`.Sampler`
        -
        - :attr:`~Sampler.parameters`, :attr:`~Sampler.properties`
        - one of
          :meth:`~.Sampler.sample`, :meth:`~.Sampler.sample_ising`, :meth:`~.Sampler.sample_qubo`
        - :meth:`~.Sampler.sample`, :meth:`~.Sampler.sample_ising`, :meth:`~.Sampler.sample_qubo`


Creating a dimod Sampler
========================

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

.. code-block:: python
    :linenos:

    class LinearIsingSampler(dimod.Sampler):
        @property
        def properties(self):
            return dict()

        @property
        def parameters(self):
            return dict()

        def sample_ising(self, h, J):
            sample = linear_ising(h, J)
            energy = dimod.ising_energy(sample, h, J)
            return dimod.Response.from_dicts([sample], {'energy': [energy]})

Now


.. _ABC: https://docs.python.org/3.6/library/abc.html#module-abc

"""
from dimod.binary_quadratic_model_convert import to_qubo, to_ising, from_qubo, from_ising
from dimod.compatibility23 import add_metaclass
from dimod.exceptions import InvalidSampler
from dimod.vartypes import Vartype

import dimod.abc as abc

__all__ = ['Sampler']


@add_metaclass(abc.SamplerABCMeta)
class Sampler:
    """The abstract base class for dimod Samplers.

    Provides the method :meth:`~.Sampler.sample`, :meth:`~.Sampler.sample_ising`,
    :meth:`~.Sampler.sample_qubo` assuming that one has been overwritten.

    """

    @abc.abstractproperty  # for python2 compatibility
    def parameters(self):
        pass

    @abc.abstractproperty  # for python2 compatibility
    def properties(self):
        pass

    @abc.samplemixinmethod
    def sample(self, bqm, **parameters):
        """todo"""
        # self._ensure_finite_cycle('sample')
        if bqm.vartype is Vartype.SPIN:
            Q, offset = to_qubo(bqm)
            response = self.sample_qubo(Q, **parameters)
            response.change_vartype(Vartype.SPIN, offset)
            return response
        elif bqm.vartype is Vartype.BINARY:
            h, J, offset = to_ising(bqm)
            response = self.sample_ising(h, J, **parameters)
            response.change_vartype(Vartype.BINARY, offset)
            return response
        else:
            raise RuntimeError("binary quadratic model has an unknown vartype")

    @abc.samplemixinmethod
    def sample_ising(self, h, J, **parameters):
        """todo"""
        # self._ensure_finite_cycle('sample_ising')
        bqm = from_ising(h, J)
        response = self.sample(bqm, **parameters)
        response.change_vartype(Vartype.SPIN)
        return response

    @abc.samplemixinmethod
    def sample_qubo(self, Q, **parameters):
        """todo"""
        # self._ensure_finite_cycle('sample_qubo')
        bqm = from_qubo(Q)
        response = self.sample(bqm, **parameters)
        response.change_vartype(Vartype.BINARY)
        return response
