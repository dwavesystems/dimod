"""
todo - describe how to use the dimod sampler template
"""
from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.exceptions import InvalidSampler
from dimod.vartypes import Vartype

__all__ = ['Sampler']


class Sampler(object):
    """todo

    """

    def __init__(self):
        self.sample_kwargs = {}
        self.properties = {}
        self._methods_cycled = set()

    def _ensure_finite_cycle(self, methodname):
        """Ensure user-derived sampler implements at least one sampling method."""

        # (In current Sampler implementation, if 3 or more base sample* methods
        # are called, we have an infinite loop. The loop can be broken by
        # overridding at least one sample method in a subclass.)
        self._methods_cycled.add(methodname)
        if len(self._methods_cycled) > 2:
            raise InvalidSampler('Sampler subclass must override at least one '
                                 'of the sampling methods')

    def sample(self, bqm, **sample_kwargs):
        """todo"""
        self._ensure_finite_cycle('sample')
        if bqm.vartype is Vartype.SPIN:
            Q, offset = bqm.to_qubo()
            response = self.sample_qubo(Q, **sample_kwargs)
            response.change_vartype(Vartype.SPIN, offset)
            return response
        elif bqm.vartype is Vartype.BINARY:
            h, J, offset = bqm.to_ising()
            response = self.sample_ising(h, J, **sample_kwargs)
            response.change_vartype(Vartype.BINARY, offset)
            return response
        else:
            raise RuntimeError("binary quadratic model has an unknown vartype")

    def sample_ising(self, h, J, **sample_kwargs):
        """todo"""
        self._ensure_finite_cycle('sample_ising')
        bqm = BinaryQuadraticModel.from_ising(h, J)
        response = self.sample(bqm, **sample_kwargs)
        response.change_vartype(Vartype.SPIN)
        return response

    def sample_qubo(self, Q, **sample_kwargs):
        """todo"""
        self._ensure_finite_cycle('sample_qubo')
        bqm = BinaryQuadraticModel.from_qubo(Q)
        response = self.sample(bqm, **sample_kwargs)
        response.change_vartype(Vartype.BINARY)
        return response
