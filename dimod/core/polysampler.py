# Copyright 2019 D-Wave Systems Inc.
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
#
# ============================================================================
"""
It is possible to construct samplers that handle binary polynomials - problems
that have binary variables but they are not constrained to quadratic
interactions.

"""
import abc
import warnings

from six import add_metaclass

from dimod.core.composite import Composite
from dimod.higherorder.polynomial import BinaryPolynomial

__all__ = 'PolySampler', 'ComposedPolySampler'


@add_metaclass(abc.ABCMeta)
class PolySampler:
    """Sampler supports binary polynomials.

    Binary polynomials are an extension of binary quadratic models that allow
    higher-order interactions.

    """
    @abc.abstractproperty  # for python2 compatibility
    def parameters(self):
        """dict: A dict where keys are the keyword parameters accepted by the sampler
        methods and values are lists of the properties relevant to each parameter.
        """
        pass

    @abc.abstractproperty  # for python2 compatibility
    def properties(self):
        """dict: A dict containing any additional information about the sampler.
        """
        pass

    @abc.abstractmethod
    def sample_poly(self, polynomial, **kwargs):
        """Sample from a higher-order polynomial."""
        pass

    def sample_hising(self, h, J, **kwargs):
        """Sample from a higher-order Ising model.

        Convert the given higher-order Ising model to a :obj:`.BinaryPolynomial`
        and call :meth:`.sample_poly`.

        Args:
            h (dict):
                The variable biases of the Ising problem. Should be a dict of
                the form `{v: bias, ...}` where `v` is a variable in the
                polynomial and `bias` is its associated coefficient.

            J (dict):
                The interaction biases of the Ising problem. Should be a dict of
                the form `{(u, v, ...): bias}` where `u`, `v`, are spin-valued
                variables in the polynomial and `bias` is their associated
                coefficient.

            **kwargs:
                See :meth:`.sample_poly` for additional keyword definitions.

        Returns:
            :obj:`.SampleSet`

        See also:
            :meth:`.sample_poly`, :meth:`.sample_hubo`

        """
        return self.sample_poly(BinaryPolynomial.from_hising(h, J), **kwargs)

    def sample_hubo(self, H, **kwargs):
        """Sample from a higher-order unconstrained binary optimization problem.

        Convert the given higher-order unconstrained binary optimization
        problem to a :obj:`.BinaryPolynomial` and then call
        :meth:`.sample_poly`.

        Args:
            H (dict):
                The coefficients of the HUBO. Should be a dict of the form
                {(u, v, ...): bias, ...} where `u`, `v`, are binary-valued
                variables in the polynomial and `bias` is their associated
                coefficient.

            **kwargs:
                See :meth:`.sample_poly` for additional keyword definitions.

        Returns:
            :obj:`.SampleSet`

        See also:
            :meth:`.sample_poly`, :meth:`.sample_hising`

        """
        return self.sample_poly(BinaryPolynomial.from_hubo(H), **kwargs)

    def sample(self, bqm, *args, **kwargs):
        msg = ("PolySampler.sample is deprecated and will be removed in dimod "
               "0.9.0. In the future, when using PolySamplers, you should use "
               ".sample_poly")
        warnings.warn(msg, DeprecationWarning)
        poly = BinaryPolynomial(bqm.quadratic)
        poly.update(((v,), bias) for v, bias in bqm.linear.items())
        return self.sample_poly(poly, *args, **kwargs)

    def sample_ising(self, *args, **kwargs):
        msg = ("PolySampler.sample_ising is deprecated and will be removed in dimod "
               "0.9.0. In the future, when using PolySamplers, you should use "
               ".sample_hising")
        warnings.warn(msg, DeprecationWarning)
        return self.sample_hising(*args, **kwargs)

    def sample_qubo(self, *args, **kwargs):
        msg = ("PolySampler.sample_qubo is deprecated and will be removed in dimod "
               "0.9.0. In the future, when using PolySamplers, you should use "
               ".sample_hubo")
        warnings.warn(msg, DeprecationWarning)
        return self.sample_hubo(*args, **kwargs)


class ComposedPolySampler(PolySampler, Composite):
    """Abstract base class for dimod composed polynomial samplers.

    Inherits from :class:`.PolySampler` and :class:`.Composite`."""
    pass
