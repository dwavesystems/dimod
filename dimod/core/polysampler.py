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

"""
Samplers that handle binary polynomials: problems with binary variables that are
not constrained to quadratic interactions.

"""

import abc
import collections.abc

from dimod.core.composite import Composite
from dimod.core.scoped import Scoped
from dimod.higherorder.polynomial import BinaryPolynomial
from dimod.sampleset import SampleSet

from dimod.typing import Bias, Variable

__all__ = 'PolySampler', 'ComposedPolySampler'


class PolySampler(Scoped):
    """Sampler that supports binary polynomials.

    Binary polynomials are an extension of binary quadratic models that allow
    higher-order interactions.

    .. versionchanged:: 0.12.19
        :class:`.PolySampler` now implements the :class:`~dimod.core.scoped.Scoped`
        interface, so it supports context manager protocol by default.

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

    def sample_hising(
            self,
            h: collections.abc.Mapping[Variable, Bias],
            J: collections.abc.Mapping[tuple[Variable, Variable], Bias],
            **kwargs,
            ) -> SampleSet:
        """Sample from a higher-order Ising model.

        Converts the given higher-order Ising model to a :obj:`.BinaryPolynomial`
        and calls :meth:`.sample_poly`.

        Args:
            h: Variable biases of the Ising problem.
            J: Interaction biases of the Ising problem.

            **kwargs:
                See :meth:`.sample_poly` for additional keyword definitions.

        Returns:
            Samples from the higher-order Ising model.

        See also:
            :meth:`.sample_poly`, :meth:`.sample_hubo`

        """
        return self.sample_poly(BinaryPolynomial.from_hising(h, J), **kwargs)

    def sample_hubo(self, H: collections.abc.Mapping[tuple[Variable, Variable], Bias],
                    **kwargs) -> SampleSet:
        """Sample from a higher-order unconstrained binary optimization problem.

        Converts the given higher-order unconstrained binary optimization
        problem to a :obj:`.BinaryPolynomial` and then calls :meth:`.sample_poly`.

        Args:
            H: Coefficients of the HUBO.

            **kwargs:
                See :meth:`.sample_poly` for additional keyword definitions.

        Returns:
            Samples from a higher-order unconstrained binary optimization problem.

        See also:
            :meth:`.sample_poly`, :meth:`.sample_hising`

        """
        return self.sample_poly(BinaryPolynomial.from_hubo(H), **kwargs)

    def close(self):
        """Release allocated resources.

        Override to release sampler-allocated resources.
        """
        pass


class ComposedPolySampler(Composite, PolySampler):
    """Abstract base class for dimod composed polynomial samplers.

    Inherits from :class:`.PolySampler` and :class:`.Composite`."""
    pass
