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
import abc

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
    @abc.abstractmethod
    def sample_poly(self, polynomial, **kwargs):
        """Sample from a higher-order polynomial."""
        pass

    def sample_hising(self, h, J, **kwargs):
        return self.sample_poly(BinaryPolynomial.from_hising(h, J), **kwargs)

    def sample_hubo(self, H, **kwargs):
        return self.sample_poly(BinaryPolynomial.from_hubo(H), **kwargs)


class ComposedPolySampler(PolySampler, Composite):
    pass
