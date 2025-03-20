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
Samplers can be composed. The `composite pattern <https://en.wikipedia.org/wiki/Composite_pattern>`_
allows layers of pre- and post-processing to be applied to binary quadratic programs without needing
to change the underlying sampler implementation.

We refer to these layers as `composites`. Each composed sampler must
include at least one sampler, and possibly many composites.

Each composed sampler is itself a dimod sampler with all of the
included methods and parameters. In this way complex samplers
can be constructed.

The dimod :class:`.ComposedSampler` abstract base class inherits from :class:`.Sampler` class
its abstract methods, properties, and mixins (for example, a `sample_Ising` method) and from
:class:`.Composite` class the `children` property and `child` mixin (`children` being a list of
supported samplers with `child` providing the first).

Examples:
    The :class:`~dimod.reference.composites.higherordercomposites.HigherOrderComposite`
    converts a binary quadratic model sampler to a binary polynomial sampler.
    Given dimod sampler :class:`.ExactSolver` for example, the composed sampler is
    used as any dimod sampler:

    >>> sampler = dimod.ExactSolver()
    >>> composed_sampler = dimod.HigherOrderComposite(sampler)
    >>> J = {("a", "b", "c"): 1}
    >>> sampleset = composed_sampler.sample_hising({}, J)
    >>> set(sampleset.first.sample.values()) == {-1}
    True

    For more examples, see the source code for the composed samplers
    documented in the :ref:`dimod_composites` section.
"""
import abc

from dimod.core.sampler import Sampler
from dimod.core.scoped import Scoped

__all__ = ['Composite', 'ComposedSampler']


class Composite(Scoped):
    """Abstract base class for dimod composites.

    Provides the :attr:`Composite.child` mixin property and defines the :attr:`Composite.children`
    abstract property to be implemented. These define the supported samplers for the composed sampler.

    :class:`.Composite` implements the :class:`~dimod.core.scoped.Scoped` interface, and
    provides :meth:`.close` method that closes all child samplers or composites.

    .. versionchanged:: 0.12.19
        Implemented the :class:`~dimod.core.scoped.Scoped` interface. Now all
        composites support context manager protocol and release scope-based
        resources of sub-samplers/composites by default.
    """

    @abc.abstractproperty
    def children(self):
        """list[ :obj:`.Sampler`]: List of child samplers that that are used by
        this composite.
        """
        pass

    @property
    def child(self):
        """:obj:`.Sampler`: The child sampler. First sampler in :attr:`Composite.children`."""
        try:
            return self.children[0]
        except IndexError:
            raise RuntimeError("A Composite must have at least one child Sampler")

    def close(self):
        """Release any scope-bound resources of child samplers or composites.

        .. note::
            If a :class:`.Composite` subclass doesn't allocate resources that have
            to be explicitly released, there's no need to override the default
            :meth:`~.Composite.close` implementation.

            However, if you do implement :meth:`~.Composite.close` on a subclass,
            make sure to either call ``super().close()``, or to explicitly close
            all child samplers/composites.

        .. versionadded:: 0.12.19
            :class:`.Composite` now implements the :class:`~dimod.core.scoped.Scoped`
            interface. The default :meth:`~.Composite.close` method recursively closes
            composite's children.
        """

        for child in self.children:
            if hasattr(child, 'close') and callable(child.close):
                child.close()
        super().close()


class ComposedSampler(Composite, Sampler):
    """Abstract base class for dimod composed samplers.

    Inherits from :class:`.Sampler` and :class:`.Composite`.

    """
    pass
