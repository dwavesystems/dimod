"""
Samplers can be composed. The composite pattern_ allows pre- and post-processing
to be applied to binary quadratic programs without needing to change
the underlying sampler implementation.

.. _pattern: https://en.wikipedia.org/wiki/Composite_pattern

We refer to these layers as `composites`. Each composed sampler must
include at least one `sampler`, and possibly many composites. See
'dimod sampler composition pattern' figure below.

Each composed sampler is itself a dimod sampler with all of the
included methods and parameters. In this way complex samplers
can be constructed.

Because an instantiated composite with its children is a 'composed sampler', dimod includes the
:class:`.ComposedSampler` abstract base class, which inherits its abstract methods and properties
as well as its mixins from :class:`.Sampler` and :class:`.Composite`.

"""
from six import add_metaclass

from dimod.core.sampler import Sampler

import dimod.abc as abc

__all__ = ['Composite', 'ComposedSampler']


@add_metaclass(abc.ABCMeta)
class Composite:
    """The abstract base class for dimod Composites.

    Provies the :attr:`~.Composite.child` mixin property.

    """
    @abc.abstractproperty
    def children(self):
        """list[ :obj:`.Sampler`]: Should be a list of samplers (or composed samplers).

        This abstract property must be implemented.


        Examples:
            .. code-block:: python

                class MyComposite(dimod.Composite):
                    def __init__(self, *children):
                        self._children = list(children)

                    @property
                    def children(self):
                        return self._children

            .. code-block:: python

                class AnotherComposite(dimod.Composite):
                    self.children = None
                    def __init__(self, child_sampler):
                        self.children = [child_sampler]
        """
        pass

    @property
    def child(self):
        """The first child in :attr:`~.Composite.children`."""
        try:
            return self.children[0]
        except IndexError:
            raise RuntimeError("A Composite must have at least one child Sampler")


class ComposedSampler(Sampler, Composite):
    """Inherits from :class:`.Sampler` and :class:`.Composite`."""
    pass
