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
    The dimod package's spin_transform.py reference example creates a composed
    sampler, `SpinReversalTransformComposite(Sampler, Composite)`, that performs
    spin reversal transforms ("gauge transformations") as a preprocessing step for
    a given sampler. The reference example implements the pseudocode below:

    .. code-block:: python

        class SpinReversalTransformComposite(Sampler, Composite):

            # Updates to inherited sampler properties and parameters
            # Definition of the composite's children (i.e., supported samplers):
            children = None
            def __init__(self, child):
                self.children = [child]

            # The composite's implementation of spin-transformation functionality:
            def sample(self, bqm, num_spin_reversal_transforms=2, spin_reversal_variables=None, **kwargs):
                response = None
                # Preprocessing code that includes instantiation of a sampler:
                # flipped_response = self.child.sample(bqm, **kwargs)
                return response

    Given a sampler, `sampler1`, the composed sampler is used as any dimod sampler.
    For example, the composed sampler inherits an Ising sampling method:

    >>> composed_sampler = dimod.SpinReversalTransformComposite(sampler1) # doctest: +SKIP
    >>> h = {0: -1, 1: 1} # doctest: +SKIP
    >>> response = composed_sampler.sample_ising(h, {}) # doctest: +SKIP

"""
from six import add_metaclass

from dimod.core.sampler import Sampler

import dimod.abc as abc

__all__ = ['Composite', 'ComposedSampler']


@add_metaclass(abc.ABCMeta)
class Composite:
    """Abstract base class for dimod composites.

    Provides the :attr:`~.Composite.children` property and :attr:`~.Composite.child` mixin property
    that define the supported samplers for the composed sampler.

    """
    @abc.abstractproperty
    def children(self):
        """list[ :obj:`.Sampler`]: List of samplers (or composed samplers).

        This abstract property must be implemented.


        Examples:
            These examples define the supported samplers for the composed sampler upon instantiation.

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
        """First child in :attr:`~.Composite.children`.

        Examples:
            This example pseudocode defines a composed sampler that uses the first supported
            sampler in a composite's list of samplers on a binary quadratic model.

            .. code-block:: python

                class MyComposedSampler(Sampler, Composite):

                    # Updates to inherited sampler properties and parameters
                    # Definition of the composite's children (i.e., supported samplers)

                    # Implementation of the composite's functionality
                    def processed_sample(self, bqm, relevant_arguments, **kwargs):
                        response = None
                        # instantiation of a sampler:
                        # sampler_response = self.child.sample(bqm, **kwargs)
                        # Code for composed processing of samples
                        return response

        """
        try:
            return self.children[0]
        except IndexError:
            raise RuntimeError("A Composite must have at least one child Sampler")


class ComposedSampler(Sampler, Composite):
    """Abstract base class for dimod composed samplers.

    Inherits from :class:`.Sampler` and :class:`.Composite`.

    """
    pass
