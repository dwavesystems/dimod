"""
dimod Composites
================

Samplers can be composed. This pattern allows pre- and post-processing
to be applied to binary quadratic programs without needing to change
the underlying sampler implementation.

We refer to these layers as `composites`. Each composed sampler must
include at least one `sampler`, and possibly many composites. See
'dimod sampler composition pattern' figure below.

Each composed sampler is itself a dimod sampler with all of the
included methods and parameters. In this way complex samplers
can be constructed.


Examples
--------

We will be using the included composite :class:`.SpinReversalTransform`
and included samplers :class:`.ExactSolver` and
:class:`.SimulatedAnnealingSampler` in our examples.

Building composed samplers are easy

>>> composed_sampler_es = dimod.SpinReversalTransform(dimod.ExactSolver())

A composite layer can be applied to any dimod sampler.

>>> composed_sampler_sa = dimod.SpinReversalTransform(dimod.SimulatedAnnealingSampler)

These composed samplers themselves behave exactly like samplers.

>>> h = {0: -1, 1: 1}
>>> response = composed_sampler_es.sample_ising(h, {})
>>> list(response.samples())
[{0: 1, 1: -1}, {0: -1, 1: -1}, {0: 1, 1: 1}, {0: -1, 1: 1}]

Composite layers can also be nested.

>>> composed_sampler_nested = dimod.SpinReversalTransform(composed_sampler_es)

"""
from dimod.core.sampler import Sampler

__all__ = ['Composite']


class Composite():
    def __init__(self, child, added_kwargs={}, removed_kwargs={}):
        self.child = child
        self.sample_kwargs = child.sample_kwargs.copy()
        self.sample_kwargs.update(added_kwargs)
        for kwarg in removed_kwargs:
            if kwarg in self.sample_kwargs:
                del self.sample_kwargs[kwarg]
