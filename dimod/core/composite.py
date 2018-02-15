"""

todo - describe how to use the dimod composite template

Samplers can be composed. This pattern allows pre- and post-processing
to be applied to binary quadratic programs without needing to change
the underlying sampler implementation.

We refer to these layers as `composites`. Each composed sampler must
include at least one `sampler`, and possibly many composites. See
'dimod sampler composition pattern' figure below.

Each composed sampler is itself a dimod sampler with all of the
included methods and parameters. In this way complex samplers
can be constructed.

"""
from dimod.core.sampler import Sampler

__all__ = ['Composite']


class Composite(object):
    """todo"""
    def __init__(self, child, added_kwargs=None, removed_kwargs=None):
        if added_kwargs is None:
            added_kwargs = {}
        if removed_kwargs is None:
            removed_kwargs = {}
        self.child = child
        self.sample_kwargs = child.sample_kwargs.copy()
        self.sample_kwargs.update(added_kwargs)
        for kwarg in removed_kwargs:
            if kwarg in self.sample_kwargs:
                del self.sample_kwargs[kwarg]
