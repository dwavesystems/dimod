from dimod.classes.sampler import Sampler
from dimod.classes.composite import Composite
from dimod.classes.structured import Structured
from dimod.decorators import bqm_structured


class StructureComposite(Sampler, Composite, Structured):
    """Creates a structured composed sampler from an unstructured sampler.

    todo
    """
    def __init__(self, sampler, nodelist, edgelist):
        Sampler.__init__(self)
        Composite.__init__(self, sampler)
        Structured.__init__(self, nodelist, edgelist)

    @bqm_structured
    def sample(self, bqm, **sample_kwargs):
        return self.child.sample(bqm, **sample_kwargs)
