from dimod.core.sampler import Sampler
from dimod.core.composite import Composite
from dimod.core.structured import Structured
from dimod.decorators import bqm_structured


class StructureComposite(Sampler, Composite, Structured):
    """Creates a structured composed sampler from an unstructured sampler.
    """
    # we will override these in the __init__, but because they are abstract properties we need to
    # signal that we are overriding them
    edgelist = None
    nodelist = None
    children = None

    def __init__(self, sampler, nodelist, edgelist):
        self.children = [sampler]
        self.nodelist = nodelist
        self.edgelist = edgelist

    @property
    def parameters(self):
        return self.child.parameters

    @property
    def properties(self):
        return self.child.properties

    @bqm_structured
    def sample(self, bqm, **sample_kwargs):
        return self.child.sample(bqm, **sample_kwargs)
