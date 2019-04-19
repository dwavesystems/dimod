"""
The D-Wave machine exhibits intrinsic control errors such that the effective
values of the final Hamiltonian coefficients may deviate slightly from their
programmed values.
A gauge transformation on the qubits induces a transformation on the Ising
model coefficients :math: `H_j` and :math:`J_{ij}`. One interesting property
of gauge transformations is that they transforms ferromagnetic couplings into
antiferromagnetic couplings, and vice versa. This is especially useful in
mitigating the coupler/bias leakage effect in chained qubits.
In the checkerboard transformation, the qubits on one shore of each tile are
flipped while the qubits on the other shore are unchanged. Because of the way
this pattern alternates from one unit cell to the next, this is called a
basket weave or checkerboard gauge transformation.

NOTE: To be used before wrapping the composite with EmbeddingComposite.

WARNING!: If num_reads is given in the keyed arguments (**kwargs), this is
passed unmodified to the Structured Sampler for every gauge.
i.e. If num_reads=1000 then there will be 4000 samples.

The gauge transformations are performed sequentially:
     I: Identity transformation     (no spins flipped)
     G: Checkered transformation    (spins flipped starting with shore 0)
    -I: Inverse transformation      (spins flipped starting with shore 1)
    -G: Checkered transformation    (spins flipped starting with shore 0)
"""

from dwave_networkx.generators import chimera, pegasus

from dimod.core.composite import Composite
from dimod.core.sampler import Sampler
from dimod.core.structured import Structured
from dimod.sampleset import SampleSet, concatenate
from dimod.vartypes import Vartype
from dimod.exceptions import InvalidComposition

__all__ = ['CheckerboardTransformComposite']

_PLOT = True

class CheckerboardTransformComposite(Sampler, Composite, Structured):
    """Composite for applying a checkerboard spin reversal transform.

    The Checkerboard Spin reversal transform [#ah]_ is applied by flipping the
    spin of variables on one of the shores of each tile in the qubit target
    graph. After sampling the transformed Ising problem, the same bits are
    flipped in the resulting sample [#km]_.

    Args:
        sampler: A `dimod` sampler object.

    References
    ----------
    .. [#ah] Adachi, S. H., & Henderson, M. P. Application of Quantum Annealing
        to Training of Deep Neural Networks. https://arxiv.org/abs/1510.06356,
        2015.
    .. [#km] Andrew D. King and Catherine C. McGeoch. Algorithm engineering
        for a quantum annealing platform. https://arxiv.org/abs/1410.2628,
        2014.

    """
    children = None
    nodelist = None
    edgelist = None
    parameters = None
    properties = None
    aggregate = None

    def __init__(self, child, target_graph_dnx=None, aggregate=False):
        self.children = [child]
        self.nodelist = child.nodelist
        self.edgelist = child.edgelist
        self.parameters = child.parameters
        self.aggregate = aggregate

        if not isinstance(child, Structured):
            raise InvalidComposition("Checkerboard transformations should only be applied to a Structured sampler")

        child.properties['graph'] = target_graph_dnx.graph
        self.properties = {"child_properties": child.properties}

        # Parses dwave_networkx target graph.
        # NOTE: A better way would be to just get that same dict ('graph') from
        # child_properties of a Structured Composite. i.e:
        # self.coordinates = coordinates(**child.properties['graph'])

        self.coordinates = coordinates(**target_graph_dnx.graph)

    def sample(self, bqm, **kwargs):
        """Sample from the binary quadratic model.

        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

        Returns:
            :obj:`.SampleSet`

        Examples:
            This example runs the checkerboard transformation on a Chimera
            graph and retrieves 1000*4 samples. 1000 for each gauge.

            >>> import dimod
            >>> import dwave_networkx as dnx
            >>> C = dnx.chimera_graph(4)
            >>> nodelist = list(C.nodes())
            >>> edgelist = list(C.edges())
            >>> structsampler = dimod.StructureComposite(dimod.RandomSampler(),
            ...                     nodelist=nodelist, edgelist=edgelist)
            ...
            >>> Q = {(v,v):0.1 for v in nodelist}
            >>> Q.update( {edge:-1.0 for edge in edgelist} )
            ...
            >>> sampler = dimod.CheckerboardTransformComposite(structsampler, C)
            >>> response = sampler.sample_qubo(Q, num_reads=1000)

        """
        # Make a main response
        responses = []

        flipped_bqm = bqm.copy()
        transform = {v: False for v in bqm.variables}

        # Identity transformation (no spins flipped)
        response = self.child.sample(bqm, **kwargs)
        responses.append(response)

        #### Alternate flipping shore 0 and 1 for even and odd tiles
        # Checkered transformation    (spins flipped starting with shore 0)
        # Inverse transformation      (spins flipped starting with shore 1)
        # Checkered transformation    (spins flipped starting with shore 0)
        for flip_even in [0, 1, 0]:
            # Create flipped BQM
            for v in bqm:
                u = self.coordinates.get_shore(v)
                tile = self.coordinates.get_tile(v)
                is_even_tile = not sum(tile)%2

                flip = u==flip_even if is_even_tile else u!=flip_even

                if flip:
                    transform[v] = not transform[v]
                    flipped_bqm.flip_variable(v)

            # Sample flipped BQM
            flipped_response = self.child.sample(flipped_bqm, **kwargs)
            tf_idxs = [flipped_response.variables.index(v) for v, flip in transform.items() if flip]

            if bqm.vartype is Vartype.SPIN:
                flipped_response.record.sample[:, tf_idxs] = -1 * flipped_response.record.sample[:, tf_idxs]
            else:
                flipped_response.record.sample[:, tf_idxs] = 1 - flipped_response.record.sample[:, tf_idxs]

            responses.append(flipped_response)

        # Merge all gauge transformation responses
        if self.aggregate:
            return concatenate(responses).aggregate()
        else:
            return concatenate(responses)


class coordinates:
    def __init__(self, **graph):
        """
        Provides a coordinate converter that is architecture-agnostic.

        Parameter
        ---------
        graph : dict
            family : | chimera | pegasus |
            columns : int
            rows : int
            tile : tuple
            labels : | int | coordinates | nice |
        """

        try:
            family = graph["family"]
            m = graph["columns"]
            n = graph["rows"]
            t = graph["tile"]
            labels = graph["labels"]
        except:
            raise ValueError("Target graph needs to have family, columns, \
            rows, tile, and labels attributes.")

        if family=="chimera":
            self._converter = chimera.chimera_coordinates(m,n=n,t=t)
            if labels=="int":
                self._int_getter = lambda v : v
                self._tuple_getter = self._converter.tuple
                self._shore_getter = lambda v: self._tuple_getter(v)[2]
                self._tile_getter = lambda v: self._tuple_getter(v)[:2]
            elif labels=="coordinate":
                self._int_getter = lambda *v : self._converter.int(v)
                self._tuple_getter = lambda *v : v
                self._shore_getter = lambda i, j, u, k: u
                self._tile_getter = lambda i, j, u, k: (i, j)

        elif family=="pegasus":
            self._converter = pegasus.pegasus_coordinates(m)
            if labels=="int":
                self._int_getter = lambda v : v
                self._tuple_getter = self._converter.tuple
                self._shore_getter = lambda v: self._tuple_getter(v)[0]
                p2n = pegasus.get_pegasus_to_nice_fn()
                self._tile_getter = lambda v: p2n(*self._tuple_getter(v))[:3]
            elif labels=="coordinate":
                self._int_getter = lambda *v : self._converter.int(v)
                self._tuple_getter = lambda *v : v
                self._shore_getter = lambda u, w, k, z : u
                p2n = pegasus.get_pegasus_to_nice_fn()
                self._tile_getter = lambda u, w, k, z : p2n(u, w, k, z)[:3]
            elif labels=="nice":
                n2p = pegasus.get_nice_to_pegasus_fn()
                self._int_getter = lambda *v : self._converter.int((n2p(*v)))
                self._tuple_getter = lambda *v : v
                self._shore_getter = lambda t, i, j, u, k: u
                self._tile_getter = lambda t, i, j, u, k : (t, i, j)

    def get_int(self, *v):
        return self._int_getter(*v)

    def get_tuple(self, *v):
        return self._tuple_getter(*v)

    def get_shore(self, *v):
        return self._shore_getter(*v)

    def get_tile(self, *v):
        return self._tile_getter(*v)
