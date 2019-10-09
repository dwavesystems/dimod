from dimod.core.composite import ComposedSampler
from dimod.sampleset import SampleSet
from dimod.traversal import connected_components
import dwave_networkx
import dimod
from dimod import ExactSolver
import networkx as nx
import itertools
from collections import deque

def sample(self, bqm, components=None, **parameters):
    """Sample from the provided binary quadratic model.

    Args:
        bqm (:obj:`dimod.BinaryQuadraticModel`):
            Binary quadratic model to be sampled from.

        components (list(set)):
            A list of disjoint set of variables that fully partition the variables

        **parameters:
            Parameters for the sampling method, specified by the child sampler.

    Returns:
        :obj:`dimod.SampleSet`

    """

    # solve the problem on the child system
    child = self.child
    variables = bqm.variables
    if components is None:
        components = list(connected_components(bqm))
    if isinstance(components, set):
        components = [components]
    sampleset = None
    for component in components:
        bqm_copy = bqm.copy()
        bqm_copy.fix_variables({i: 0 for i in (variables - component)})
        if sampleset is None:
            # here .truncate(1) is used to pick the best solution only. The other options
            # for future development is to combine all sample with all.
            # This way you'd get the same behaviour as the ExactSolver
            sampleset = child.sample(bqm_copy, **parameters).truncate(1)
        else:
            sampleset = sampleset.truncate(1).append_variables(child.sample(bqm_copy, **parameters).truncate(1))

    if sampleset is None:
        return SampleSet.from_samples_bqm({}, bqm)
    else:
        return SampleSet.from_samples_bqm(sampleset, bqm)


def build_biconnected_graph(bcc, cut_vertices):
    bcc = [tuple(c) for c  in bcc]
    T = nx.Graph()
    for c in bcc:
        T.add_node(c, cuts = [v for v in c if v in cut_vertices])
    for (c1, c2) in itertools.combinations(bcc, 2):
        cv = list(set(c1).intersection(set(c2)))
        if len(cv) > 0:
            T.add_edge(c1, c2, cut=cv[0])
    return T

if __name__ == '__main__':
    comps =  [[0,1,2], [2,3,4], [3,5,6], [4,7,8]]
    J = {(u, v): 1  for c in comps for (u, v) in itertools.combinations(c, 2)}
    bqm = dimod.BinaryQuadraticModel.from_ising({}, J)
    sampler = ExactSolver()
    response = sampler.sample(bqm)
    print(response.first.energy)


    G = bqm.to_networkx_graph()
    cut_vertices = list(nx.articulation_points(G))
    print(list(cut_vertices))
    bcc = list(nx.biconnected_components(G))
    print(list(bcc))

    # build the graph structure (make this faster at some point)
    T = build_biconnected_graph(bcc, cut_vertices)
    print(T.nodes(data=True))
    print(T.edges(data=True))

    # work up the edges of the tree from leafs to root, sampling at each step. Record energy and best state in each
    # bcc for each cut vertex configuration. From root, propagate solutions back down the tree.
    #source = T.nodes()[0]
    print(list(nx.dfs_postorder_nodes(T)))
    #for bcc in nx.dfs_postorder_nodes(T):




