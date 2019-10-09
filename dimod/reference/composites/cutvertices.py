from dimod.core.composite import ComposedSampler
from dimod.sampleset import SampleSet, as_samples
from dimod.traversal import connected_components
import dwave_networkx
import dimod
from dimod import ExactSolver
import networkx as nx
import itertools
from collections import deque
from dimod import Vartype


def build_biconnected_tree_decomp(G):
    """
    Build a tree decomposition of a graph based on its biconnected components.

    Args:
        G: a networkx Graph.

    Returns:
        T: a networkx Digraph that is a tree.

        root: the root vertex of T.

        Each vertex x of T is a tuple of vertices V_x in G that induces a biconnected component. Associated with x is
        the data "cuts", indicating with vertices in V_x are cut vertices.
        An arc (x, y) in T indicates that V_x and V_y share a cut vertex c. The vertex c is in the data "cut" of (x, y).


    """


    cut_vertices = list(nx.articulation_points(G))
    bcc = [tuple(c) for c in nx.biconnected_components(G)]

    # build components associated with each cut vertex and digraph nodes
    components = {v: [] for v in cut_vertices}
    T = nx.DiGraph()
    for c in bcc:
        T.add_node(c, cuts = [v for v in c if v in cut_vertices])
        for v in T.nodes[c]['cuts']:
            components[v].append(c)


    # bfs on components to find tree structure
    visited = {c: False for c in bcc}
    root = bcc[0]
    queue = [root]
    for v in T.nodes():
        T.nodes[v]['child_cuts'] = []
    while queue:
        c1 = queue.pop(0)
        visited[c1] = True
        for v in T.nodes[c1]['cuts']:
            for c2 in components[v]:
                if not(visited[c2]):
                    T.add_edge(c1, c2, cut=v)
                    T.nodes[c2]['parent_cut'] = v
                    T.nodes[c2]['parent_node'] = c1
                    T.nodes[c1]['child_cuts'].append(c2)
                    queue.append(c2)

    return T, root


def sub_bqm(bqm, variables):
    # build bqm out of a small subset of variables.
    linear = {v: bqm.linear[v] for v in variables}
    quadratic = {(u, v): bqm.quadratic[(u, v)] for (u, v) in itertools.combinations(variables, 2)
                    if (u, v) in bqm.quadratic}
    return dimod.BinaryQuadraticModel(linear, quadratic, offset=0, vartype=bqm.vartype)


def get_marginals(bqm, x, sampler, **parameters):
    not_one = -1 if bqm.vartype == Vartype.SPIN else 0
    marginals = dict()
    for value in [1, not_one]:

        bqm2 = bqm.copy()
        bqm2.fix_variable(x, value)
        marginals[value] = sampler.sample(bqm2, **parameters).truncate(1)

    delta = marginals[1].record.energy[0] - marginals[not_one].record.energy[0]
    return marginals, delta


class BiconnectedTreeDecomposition(object):

    def __init__(self, bqm):
        self.bqm = bqm
        G = bqm.to_networkx_graph()

        # build the tree decomposition:
        self.T, self.root = build_biconnected_tree_decomp(G)


    def sample(self, sampler, **parameters):

        # work up the edges of the tree from leafs to root, sampling at each step. Record energy and best state in each
        # bcc for each cut vertex configuration. From root, propagate solutions back down the tree.

        T = self.T
        root = self.root
        bqm = self.bqm


        cut_vertex_marginals = dict()
        delta_energies = dict()

        # build marginals bottom-up.
        for c in nx.dfs_postorder_nodes(T, source=root):

            # get component
            bqm_copy = sub_bqm(bqm, c)
            # adjust linear biases on child cut vertices
            child_cut_vertices = [T.edges[(c, cc)]['cut'] for (_, cc) in T.out_edges(c)]
            for cv in child_cut_vertices:
                bqm_copy.add_variable(cv, delta_energies[cv] - bqm_copy.linear[cv])

            if c != root:
                # not at root yet. Unique parent cut vertex.
                #parent_component, _ = next(iter(T.in_edges(c)))
                #parent_cut_vertex = T.edges[(parent_component, c)]['cut']
                parent_component, parent_cut_vertex = T.nodes[c]['parent_node'], T.nodes[c]['parent_cut']

                # here .truncate(1) is used to pick the best solution only.
                marginal, delta = get_marginals(bqm_copy, parent_cut_vertex, sampler, **parameters)
                cut_vertex_marginals[parent_cut_vertex] = marginal
                delta_energies[parent_cut_vertex] = delta
            else:
                # solve at the root node.
                sampleset = sampler.sample(bqm_copy).truncate(1)

        # sample from bqm top-down.
        for c in nx.dfs_preorder_nodes(T, source=root):
            if c != root:
                # Unique parent cut vertex.
                parent_component, parent_cut_vertex = T.nodes[c]['parent_node'], T.nodes[c]['parent_cut']

                # add component to solution according to value of parent cut vertex.
                samples, labels = as_samples(sampleset)
                parent_cut_value = samples[0, labels.index(parent_cut_vertex)]
                sampleset = sampleset.append_variables(cut_vertex_marginals[parent_cut_vertex][parent_cut_value])

        # recompute energies (total energy was messed up by linear biases):
        sampleset = SampleSet.from_samples_bqm(sampleset, bqm)
        return sampleset



if __name__ == '__main__':
    comps =  [[0,1,2], [2,3,4], [3,5,6], [4,7,8]]
    J = {(u, v): 1  for c in comps for (u, v) in itertools.combinations(c, 2)}
    h = {0: 10}
    bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
    sampler = ExactSolver()
    response = sampler.sample(bqm)
    print(response.first.energy)

    parameters = dict()
    tree_decomp = BiconnectedTreeDecomposition(bqm)
    sampleset = tree_decomp.sample(sampler, **parameters)

    print(sampleset)




