# Copyright 2019 D-Wave Systems Inc.
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
#
# =============================================================================
"""
A composite that breaks the problem into sub-problems corresponding to the
biconnected components of the binary
quadratic model graph before sending to its child sampler.
"""

from dimod.sampleset import as_samples
from dimod.core.composite import ComposedSampler
from dimod.sampleset import SampleSet
import dimod
import networkx as nx
import itertools


__all__ = ['CutVertexComposite']


class CutVertexComposite(ComposedSampler):
    """Composite to decompose a problem into biconnected components
    and solve each.

    Biconnected components of a bqm graph are computed (if not provided),
    and each subproblem is passed to the child sampler.
    Returned samples from each child sampler are merged. Only the best solution
    of each response is pick and merge with others
    (i.e. this composite returns a single solution).

    Args:
       sampler (:obj:`dimod.Sampler`):
            A dimod sampler

    Examples:
       This example uses :class:`.CutVertexComposite` to instantiate a
       composed sampler that submits a simple Ising problem to a sampler.
       The composed sampler finds the cut vertex ("2"), breaks the problem into biconnected components ({0, 1,
       2} and {2, 3, 4}), solves each biconnected components and combines the results into a single solution.

       >>> h = {}
       >>> J = {e: -1 for e in [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]}
       >>> sampler = dimod.CutVertexComposite(dimod.ExactSolver())
       >>> sampleset = sampler.sample_ising(h, J)

    """

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        params = self.child.parameters.copy()
        return params

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample(self, bqm, tree_decomp=None, **parameters):
        """Sample from the provided binary quadratic model.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            tree_decomp: BiconnectedTreeDecomposition
                Tree decomposition of the bqm. Computed if not provided.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """

        if not(len(bqm.variables)):
            return SampleSet.from_samples_bqm({}, bqm)

        if tree_decomp is None:
            tree_decomp = BiconnectedTreeDecomposition(bqm)

        return tree_decomp.sample(self.child, **parameters)


def sub_bqm(bqm, variables):
    # build bqm out of a small subset of variables. Equivalent to fixing all other variables to 0.
    linear = {v: bqm.linear[v] for v in variables}
    quadratic = {(u, v): bqm.quadratic[(u, v)] for (u, v) in itertools.combinations(variables, 2)
                 if (u, v) in bqm.quadratic}
    return dimod.BinaryQuadraticModel(linear, quadratic, offset=0, vartype=bqm.vartype)


def get_marginals(bqm, x, sampler, **parameters):
    # Get a sample from bqm with x fixed to each of its two possible values.
    # Return the samples and the difference in energy.

    not_one = -1 if bqm.vartype == dimod.Vartype.SPIN else 0
    marginals = dict()
    for value in [1, not_one]:

        bqm2 = bqm.copy()
        bqm2.fix_variable(x, value)
        # here .truncate(1) is used to pick the best solution only.
        marginals[value] = sampler.sample(bqm2, **parameters).truncate(1)

    delta = marginals[1].record.energy[0] - marginals[not_one].record.energy[0]
    return marginals, delta


class BiconnectedTreeDecomposition(object):

    def __init__(self, bqm):
        self.bqm = bqm
        G = bqm.to_networkx_graph()

        if not(nx.is_connected(G)):
            raise ValueError("bqm is not connected. Use ConnectedComponentsComposite(CutVertexComposite(...)).")

        # build the tree decomposition:
        self.T, self.root = self.build_biconnected_tree_decomp(G)

    @staticmethod
    def build_biconnected_tree_decomp(G):
        """
        Build a tree decomposition of a graph based on its biconnected components.

        Args:
            G: a networkx Graph.

        Returns:
            T: a networkx Digraph that is a tree.

            root: the root vertex of T.

            Each vertex x of T is a tuple of vertices V_x in G that induces a biconnected component. Associated with
            x is the data:
                "cuts": a list of vertices of G in V_x that are cut vertices.
                "parent_cut": the cut vertex in V_x connecting V_x to its parent in the tree.
                "child_nodes": the children of V_x in the tree.
            An arc (x, y) in T indicates that V_x and V_y share a cut vertex c, and V_x is the parent of V_y. The
            vertex c is in the data "cut" of arc (x, y).

        """

        cut_vertices = list(nx.articulation_points(G))
        biconnected_components = [tuple(c) for c in nx.biconnected_components(G)]

        # build components associated with each cut vertex and digraph nodes
        components = {v: [] for v in cut_vertices}
        T = nx.DiGraph()
        for c in biconnected_components:
            T.add_node(c, cuts=[v for v in c if v in cut_vertices])
            for v in T.nodes[c]['cuts']:
                components[v].append(c)

        # bfs on components to find tree structure
        root = biconnected_components[0]
        queue = [root]
        for v in T.nodes():
            T.nodes[v]['child_cuts'] = []
            T.nodes[v]['child_nodes'] = []
        visited = {c: False for c in biconnected_components}
        visited[root] = True
        while queue:
            c1 = queue.pop(0)
            for v in T.nodes[c1]['cuts']:
                for c2 in components[v]:
                    if not (visited[c2]):
                        T.add_edge(c1, c2, cut=v)
                        T.nodes[c2]['parent_cut'] = v
                        T.nodes[c2]['parent_node'] = c1
                        T.nodes[c1]['child_cuts'].append(v)
                        T.nodes[c1]['child_nodes'].append(c2)
                        queue.append(c2)
                        visited[c2] = True

        return T, root

    def sample(self, sampler, **parameters):
        """
        Sample from a bqm by sampling from its biconnected components.

        Args:
            sampler: dimod sampler used to sample from components.

            parameters: parameters passed to sampler.

        Returns:
            sampleset: a dimod sampleset.


        Method: dynamic programming on the tree decomposition from the biconnected components.
        Each node in the tree decomposition is a biconnected component in the bqm.
        Working up from the leaves of the tree to the root, sample each biconnected component. Record the energy and
        best state in the component, for each configuration of the cut vertex associated with the parent of that
        component in the tree. When sampling, the linear biases of the cut vertices associated with children in the
        tree are modified so that the energy difference between the best states from the remainder of the tree below
        the cut vertex are accounted for.
        Then, working down from the root of the tree to the leaves, sample from each biconnected component. Use the
        resulting value at a cut vertex to sample from the remainder of the tree below that cut vertex.
        """

        T = self.T
        root = self.root
        bqm = self.bqm

        cut_vertex_conditionals = dict()
        delta_energies = dict()

        # build marginals from leaves of tree up.
        for c in nx.dfs_postorder_nodes(T, source=root):

            # get component
            bqm_copy = sub_bqm(bqm, c)
            # adjust linear biases on child cut vertices
            child_cut_nodes = T.nodes[c]['child_nodes']
            for cc in child_cut_nodes:
                linear_offset = delta_energies[cc] if bqm.vartype == dimod.BINARY else delta_energies[cc]/2.
                cv = T.nodes[cc]['parent_cut']
                bqm_copy.add_variable(cv, linear_offset - bqm.linear[cv])

            if c != root:
                # not at root yet. Unique parent cut vertex.
                parent_cut_vertex = T.nodes[c]['parent_cut']
                marginal, delta = get_marginals(bqm_copy, parent_cut_vertex, sampler, **parameters)
                cut_vertex_conditionals[c] = marginal
                delta_energies[c] = delta
            else:
                # solve at the root node.
                # here .truncate(1) is used to pick the best solution only.
                sampleset = sampler.sample(bqm_copy).truncate(1)

        # sample bqm from root of tree down.
        for c in nx.dfs_preorder_nodes(T, source=root):
            if c != root:
                # Unique parent cut vertex.
                parent_cut_vertex = T.nodes[c]['parent_cut']

                # add component to solution according to value of parent cut vertex.
                samples, labels = as_samples(sampleset)
                if samples.shape[0] == 1:
                    # Extract a single sample from the cut vertex conditionals.
                    parent_cut_value = samples[0, labels.index(parent_cut_vertex)]
                    sampleset = sampleset.append_variables(cut_vertex_conditionals[c][parent_cut_value])
                else:
                    # For now we're only producing a single sample. To produce multiple samples, resample from each
                    # biconnected component with the parent cut vertices fixed to each of its two values.
                    raise NotImplementedError

        # recompute energies (total energy was messed up by linear biases):
        sampleset = SampleSet.from_samples_bqm(sampleset, bqm)
        return sampleset







