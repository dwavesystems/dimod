import networkx as nx
import numpy as np

__all__ = ["MarkovNetwork"]


class MarkovNetwork(nx.DiGraph):

    def __init__(self, edges, using_quadratic_values=False):
        super(MarkovNetwork, self).__init__()

        self.phi_edges = edges
        self.using_quadatric_values = using_quadratic_values

        self.PHI_NDX_NAME_X = 0
        self.PHI_NDX_NAME_Y = 1
        self.PHI_NDX_A = 2
        self.PHI_NDX_B = 3
        self.PHI_NDX_C = 4
        self.PHI_NDX_D = 5

        self.nodename_to_ordinal = {}
        self.nodenames = []

        self._setup()

    def _setup(self):
        for e in self.phi_edges:
            self._add_edge(e)

    def _add_edge(self, edge):
        nx, ny = edge[self.PHI_NDX_NAME_X: self.PHI_NDX_NAME_Y+1]
        nxi = self._set_node_to_ordinal(nx)
        nyi = self._set_node_to_ordinal(ny)
        phi11, phi10, phi01, phi00, a, b, c, d = self.compute_edge_weights(edge, nxi, nyi)

        # swap order across diagonal
        if nxi > nyi:
            nx, ny = ny, nx
            nxi, nyi = nyi, nxi

        attrs = {'edge_vector': [nx, ny,
                                 nxi, nyi,
                                 phi11, phi10, phi01, phi00,
                                 a, b, c, d]
                 }

        self.add_edge(nx, ny, **attrs)

    def _set_node_to_ordinal(self, node):
        if node in self.nodename_to_ordinal:
            nodei = self.nodename_to_ordinal[node]
        else:
            self.nodenames.append(node)
            self.nodename_to_ordinal[node] = len(self.nodenames) - 1
            nodei = len(self.nodenames) - 1
        return nodei

    def compute_edge_weights(self, edge, nxi, nyi):
        """

        Args:
            edge_data: NetworkX edge with data, (u,v, {attr dict})
                Ex. (nx, ny, **{'ordinal': (nxi, nyi), 'abcd': (a,b,c,d), 'reversed': True})

            Order of phi values :
                1) when nx = 1 and ny = 1 => phi11
                2) when nx = 1 and ny = 0 => phi10
                3) when nx = 0 and ny = 1 => phi01
                4) when nx = 0 and ny = 0 => phi00

        Returns: tuple of floats,  (phi11, phi10, phi01, phi00, a, b, c, d)

        """
        if not self.using_quadatric_values:
            phi11, phi10, phi01, phi00 = edge[self.PHI_NDX_A: self.PHI_NDX_D+1]
            if nxi > nyi:  # node QUBO assignment needs to be reversed
                phi10, phi01 = phi01, phi10
            # general calculation of phi values in a QUBO
            a = phi10 - phi00
            b = phi01 - phi00
            c = phi11 - phi10 - phi01 + phi00
            d = phi00
        else:
            # This computation assumes that the edge vector contains pre-computed phi values
            a, b, c, d = edge[self.PHI_NDX_A: self.PHI_NDX_D+1]
            phi11 = a + b + c + d
            phi10 = d + a
            phi01 = d + b
            phi00 = d
            if nxi > nyi:  # node QUBO assignment needs to be reversed and recalculated
                phi10, phi01 = phi01, phi10
                a = phi10 - phi00
                b = phi01 - phi00
                c = phi11 - phi10 - phi01 + phi00
                d = phi00
        return phi11, phi10, phi01, phi00, a, b, c, d

    def get_markov_edges(self):
        # Less 'state', for Marcus :)
        for edge in self.edges(data=True):
            yield edge[-1]['edge_vector']

    def _create_Q(self):
        Q_matrix = np.zeros([len(self.nodename_to_ordinal), len(self.nodename_to_ordinal)])

        # list indices of respective fields
        NDX_NX, NDX_NY, NDX_A, NDX_B, NDX_C = 2, 3, 8, 9, 10

        # this for loop fills a Q matrix with the quadratic values (note some values will be increments)
        # this is the step that converts the Markov graph with quadratics into a Q(UBO) matrix
        # In Q matrix form, it is a simple transformation to clamp out nodes of the graph
        for edge in self.get_markov_edges():
            nxi = edge[NDX_NX]
            nyi = edge[NDX_NY]
            Q_matrix[nxi, nxi] += edge[NDX_A]
            Q_matrix[nyi, nyi] += edge[NDX_B]
            Q_matrix[nxi, nyi] += edge[NDX_C]

        return Q_matrix

    def _create_and_clamp_Q(self, forced_vars):
        """
        1. Create the dense Q matrix, a Numpy ndarray of size NxN, where N = #nodes
        2. Clamp out vertices/(rows/columns) from the forced_vars list

        Args:
            forced_vars: list of tuples, (node/idx, val), where val \in {0,1}

        Returns:
            N x N array, dict mapping {clamped node -> ordinal}
        """
        Q = self._create_Q()
        clamped_nodename_to_ordinal = self.nodename_to_ordinal.copy()

        # With the Q form of the Markov network, clamp the values in the forced vars list,
        # which reduces the size of the problem
        for node, t_f in forced_vars:  # for each forced_vars node, clamp
            ij_clamp = self.nodename_to_ordinal[node]
            # get rid of index for clamped matrix
            clamped_nodename_to_ordinal.pop(node)
            if t_f:
                for i in range(0, ij_clamp):
                    Q[i, i] += Q[i, ij_clamp]
                for i in range(ij_clamp, len(self.nodename_to_ordinal)):
                    Q[i, i] += Q[ij_clamp, i]
        return Q, clamped_nodename_to_ordinal

    def create_clampedQ(self, forced_vars):
        """
        This method generates the clamp matrix in sparse matrix form for use as a QUBO.

        Args:
            forced_vars: list of list of nodes to force, of form [node, 1/0]

        Returns:
            sparse matrix (dictionary), dict of clamped nodes {clamped node -> ordinal}
        """
        # _create_and_clamp_Q: With the Q form of the Markov network, clamp the values in the forced vars list,
        # which reduces the size of the problem
        # Simultaneously, create a correspondence (dictionary) between the Q matrix variables
        # and the original Markov network variables, stored in 'clamped_nodename_to_ordinal'
        Q, clamped_nodename_to_ordinal = self._create_and_clamp_Q(forced_vars)
        i_compr = []
        for i, (node, i_n) in enumerate(sorted(clamped_nodename_to_ordinal.items(), key=lambda x: x[1])):
            i_compr.append(i_n)
            clamped_nodename_to_ordinal[node] = i

        # Create the compressed Q from Q, which has been clamped,
        # but still has values in the clamped rows and columns
        return {(i, i+j): Q[i_compr_x, i_compr_y]
                for i, i_compr_x in enumerate(i_compr)
                for j, i_compr_y in enumerate(i_compr[i:])
                if Q[i_compr_x, i_compr_y] != 0.0
                }, clamped_nodename_to_ordinal
