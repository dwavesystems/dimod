"""
TODO: Note that this requires SAPI for now, we would like to remove
this requirement
"""
from dwave_sapi2.local import local_connection
from dwave_sapi2.core import solve_ising, solve_qubo
from dwave_sapi2.util import get_chimera_adjacency, qubo_to_ising
from dwave_sapi2.embedding import find_embedding, embed_problem, unembed_answer


from dwave_qasolver.solver_template import DiscreteModelSolver

from dwave_qasolver.decorators import solve_qubo_api


class SoftwareOptimizer(DiscreteModelSolver):
    @solve_qubo_api()
    def solve_qubo(self, Q, detailed=False, **args):
        # relabel Q with indices
        label = {}
        idx = 0
        for n1, n2 in Q:
            if n1 not in label:
                label[n1] = idx
                idx += 1
            if n2 not in label:
                label[n2] = idx
                idx += 1
        Qrl = {(label[n1], label[n2]): Q[(n1, n2)] for (n1, n2) in Q}

        # get the solfware solver from sapi
        solver = local_connection.get_solver("c4-sw_optimize")
        A = get_chimera_adjacency(4, 4, 4)

        # convert the problem to Ising
        (h, J, ising_offset) = qubo_to_ising(Qrl)

        # get the embedding, this function assumes that the given problem is
        # unstructured
        embeddings = find_embedding(Qrl, A, tries=50)
        if not embeddings:
            raise Exception('problem is too large to be embedded')
        [h0, j0, jc, embeddings] = embed_problem(h, J, embeddings, A)

        # actually solve the thing
        j = j0
        j.update(jc)
        result = solve_ising(solver, h0, j)
        ans = unembed_answer(result['solutions'], embeddings, 'minimize_energy', h, J)

        # unapply the relabelling and convert back from spin
        inv_label = {label[n]: n for n in label}
        return {inv_label[i]: (spin + 1) / 2 for i, spin in enumerate(ans[0])}
