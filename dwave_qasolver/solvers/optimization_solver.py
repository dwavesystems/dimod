"""
TODO: Note that this requires SAPI for now, we would like to remove
this requirement
"""
from dwave_sapi2.local import local_connection
from dwave_sapi2.core import solve_ising
from dwave_sapi2.util import get_hardware_adjacency, qubo_to_ising
from dwave_sapi2.embedding import find_embedding, embed_problem, unembed_answer


from dwave_qasolver.solver_template import DiscreteModelSolver
from dwave_qasolver.solution_templates import SpinSolution
from dwave_qasolver.decorators import solve_qubo_api, solve_ising_api
from dwave_qasolver.decorators import qubo_index_labels, ising_index_labels


_sapi_solver = local_connection.get_solver("c4-sw_optimize")


class SoftwareOptimizer(DiscreteModelSolver):
    @solve_qubo_api()
    @qubo_index_labels()
    def solve_qubo(self, Q, **args):
        (h, J, ising_offset) = qubo_to_ising(Q)
        solutions = self.solve_ising(h, J, **args)
        return solutions.as_bool()

    @solve_ising_api()
    @ising_index_labels()
    def solve_ising(self, h, J, **args):

        if not J:
            solutions = [{node: h[node] < 0 and 1 or -1 for node in h}]
            return SpinSolution(solutions)

        solver = _sapi_solver

        A = get_hardware_adjacency(solver)

        embeddings = find_embedding(J, A)

        (h0, j0, jc, new_emb) = embed_problem(h, J, embeddings, A)
        emb_j = j0.copy()
        emb_j.update(jc)
        result = solve_ising(solver, h0, emb_j, num_reads=6)
        new_answer = unembed_answer(result['solutions'], new_emb, 'minimize_energy', h, J)

        solutions = [{idx: spin for idx, spin in enumerate(ans)} for ans in new_answer]

        return SpinSolution(solutions)
