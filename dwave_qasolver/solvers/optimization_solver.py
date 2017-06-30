"""
TODO: Note that this requires SAPI for now, we would like to remove
this requirement
"""
from dwave_sapi2.local import local_connection
from dwave_sapi2.core import solve_ising
from dwave_sapi2.util import get_hardware_adjacency, qubo_to_ising
from dwave_sapi2.embedding import find_embedding, embed_problem, unembed_answer


from dwave_qasolver.solver_template import DiscreteModelSolver
from dwave_qasolver.solution_templates import SpinResponse
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
            return SpinResponse(solutions)

        solver = _sapi_solver

        A = self.structure

        embeddings = find_embedding(J, A)

        (h0, j0, jc, new_emb) = embed_problem(h, J, embeddings, A)
        emb_j = j0.copy()
        emb_j.update(jc)

        structured_solution = self.solve_structured_ising(h0, emb_j, **args)

        new_answer = unembed_answer(structured_solution, new_emb, 'minimize_energy', h, J)

        solutions = [{idx: spin for idx, spin in enumerate(ans)} for ans in new_answer]

        return SpinResponse(solutions)

    @solve_ising_api()
    def solve_structured_ising(self, h, J, **args):
        """Solves an Ising problem native to the system.

        TODO
        """
        result = solve_ising(_sapi_solver, h, J, **args)

        solutions = [{idx: spin for idx, spin in enumerate(soln)}
                     for soln in result['solutions']]

        return SpinResponse(solutions)

    @property
    def structure(self):
        return get_hardware_adjacency(_sapi_solver)
