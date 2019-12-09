# Copyright 2018 D-Wave Systems Inc.
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
Lower bounds on the energy of an Binary Quadratic Model using linear programs.

"""

import dimod
import time
from mip.model import *
from mip.callbacks import ConstrsGenerator, CutPool
import itertools
import numpy as np


class TriangleGenerator(ConstrsGenerator):
    # Constraint generator structure for the cut polytope inequalities used in the linear program.

    def __init__(self, bqm, nodes, couplers, all_triangles=True, verbose=False):
        self.nodes = nodes
        self.couplers = couplers
        self.G = bqm.to_networkx_graph()
        self.verbose = verbose
        self.total_cuts = 0
        self.all_triangles = all_triangles

        # Generate triangles to be considered for constraint generation.
        self.triangles = dict()
        for a in self.G.nodes():
            self.triangles[a] = []
            if all_triangles:
                # include all triangles.
                self.triangles[a] = [(b, c) for (b, c) in itertools.combinations(self.G[a], 2) if b in self.G[c]]
            else:
                # Pick one largest triangle for each node.
                best_weight = np.Inf
                for (b, c) in itertools.combinations(self.G[a], 2):
                    if b in self.G[c]:
                        weight6 = bqm.linear[a] + bqm.linear[b] + bqm.linear[c] \
                                  - bqm.quadratic[b, c] - bqm.quadratic[a, b] - bqm.quadratic[a, c] - 1
                        weight4 = bqm.quadratic[a, b] - bqm.quadratic[a, c] - bqm.linear[a] - bqm.quadratic[b, c]
                        weight = min(weight4, weight6)
                        if weight < best_weight:
                            best_weight = weight
                            self.triangles[a] = [(b, c)]

        self.possible_cuts = sum(len(val) for val in self.triangles.values())
        self.node_index = {a: i for i, a in enumerate(self.G.nodes())}

    def generate_constrs(self, model: Model):
        # Constraint generator callback function.

        triangles = self.triangles
        cp = CutPool()
        model_var_names = {v.name: v for v in model.vars}
        node_index = self.node_index

        def node_var(a):
            return model_var_names[str(a)]

        def coupler_var(a, b):
            return model_var_names[str(sorted((a, b)))]

        cut4, cut6 = 0, 0
        vars_checked = 0
        tol = 1e-3
        for a, pairs in triangles.items():
            a_var = node_var(a)
            # Add constraints: x_ab + x_ac <= x_a + x_bc  and  x_a + x_b + x_c - x_ab - x_ac - x_bc <= 1
            # These constraints can only be violated if x_a > 0 and x_a < 1.
            if a_var.x > tol and 1-a_var.x > tol:
                vars_checked += 1
                for (b, c) in pairs:

                    ab_var, bc_var, ac_var = coupler_var(a, b), coupler_var(b, c), coupler_var(a, c)

                    if ab_var.x + ac_var.x > a_var.x + bc_var.x + tol:
                        cut4 += 1
                        cut = ab_var + ac_var <= a_var + bc_var
                        cp.add(cut)

                    # symmetry breaking:
                    if not self.all_triangles or (node_index[a] < node_index[b] and node_index[a] < node_index[c]):
                        b_var, c_var = node_var(b), node_var(c)
                        if a_var.x + b_var.x + c_var.x - ab_var.x - ac_var.x - bc_var.x > 1 + tol:
                            cut6 += 1
                            cut = a_var + b_var + c_var - ab_var - ac_var - bc_var <= 1
                            cp.add(cut)

                    # don't add too many constraints at once
                    if len(cp.cuts) > 256:
                        for cut in cp.cuts:
                            model += cut
                        self.total_cuts += len(cp.cuts)
                        if self.verbose:
                            print("3-cycles: 4-cuts: {}, 6-cuts: {}, total cuts: {} out of {}, vars checked: {}".
                                  format(cut4, cut6, self.total_cuts, self.possible_cuts, vars_checked))
                        return True

        self.total_cuts += len(cp.cuts)
        if self.verbose:
            print("3-cycles: 4-cuts: {}, 6-cuts: {}, total cuts: {} out of {}, vars checked: {}".
                  format(cut4, cut6, self.total_cuts, self.possible_cuts, vars_checked))
        for cut in cp.cuts:
            model += cut
        return len(cp.cuts) > 0


class FourCycleGenerator(ConstrsGenerator):
    # Constraint generator structure for the cut polytope inequalities used in the linear program.

    def __init__(self, bqm, nodes, couplers, all_fourcycles=True, verbose=False):
        self.nodes = nodes
        self.couplers = couplers
        self.G = bqm.to_networkx_graph()
        self.verbose = verbose
        self.total_cuts = 0

        # Generate four-cycles to be considered for constraint generation.
        self.fourcycles = dict()
        node_index = {a: i for i, a in enumerate(self.G.nodes())}
        for a in self.G.nodes():
            self.fourcycles[a] = []
            excluded = set(self.G[a]).union({a})
            if all_fourcycles:
                # include all four-cycles, excluding triangles.
                # each four cycle (a, b) implies a constraint a + b - ab - bc + cd - da <= 1.
                for (b, d) in itertools.combinations(self.G[a], 2):
                    c_list = set(self.G[b]).intersection(self.G[d]).difference(excluded)
                    for c in c_list:
                        # break symmetry: only include one orientation of each cycle (a, b, c, d) or (a, d, c, b).
                        # choose the orientation in which the node with the smallest index is followed by the next
                        # smallest index.
                        quad = (a, b, c, d)
                        smallest_idx = quad.index(min(quad, key=lambda x: node_index[x]))
                        orientation = node_index[quad[(smallest_idx+1) % 4]] < node_index[quad[(smallest_idx-1) % 4]]
                        if orientation:
                            self.fourcycles[a].append((b, c, d))
                        else:
                            self.fourcycles[a].append((d, c, b))

            else:
                # pick one largest four-cycle for each node.
                best_weight = np.Inf
                for (b, d) in itertools.combinations(self.G[a], 2):
                    c_list = set(self.G[b]).intersection(self.G[d]).difference(excluded)
                    for c in c_list:
                        # pick one largest triangle for each node
                        weight = bqm.linear[a] + bqm.linear[b] + bqm.quadratic[c, d] \
                                 - bqm.quadratic[a, b] - bqm.quadratic[b, c] - bqm.quadratic[d, a]
                        weight = min(weight-1, -weight)
                        if weight < best_weight:
                            best_weight = weight
                            self.fourcycles[a] = [(b, c, d)]

        self.possible_cuts = sum(len(val) for val in self.fourcycles.values())

    def generate_constrs(self, model: Model):
        # Constraint generator callback function.

        fourcycles = self.fourcycles
        cp = CutPool()
        model_var_names = {v.name: v for v in model.vars}

        def node_var(a):
            return model_var_names[str(a)]

        def coupler_var(a, b):
            return model_var_names[str(sorted((a, b)))]

        cut_hi, cut_lo = 0, 0
        vars_checked = 0
        tol = 1e-3
        for a, triples in fourcycles.items():
            a_var = node_var(a)

            # Add constraints:
            # a + b - ab - bc + cd - da <= 1
            # -a - b + ab + bc - cd + da <= 0

            # Constraints can only be violated if x_a > 0 and x_a < 1.
            if a_var.x > tol and 1-a_var.x > tol:
                vars_checked += 1
                for (b, c, d) in triples:
                    ab_var, bc_var, cd_var, da_var = coupler_var(a, b), coupler_var(b, c), coupler_var(c, d), \
                                                     coupler_var(d, a)

                    b_var, c_var, d_var = node_var(b), node_var(c), node_var(d)
                    tot = a_var.x + b_var.x - ab_var.x - bc_var.x + cd_var.x - da_var.x
                    var_tot = a_var + b_var - ab_var - bc_var + cd_var - da_var
                    if tot > 1 + tol:
                        cut_hi += 1
                        cut = var_tot <= 1
                        cp.add(cut)

                    if -tot > tol:
                        cut_lo += 1
                        cut = -var_tot <= 0
                        cp.add(cut)

                    # don't add too many constraints at once
                    if len(cp.cuts) > 256:
                        for cut in cp.cuts:
                            model += cut
                        self.total_cuts += len(cp.cuts)
                        if self.verbose:
                            print("4-cycles: hi-cuts: {}, lo-cuts: {}, total cuts: {} out of {}, vars checked: {}".
                                  format(cut_hi, cut_lo, self.total_cuts, self.possible_cuts, vars_checked))
                        return True

        self.total_cuts += len(cp.cuts)
        if self.verbose:
            print("4-cycles: hi-cuts: {}, lo-cuts: {}, total cuts: {} out of {}, vars checked: {}".
                  format(cut_hi, cut_lo, self.total_cuts, self.possible_cuts, vars_checked))
        for cut in cp.cuts:
            model += cut
        return len(cp.cuts) > 0


def lp_lower_bound(input_bqm, cycle_cuts=None, verbosity=0, max_time=None, initial_state=None, integer_solve=False):
    """Find a lower bound on the ground state bound of a binary quadratic model using a linear program relaxation.

    Args:
        input_bqm (:obj:`.BinaryQuadraticModel`):
            A binary polynomial.

        cycle_cuts (0, 3, or 4, default = None):
            Maximum length of cycle cut inequalities in the linear programming relaxation. If cycle_cuts = 0,
            no cycle_cut inequalities are added. If cycle_cuts = 3, cycle_cuts of length 3 are included. If
            cycle_cuts = 4, cycle_cuts of length 3 and 4 are used. If cycle_cuts are note specified, by default
            cycle_cuts = 4 unless integer_solve=True, in which case cycles_cuts = 0.
            The more cycle_cuts used, the better the bound, but the longer the computation of the bound will take.
            See below for more details on cycle cuts.

        verbosity (0, 1, 2, default = 0):
            How much debugging output to print.

        max_time: (integer, default=None):
            maximum number of seconds allowed.

        initial_state (dict, default=None):
            An optional initial state to provided to the linear program.

        integer_solve (boolean, default=False):
            Solve the bqm exactly as an integer program. Typically much slower than finding lower bounds. If the
            problem is solved to optimality (i.e. max_time=None), the returned solution is a ground state of the bqm.


    Returns:
        float: lower bound on the energy of the bqm.

        dict: additional information resulting from the lower bound computation, including the following keys:
            'state': values of the variables in the LP relaxation.
            'optimality': whether or not the program was solved to optimality.



    This method solves a linear programming relaxation of a QUBO. More precisely, a QUBO of the form

    \min_\{x_i \in {0,1}\} \sum_{i < j} x_i Q_ij x_j

    is linearized by introducing auxiliary variables x_{ij} such that x_{ij} = x_i x_j. This constraint is enforced
    using the linear equations

    x_ij <= x_i, x_j
    x_ij >= x_i + x_j - 1

    Finally, the linear optimization problem

    \min_\{x \in {0,1}^n\} \sum_{i < j} Q_ij x_{ij}
    s.t.    x_ij <= x_i, x_j
            x_ij >= x_i + x_j - 1

    is relaxed from \{0,1\} discrete variables to continuous variables in [0,1]. The resulting optimization problem
    is solved to optimality using linear programming. Since it is a relaxation of the original QUBO, its optimal
    objective value is a lower bound on on the QUBO objective value. See [#bh]_ for more details.

    If cycle_cuts are used, additional inequalities are included in the linear program which make the bound tighter.
    If (a, b, c) is a 3-cycle in the QUBO, then the following inequalities hold for the {0,1} variables:

    (a):    x_ab + x_ac <= x_a + x_bc   (and similary for b and c)
    x_a + x_b + x_c - x_ab - x_ac - x_bc <= 1

    If (a, b, c, d) is a 4-cycle, the following inequalities hold:

    (a,b):  x_a + x_b - x_ab - x_bc + x_cd - x_da <= 1
           -x_a - x_b + x_ab + x_bc - x_cd + x_da <= 0      (and similarly for (b,c), (c,d), and (d,a).)

    These inequalities are equivalent to the cut-polytope inequalities of [#bm]_.

    Requires the package python-mip.

    .. [#bh] Endre Boros, E. and Hammer, P.L.: "Pseudo-Boolean optimization". Disc. Appl. Math. 123, 155-225 (2002).

    .. [#bm] Barahona, F. and Mahjoub, A.R.: ""On the cut polytope". Math. Prog. 36, 157-173 (1986).

    """

    vartype = input_bqm.vartype
    bqm = input_bqm.change_vartype(dimod.BINARY, inplace=False)

    if not bqm:
        bound = bqm.offset
        info = {'state': None}
        return bound, info

    # Build the model.
    solver = Model(sense=MINIMIZE, solver_name=CBC)
    solver.verbose = verbosity >= 2
    start_time = time.time()

    # Create the variables.
    if integer_solve:
        var_args = {'var_type': INTEGER, 'lb': 0, 'ub': 1}
    else:
        var_args = {'var_type': CONTINUOUS, 'lb': 0, 'ub': 1}
    nodes = {v: solver.add_var(name=str(v), **var_args) for v in bqm.variables}
    couplers = {e: solver.add_var(name=str(sorted(e)), **var_args) for e in bqm.quadratic.keys()}

    # Create the constraints.
    for e in bqm.quadratic.keys():
        u, v = e
        solver += couplers[e] <= nodes[u]
        solver += couplers[e] <= nodes[v]
        solver += couplers[e] >= nodes[u]+nodes[v]-1

    # Set objective function
    linear_obj = xsum(val*nodes[v] for v, val in bqm.linear.items())
    quadratic_obj = xsum(val*couplers[e] for e, val in bqm.quadratic.items())
    solver.objective = (linear_obj + quadratic_obj)

    # Add cycle cut generator
    if cycle_cuts is None:
        cycle_cuts = 0 if integer_solve else 4
    if cycle_cuts > 0:
        if integer_solve:
            solver.cuts_generator = TriangleGenerator(bqm, nodes, couplers)
        else:
            cut_generators = [TriangleGenerator(bqm, nodes, couplers, verbose=bool(verbosity))]
            if cycle_cuts == 4:
                cut_generators.append(FourCycleGenerator(bqm, nodes, couplers, verbose=bool(verbosity)))

    # Set the solver parameters.
    # see https://python-mip.readthedocs.io/ for additional solver options.
    if initial_state:
        solver.start = [(var, initial_state[v]) for v, var in nodes.items()]
    solver.clique = 0   # clique cuts are not helpful

    # Solve the linear program.
    solver_args = dict()
    new_constraints = True
    status = None
    while new_constraints:
        if max_time is not None:
            time_expired = time.time() - start_time
            if time_expired > max_time:
                if status == OptimizationStatus.OPTIMAL:
                    status = OptimizationStatus.FEASIBLE
                break
            solver_args = {'max_seconds': max_time - time_expired}

        status = solver.optimize(**solver_args)
        new_constraints = False
        if verbosity:
            print("Current lower bound: ", solver.objective_value + bqm.offset, "\n")

        if cycle_cuts > 0 and not integer_solve:
            for generator in cut_generators:
                if generator.generate_constrs(solver):
                    new_constraints = True

    # Check for feasibility/optimality and extract state.
    if status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:

        optimality = status == OptimizationStatus.OPTIMAL

        if vartype == dimod.BINARY:
            state = {v: nodes[v].x for v in bqm.variables}
            coupler_state = {e: couplers[e].x for e in bqm.quadratic.keys()}
        else:
            state = {v: 2*nodes[v].x-1 for v in bqm.variables}
            coupler_state = {e: 2*couplers[e].x-1 for e in bqm.quadratic.keys()}
        info = {'state': state, 'coupler_state': coupler_state, 'optimality': optimality}

        if integer_solve:
            bound = solver.objective_bound + bqm.offset
            upper_bound = solver.objective_value + bqm.offset
            info['upper_bound'] = upper_bound
        else:
            bound = solver.objective_value + bqm.offset

        if verbosity:
            print("Final lower bound:", bound)
            if integer_solve:
                print("Final upper bound:", solver.objective_value + bqm.offset)

    else:
        bound = None
        info = {'state': None}

    return bound, info
