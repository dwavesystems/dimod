import numpy as np
import dimod
from dimod.vartypes import Vartype


def find_contractible_variables_naive(bqm):
    """
    Given a bqm, find pairs of variables (x, y) such that ground states satisfy x=y (or x = ~y).

    Input:
        bqm: a BinaryQuadraticModel.

    Returns:
        contractible_variables: a dictionary with keys (x,y) and values True if x=y and False if x=-y in the ground
        states.

    Method: if the magnitude of an interaction J_xy is larger than the magnitude of the sum of the magnitudes of the
    other couplers and bias of either x or y, then x and y should be fixed.
    """

    # Use the spin version
    bqm = bqm.spin

    contractible_variables = dict()
    magnitudes = dict()
    for x, neighbours in bqm.adj.items():
        magnitudes[x] = sum([np.abs(j) for j in neighbours.values()])
        if x in bqm.linear.keys():
            magnitudes[x] += np.abs(bqm.linear[x])

    for (x, y), val in bqm.quadratic.items():
        if np.abs(val) > min(magnitudes[x], magnitudes[y]) - np.abs(val):
            # value in contractible_variables is True if x=y, False if x=-y.
            contractible_variables[(x, y)] = np.sign(val) < 0

    return contractible_variables


def find_and_contract_all_variables_naive(bqm):
    """
    Given a bqm, naively find pairs of variables (x, y) such that ground states satisfy x=y (or x = ~y) and contract
    them, repeating until no further contractions are possible.

    Input:
        bqm: a BinaryQuadraticModel.

    Returns:
        bqm2: a BinaryQuadraticModel with variables contracted.

        variable_map: a dictionary indicating variable contractions that took place.
            variable_map[u] = (v, uv_equal) indicates that variable u was contracted to variable v, with u=v if
            uv_equal=1 and u=~v otherwise.

    Example:
        This example creates a binary quadratic model with a strong interaction between variables 0 and 3,
        and contracts that pair of variables to a single variable. After sampling from the contracted bqm,
        solutions are expanded to produce solutions to the original bqm.

        >>> J = {(u, v): 1 for (u, v) in [(0, 1), (0, 2), (1, 2)]}
        >>> J[(0, 3)] = -10
        >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, J)

        >>> contracted_bqm, variable_map, _ = find_and_contract_all_variables_roof_naive(bqm, sampling_mode=True)
        >>> print(contracted_bqm)

        >>> contracted_sampleset = dimod.ExactSolver().sample(contracted_bqm)
        >>> sampleset = uncontract_solution(contracted_sampleset, variable_map)
        >>> print(sampleset)

    """

    bqm2 = bqm.copy()
    variable_map = {u: (u, True) for u in bqm2.linear}
    finished = False
    while not finished:

        finished = True
        contractible_variables = find_contractible_variables_naive(bqm2)

        if len(contractible_variables):
            # found something to contract.
            finished = False
            new_map = bqm2.contract_all_variables(contractible_variables)
            variable_map = {u: (new_map[v][0], new_map[v][1] == uv_equal) for u, (v, uv_equal) in variable_map.items()}

    return bqm2, variable_map


def find_contractible_variables_roof_duality(bqm, sampling_mode=True, vars_to_test=None):
    """
    Given a bqm, find pairs of variables (x, y) such that ground states satisfy x=y (or x = ~y) by "extended roof
    duality".


    Args:
        bqm (:obj:`.BinaryQuadraticModel`)
            A binary quadratic model.

        sampling_mode (bool, optional, default=True):
            In sampling mode, only roof-duality is used. When `sampling_mode` is false, strongly
            connected components are used to fix more variables, but in some optimal solutions
            these variables may take different values.

        vars_to_test (list, optional, default=None):
            List of variables to probe. If not provided, all variables are tested.

    Returns:
        contractible_variables: a dictionary with keys (x,y) and values True if x=y and False if x=-y in the ground
        states.

        fixable_variables: a dictionary of variable assignments that can be fixed in the ground states.

    Method: pick a variable x, and probe it by fixing to both 1 and then -1. Run roof duality on both of these
    smaller problems. If y gets fixed in both smaller problems, we can fix y (to x, ~x, 1, or -1).
    Details are described in https://pub.ist.ac.at/~vnk/papers/RKLS-CVPR07.pdf.

    This implementation is very simple and inefficient. It seems that there are several open source implementations
    available, all of which wrap Kolmogorov's original roof duality implementation. (For example, opengm.)
    """

    if vars_to_test is None:
        # test all vars.
        vars_to_test = bqm.variables

    not_one = -1 if bqm.vartype == Vartype.SPIN else 0

    contractible_variables = dict()
    fixable_variables = dict()
    for x in vars_to_test:
        bqm2 = bqm.copy()
        bqm2.fix_variable(x, 1)
        fixed_vars = dimod.roof_duality.fix_variables(bqm2, sampling_mode=sampling_mode)
        if fixed_vars:
            bqm2 = bqm.copy()
            bqm2.fix_variable(x, not_one)
            fixed_vars_2 = dimod.roof_duality.fix_variables(bqm2, sampling_mode=sampling_mode)
            for y in fixed_vars_2:
                if y in fixed_vars and (y, x) not in contractible_variables and y not in fixable_variables:
                    if fixed_vars[y] == fixed_vars_2[y]:
                        # in this case y should simply be fixed to its value.
                        fixable_variables[y] = fixed_vars[y]
                    # otherwise, y is fixed to x or ~x.
                    contractible_variables[(x, y)] = (fixed_vars[y] == 1)

    return contractible_variables, fixable_variables


def find_and_contract_all_variables_roof_duality(bqm, sampling_mode=True):
    """
    Given a bqm, find all pairs of variables (x, y) such that ground states satisfy x=y (or x = ~y) by "extended roof
    duality" and contract them, repeating until no further contractions are possible.

    Input:
        bqm: a BinaryQuadraticModel.

        sampling_mode: boolean indicating if variables are contractible for all ground states (sampling_mode=True) or
            some ground state (sampling_mode=False).

    Returns:
        bqm2: a BinaryQuadraticModel with variables contracted.

        variable_map: a dictionary indicating variable contractions that took place.
            variable_map[u] = (v, uv_equal) indicates that variable u was contracted to variable v, with u=v if
            uv_equal=1 and u=~v otherwise.

        fixable_variables: a dictionary of variable assignments that were fixed in the ground states.

    Example:
        This example creates a binary quadratic model with a strong interaction between variables 0 and 3,
        and contracts that pair of variables to a single variable. After sampling from the contracted bqm,
        solutions are expanded to produce solutions to the original bqm.

        >>> J = {(u, v): 1 for (u, v) in [(0, 1), (0, 2), (1, 2)]}
        >>> J[(0, 3)] = -10
        >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, J)

        >>> contracted_bqm, variable_map, _ = find_and_contract_all_variables_roof_duality(bqm, sampling_mode=True)
        >>> print(contracted_bqm)

        >>> contracted_sampleset = dimod.ExactSolver().sample(contracted_bqm)
        >>> sampleset = uncontract_solution(contracted_sampleset, variable_map)
        >>> print(sampleset)


    Method: pick a variable to probe by extended roof duality, and if it contractible, contract it right away.
    Repeat until no contractions are possible.
    (When there are many contractible variables, this method is much faster than first identifying all contractible
    variables, contracting them, and repeating.)

    Note: it is possible to modify the roof duality algorithm to incorporate probing more directly, which would
    accompligh the same thing as this function but be a lot faster. See:
    https://pub.ist.ac.at/~vnk/papers/RKLS-CVPR07.pdf, section 3.1.1.
    """

    bqm2 = bqm.copy()
    variable_map = {u: (u, True) for u in bqm2.linear}

    # First find globally fixable variables.
    fixed_variables = dimod.roof_duality.fix_variables(bqm2, sampling_mode=sampling_mode)
    bqm2.fix_variables(fixed_variables)

    finished = False
    while not finished:
        finished = True
        # choose a good variable order: order by size of interaction magnitude.
        max_e = {v: max(abs(e) for e in nbrs.values()) if nbrs else 0 for v, nbrs in bqm.adj.items()}
        all_vars = sorted(max_e.items(), key=lambda k: -k[1])
        all_vars = [k for k, v in all_vars]   # variables only
        for x in all_vars:
            if x in bqm2.variables:
                contractible_variables, fixable_variables = \
                    find_contractible_variables_roof_duality(bqm2, sampling_mode=sampling_mode, vars_to_test=[x])

                if len(fixed_variables):
                    # found something to fix.
                    bqm2.fix_variables(fixable_variables)
                    fixed_variables.update(fixable_variables)

                if len(contractible_variables):
                    # found something to contract.
                    finished = False
                    new_map = bqm2.contract_all_variables(contractible_variables)
                    variable_map = {u: (new_map[v][0], new_map[v][1] == uv_equal) for u, (v, uv_equal) in
                                    variable_map.items()}

    # final check for a single fixable variable
    if len(bqm2.variables)==1:
        fixable_variables = dimod.fix_variables(bqm2, sampling_mode=sampling_mode)
        fixed_variables.update(fixable_variables)
        bqm2.fix_variables(fixable_variables)

    # update variable_map to account for fixed_variables
    # That is, variables that were mapped to fixed variables should now be fixed.
    for u, (v, uv_equal) in variable_map.items():
        if v in fixed_variables:
            if uv_equal:
                fixed_variables[u] = fixed_variables[v]
            else:
                fixed_variables[u] = -fixed_variables[v] if bqm.vartype == Vartype.SPIN else 1 - fixed_variables[v]
            variable_map[u] = (u, True)

    return bqm2, variable_map, fixed_variables


def uncontract_solution(sampleset, variable_map):
    """Translation solutions to a contracted bqm into solutions for an uncontracted bqm.

    Input:
        sampleset: a dimod.SampleSet for the contracted bqm.

        variable_map: a dictionary indicating variable contractions that took place.
            variable_map[u] = (v, uv_equal) indicates that variable u was contracted to variable v, with u=v if
            uv_equal=1 and u=~v otherwise.

    Returns:
        new_sampleset: a dimod.SampleSet for the uncontracted bqm.


    """

    if sampleset.vartype != Vartype.SPIN:
        raise NotImplementedError

    samples, labels = dimod.sampleset.as_samples(sampleset)
    new_labels = list(variable_map.keys())
    new_samples = np.zeros((samples.shape[0], len(variable_map)))
    label_index = {v: labels.index(v) for v in labels}
    for i, sample in enumerate(samples):
        if sampleset.vartype == Vartype.SPIN:
            new_samples[i] = [sample[label_index[v]]*(2*uv_equal - 1) for u, (v, uv_equal) in variable_map.items()]
        else:
            new_samples[i] = [int(sample[label_index[v]] == uv_equal) for u, (v, uv_equal) in variable_map.items()]

    new_sampleset = dimod.sampleset.SampleSet.from_samples((new_samples, new_labels), vartype=Vartype.SPIN,
                                                           **sampleset.data_vectors)
    return new_sampleset
