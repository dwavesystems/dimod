from dimod.roof_duality._fix_variables import fix_variables_wrapper
from dimod.vartypes import Vartype


def fix_variables(bqm, sampling_mode=True):
    """Determine assignments for some bqm variables.

    fix_variables uses maximum flow in the implication network to correctly fix variables (that is,
    one can find an assignment for the other variables that attains the optimal value). The
    variables that roof duality fixes will take the same values in all optimal solutions.

    Args:
        bqm (:obj:`.BinaryQuadraticModel`)
            A binary quadratic model.

        sampling_mode (bool, optional, default=True):
            Setting sampling_model to False can fix more variables, but in some optimal solutions
            these variables may take different values.

    Returns:
        dict: Variable assignments for some bqm variables.

    Examples:

        >>> bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
        >>> bqm.add_variable('a', 1.0)
        >>> dimod.fix_variables(bqm)
        {'a': -1}

        There are two ground states, so no variables fixed.

        >>> bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
        >>> bqm.add_interaction('a', 'b', -1.0)
        >>> dimod.fix_variables(bqm)
        {}

        With sampling_model off, additional variables are fixed.

        >>> bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
        >>> bqm.add_interaction('a', 'b', -1.0)
        >>> dimod.fix_variables(bqm, sampling_mode=False) # doctest: +SKIP
        {'a': 1, 'b': 1}

    """
    if sampling_mode:
        method = 2  # roof-duality only
    else:
        method = 1  # roof-duality and strongly connected components

    linear = bqm.linear
    if all(v in linear for v in range(len(bqm))):
        # we can work with the binary form of the bqm directly
        fixed = fix_variables_wrapper(bqm.binary, method)
    else:
        try:
            inverse_mapping = dict(enumerate(sorted(linear)))
        except TypeError:
            # in python3 unlike types cannot be sorted
            inverse_mapping = dict(enumerate(linear))
        mapping = {v: i for i, v in inverse_mapping.items()}

        fixed = fix_variables_wrapper(bqm.relabel_variables(mapping, inplace=False).binary, method)
        fixed = {inverse_mapping[v]: val for v, val in fixed.items()}

    if bqm.vartype is Vartype.SPIN:
        return {v: 2*val - 1 for v, val in fixed.items()}
    else:
        return fixed
