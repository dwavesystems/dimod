# Copyright 2021 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dimod.vartypes import Vartype

from dwave.preprocessing.cyfix_variables import fix_variables_wrapper

def roof_duality(bqm, *, strict=True):
    """Determine a lower bound for a binary quadratic model's energy, as well as 
    minimizing assignments for some of its variables, using the roof duality 
    algorithm.

    Args:
        bqm (:class:`.BinaryQuadraticModel`):
            A binary quadratic model.

        strict (bool, optional, default=True):
            If True, only fixes variables for which assignments are true for all 
            minimizing points (strong persistency). If False, also fixes variables 
            for which the assignments are true for some but not all minimizing 
            points (weak persistency).

    Returns:
        (float, dict) A 2-tuple containing:
        
            float: Lower bound for the energy of ``bqm``.
            
            dict: Variable assignments for some variables of ``bqm``.

    Examples:
        This example creates a binary quadratic model with a single ground state
        and finds both an energy lower bound and a minimizing assignment to the 
        model's single variable. 

        >>> import dimod
        >>> from dwave.preprocessing.lower_bounds import roof_duality
        >>> bqm = dimod.BinaryQuadraticModel.from_ising({'a': 1.0}, {})
        >>> roof_duality(bqm)
        (-1.0, {'a': -1})

        This example has two ground states, :math:`a=b=-1` and :math:`a=b=1`, with
        no variable having a single value for all ground states, so neither variable
        is fixed.

        >>> bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
        >>> bqm.add_interaction('a', 'b', -1.0)
        >>> roof_duality(bqm) # doctest: +SKIP
        (-1.0, {})

        This example sets ``strict`` to False, so variables are fixed to an assignment
        that attains the ground state.

        >>> bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
        >>> bqm.add_interaction('a', 'b', -1.0)
        >>> roof_duality(bqm, strict=False) # doctest: +SKIP
        (-1.0, {'a': -1, 'b': -1})

    """
    bqm_ = bqm

    if bqm_.vartype is Vartype.SPIN:
        bqm_ = bqm.change_vartype(Vartype.BINARY, inplace=False)

    linear = bqm_.linear

    if all(v in linear for v in range(len(bqm_))):
        # we can work with the binary form of the bqm directly
        lower_bound, fixed = fix_variables_wrapper(bqm_, strict)
    else:
        inverse_mapping = dict(enumerate(linear))
        mapping = {v: i for i, v in inverse_mapping.items()}

        # no need to make a copy if we've already done so
        inplace = bqm.vartype is Vartype.SPIN
        bqm_ = bqm_.relabel_variables(mapping, inplace=inplace)

        lower_bound, fixed = fix_variables_wrapper(bqm_, strict)
        fixed = {inverse_mapping[v]: val for v, val in fixed.items()}

    if bqm.vartype is Vartype.SPIN:
        return lower_bound, {v: 2*val - 1 for v, val in fixed.items()}
    else:
        return lower_bound, fixed
