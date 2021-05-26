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
# ================================================================================================
from dimod.vartypes import Vartype


def fix_variables(bqm, sampling_mode=True):
    """Determine assignments for some variables of a binary quadratic model.

    Roof duality finds a lower bound for the minimum of a quadratic polynomial. It
    can also find minimizing assignments for some of the polynomial's variables;
    these fixed variables take the same values in all optimal solutions [BHT]_ [BH]_.
    A quadratic pseudo-Boolean function can be represented as a network to find
    the lower bound through network-flow computations. `fix_variables` uses maximum
    flow in the implication network to correctly fix variables. Consequently, you can
    find an assignment for the remaining variables that attains the optimal value.

    Args:
        bqm (:obj:`.BinaryQuadraticModel`)
            A binary quadratic model.

        sampling_mode (bool, optional, default=True):
            In sampling mode, only roof-duality is used. When `sampling_mode` is false, strongly
            connected components are used to fix more variables, but in some optimal solutions
            these variables may take different values.

    Returns:
        dict: Variable assignments for some variables of the specified binary quadratic model.

    .. [BHT] Boros, E., P.L. Hammer, G. Tavares. Preprocessing of Unconstraint Quadratic Binary
        Optimization. Rutcor Research Report 10-2006, April, 2006.

    .. [BH] Boros, E., P.L. Hammer. Pseudo-Boolean optimization. Discrete Applied Mathematics 123,
        (2002), pp. 155-225

    """
    # todo: add temporary dependency on dwave-preprocessing
    raise NotImplementedError
