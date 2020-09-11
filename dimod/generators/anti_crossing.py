# Copyright 2020 D-Wave Systems Inc.
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

from dimod import BinaryQuadraticModel

__all__ = ['anti_crossing_clique', 'anti_crossing_loops']


def anti_crossing_clique(num_variables):
    """Anti crossing problems with a single clique.

    Given the number of variables, the code will generate a clique of size
    num_variables/2, each variable ferromagnetically interacting with a partner 
    variable with opposite bias. A single variable in the cluster will have 
    no bias applied.

    Args:
        num_variables (int):
            Number of variables used to generate the problem. Must be an even number
            greater than 6.

    Returns:
        :obj:`.BinaryQuadraticModel`.

    """

    if num_variables % 2 != 0 or num_variables < 6:
        raise ValueError('num_variables must be an even number > 6')

    bqm = BinaryQuadraticModel({}, {}, 0, 'SPIN')

    hf = int(num_variables / 2)
    for n in range(hf):
        for m in range(n + 1, hf):
            bqm.add_interaction(n, m, -1)

        bqm.add_interaction(n, n + hf, -1)

        bqm.add_variable(n, 1)
        bqm.add_variable(n + hf, -1)

    bqm.set_linear(1, 0)
    
    return bqm

def anti_crossing_loops(num_variables):
    """Anti crossing problems with two loops. These instances are copies of the
    instance studied in [DJA]_.

    Args:
        num_variables (int): 
            Number of variables used to generate the problem. Must be an even number
            greater than 8.

    Returns:
        :obj:`.BinaryQuadraticModel`.

    .. [DJA] Dickson, N., Johnson, M., Amin, M. et al. Thermally assisted
        quantum annealing of a 16-qubit problem. Nat Commun 4, 1903 (2013).
        https://doi.org/10.1038/ncomms2920 

    """

    bqm = BinaryQuadraticModel({}, {}, 0, 'SPIN')

    if num_variables % 2 != 0 or num_variables < 8:
        raise ValueError('num_variables must be an even number > 8')

    hf = int(num_variables / 4)

    for n in range(hf):   
        if n % 2 == 1:
            bqm.set_quadratic(n, n + hf, -1)

        bqm.set_quadratic(n, (n + 1) % hf, -1)
        bqm.set_quadratic(n + hf, (n + 1) % hf + hf, -1)

        bqm.set_quadratic(n, n + 2 * hf, -1)
        bqm.set_quadratic(n + hf, n + 3 * hf, -1)  

        bqm.add_variable(n, 1)
        bqm.add_variable(n + hf, 1)
        bqm.add_variable(n + 2 * hf, -1)
        bqm.add_variable(n + 3 * hf, -1)

    bqm.set_linear(0, 0) 
    bqm.set_linear(hf, 0)

    return bqm
