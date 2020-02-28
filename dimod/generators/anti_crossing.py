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



def anti_crossing_clique(num_qubits):
    """Anti crossing problems with a single clique.

    Given the number of qubits, the code will generate a clique of
    size num_qubits/2 each qubit ferromagnetically coupled to a partner qubit with opposite bias. Single qubit in
    the cluster will have no bias applied

    Args:
        num_qubits (int):
            number of qubits to use to generate the problem

    Returns:
        :obj:`.BinaryQuadraticModel`.

    """

    if num_qubits % 2 != 0 or num_qubits < 6:
        raise ValueError('num_qubits  must be an even number > 6')
    J = {}
    h = {}
    hf = int(num_qubits / 2)
    for n in range(hf):
        for m in range(n + 1, hf):
            J[(n, m)] = -1
        J[(n, n + hf)] = -1
        h[n] = 1
        h[n + hf] = -1
    h[1] = 0
    return BinaryQuadraticModel.from_ising(h, J)


def anti_crossing_loops(num_qubits):
    """ Anti crossing problems with two loops. These instances are copies of the instance studied in
    [Nature Comms. 4, 1903 (2013)]

    Args:
        num_qubits (int): number of qubits to use to generate the problem

    Returns:
        :obj:`.BinaryQuadraticModel`.

    """

    if num_qubits % 2 != 0 or num_qubits < 8:
        raise ValueError('num_qubits  must be an even number > 8')
    J = {}
    h = {}
    hf = int(num_qubits / 4)

    for n in range(hf):
        if n % 2 == 1:
            J[(n, n + hf)] = -1

        J[(n, (n + 1) % hf)] = -1
        J[(n + hf, (n + 1) % hf + hf)] = -1

        J[(n, n + 2 * hf)] = -1
        J[(n + hf, n + 3 * hf)] = -1

        h[n] = 1
        h[n + hf] = 1
        h[n + 2 * hf] = -1
        h[n + 3 * hf] = -1
    h[0] = 0
    h[hf] = 0

    J = {tuple(sorted(k)): v for k, v in J.items()}
    return BinaryQuadraticModel.from_ising(h, J)

