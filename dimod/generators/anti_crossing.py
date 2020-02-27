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


def anti_crossing(num_qubits):
    """ given number of qubits, the code will generate instances with perturbative anticrossing. These
        instances are known to be hard for Quantum annealing. For more information please refer to
        [Phys. Rev. A 85, 032303] and [Phys. Rev. A 96, 042322] """

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
    return h, J
