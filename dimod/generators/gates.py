# Copyright 2021 D-Wave Systems Inc.
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

from collections import defaultdict
from typing import Optional
from dimod.binary.binary_quadratic_model import BinaryQuadraticModel
from dimod.typing import Variable
from dimod.vartypes import Vartype

__all__ = ['and_gate',
           'fulladder_gate',
           'halfadder_gate',
           'or_gate',
           'xor_gate',
           'multiplication_circuit',
           ]


def and_gate(in0: Variable, in1: Variable, out: Variable,
             *, strength: float = 1.0) -> BinaryQuadraticModel:
    """Return a binary quadratic model with ground states corresponding to an
    AND gate.

    Args:
        in0: The variable label for one of the inputs.
        in1: The variable label for one of the inputs.
        out: The variable label for the output.
        strength: The energy of the lowest-energy infeasible state.

    Returns:
        A binary quadratic model with ground states corresponding to an AND
        gate. The model has three variables and three interactions.

    """
    bqm = BinaryQuadraticModel(Vartype.BINARY)

    # add the variables (in order)
    bqm.add_variable(in0)
    bqm.add_variable(in1)
    bqm.add_variable(out, bias=3)

    # add the quadratic biases
    bqm.add_quadratic(in0, in1, 1)
    bqm.add_quadratic(in0, out, -2)
    bqm.add_quadratic(in1, out, -2)

    # the bqm currently has a strength of 1, so just need to scale
    if strength <= 0:
        raise ValueError("strength must be positive")
    bqm.scale(strength)

    return bqm


def fulladder_gate(in0: Variable, in1: Variable, in2: Variable, sum_: Variable, carry: Variable,
                   *, strength: float = 1.0) -> BinaryQuadraticModel:
    """Return a binary quadratic model with ground states corresponding to a
    full adder gate.

    Args:
        in0: The variable label for one of the inputs.
        in1: The variable label for one of the inputs.
        in2: The variable label for one of the inputs
        sum_: The variable label for the sum output.
        carry: The variable label for the carry output.
        strength: The energy of the lowest-energy infeasible state.

    Returns:
        A binary quadratic model with ground states corresponding to a full
        adder gate. The model has five variables and ten interactions.

    """
    bqm = BinaryQuadraticModel(Vartype.BINARY)

    # add the variables (in order)
    bqm.add_variable(in0, bias=1)
    bqm.add_variable(in1, bias=1)
    bqm.add_variable(in2, bias=1)
    bqm.add_variable(sum_, bias=1)
    bqm.add_variable(carry, bias=4)

    # add the quadratic biases
    bqm.add_quadratic(in0, in1, 2)
    bqm.add_quadratic(in0, in2, 2)
    bqm.add_quadratic(in0, sum_, -2)
    bqm.add_quadratic(in0, carry, -4)
    bqm.add_quadratic(in1, in2, 2)
    bqm.add_quadratic(in1, sum_, -2)
    bqm.add_quadratic(in1, carry, -4)
    bqm.add_quadratic(in2, sum_, -2)
    bqm.add_quadratic(in2, carry, -4)
    bqm.add_quadratic(sum_, carry, 4)

    # the bqm currently has a strength of 1, so just need to scale
    if strength <= 0:
        raise ValueError("strength must be positive")
    bqm.scale(strength)

    return bqm


def halfadder_gate(in0: Variable, in1: Variable, sum_: Variable, carry: Variable,
                   *, strength: float = 1.0) -> BinaryQuadraticModel:
    """Return a binary quadratic model with ground states corresponding to a
    half adder gate.

    Args:
        in0: The variable label for one of the inputs.
        in1: The variable label for one of the inputs.
        sum_: The variable label for the sum output.
        carry: The variable label for the carry output.
        strength: The energy of the lowest-energy infeasible state.

    Returns:
        A binary quadratic model with ground states corresponding to a half
        adder gate. The model has four variables and six interactions.

    """
    bqm = BinaryQuadraticModel(Vartype.BINARY)

    # add the variables (in order)
    bqm.add_variable(in0, bias=1)
    bqm.add_variable(in1, bias=1)
    bqm.add_variable(sum_, bias=1)
    bqm.add_variable(carry, bias=4)

    # add the quadratic biases
    bqm.add_quadratic(in0, in1, 2)
    bqm.add_quadratic(in0, sum_, -2)
    bqm.add_quadratic(in0, carry, -4)
    bqm.add_quadratic(in1, sum_, -2)
    bqm.add_quadratic(in1, carry, -4)
    bqm.add_quadratic(sum_, carry, 4)

    # the bqm currently has a strength of 1, so just need to scale
    if strength <= 0:
        raise ValueError("strength must be positive")
    bqm.scale(strength)

    return bqm


def or_gate(in0: Variable, in1: Variable, out: Variable,
            *, strength: float = 1.0) -> BinaryQuadraticModel:
    """Return a binary quadratic model with ground states corresponding to an
    OR gate.

    Args:
        in0: The variable label for one of the inputs.
        in1: The variable label for one of the inputs.
        out: The variable label for the output.
        strength: The energy of the lowest-energy infeasible state.

    Returns:
        A binary quadratic model with ground states corresponding to an OR
        gate. The model has three variables and three interactions.

    """
    bqm = BinaryQuadraticModel(Vartype.BINARY)

    # add the variables (in order)
    bqm.add_variable(in0, bias=1)
    bqm.add_variable(in1, bias=1)
    bqm.add_variable(out, bias=1)

    # add the quadratic biases
    bqm.add_quadratic(in0, in1, 1)
    bqm.add_quadratic(in0, out, -2)
    bqm.add_quadratic(in1, out, -2)

    # the bqm currently has a strength of 1, so just need to scale
    if strength <= 0:
        raise ValueError("strength must be positive")
    bqm.scale(strength)

    return bqm


def xor_gate(in0: Variable, in1: Variable, out: Variable, aux: Variable,
             *, strength: float = 1.0) -> BinaryQuadraticModel:
    """Return a binary quadratic model with ground states corresponding to an
    XOR gate.

    Note that it is not possible to construct a binary quadratic model with
    only three variables for an XOR gate.

    Args:
        in0: The variable label for one of the inputs.
        in1: The variable label for one of the inputs.
        out: The variable label for the output.
        aux: The variable label for an auxiliary variable.
        strength: The energy of the lowest-energy infeasible state.

    Returns:
        A binary quadratic model with ground states corresponding to an XOR
        gate. The model has four variables and six interactions.

    """
    # the sum of a halfadder is XOR
    return halfadder_gate(in0, in1, out, aux, strength=strength)

def multiplication_circuit(nbit: int, multiplicand_nbit: Optional[int] = None) -> BinaryQuadraticModel:
    """Return a binary quadratic model with ground states corresponding to
    a multiplication circuit.

    The generated BQM represents the binary multiplication :math:`ab=p`,
    where the multiplicands are binary variables of length `nbit`; for example,
    :math:`2^ma_{nbit} + ... + 4a_2 + 2a_1 + a0`.
    The square below shows a graphic representation of the circuit::
      ________________________________________________________________________________
      |                                         and20         and10         and00    |
      |                                           |             |             |      |
      |                           and21         add11──and11  add01──and01    |      |
      |                             |┌───────────┘|┌───────────┘|             |      |
      |             and22         add12──and12  add02──and02    |             |      |
      |               |┌───────────┘|┌───────────┘|             |             |      |
      |             add13─────────add03           |             |             |      |
      |  ┌───────────┘|             |             |             |             |      |
      | p5            p4            p3            p2            p1            p0     |
      --------------------------------------------------------------------------------

    Args:
        nbit: Number of bits in the multiplier.
        multiplicand_nbit: Number of bits in the multiplicand. If a false value is provided the arguments of equal size are used.
    Returns:
        A binary quadratic model with ground states corresponding to a
        multiplication circuit.
    Examples:
        This example creates a multiplication circuit BQM that multiplies two
        2-bit numbers. It fixes the multiplacands as :math:`a=2, b=3`
        (:math:`10` and :math:`11`) and uses a brute-force solver to find the
        product, :math:`p=6` (:math:`110`).

        >>> from dimod.generators import multiplication_circuit
        >>> from dimod import ExactSolver
        >>> bqm = multiplication_circuit(2)
        >>> for fixed_var, fixed_val in {'a0': 0, 'a1': 1, 'b0':1, 'b1': 1}.items():
        ...    bqm.fix_variable(fixed_var, fixed_val)
        >>> best = ExactSolver().sample(bqm).first
        >>> p = {key: best.sample[key] for key in best.sample.keys() if "p" in key}
        >>> print(p)
        {'p0': 0, 'p1': 1, 'p2': 1}
    """

    if nbit < 1:
        raise ValueError("nbit must be a positive integer")

    num_multiplier_bits = nbit
    num_multiplicand_bits = multiplicand_nbit or nbit

    if num_multiplicand_bits < 1:
        raise ValueError("the multiplicand must have a positive size")

    bqm = BinaryQuadraticModel(Vartype.BINARY)

    # throughout, we will use the following convention:
    #   i to refer to the bits of the multiplier
    #   j to refer to the bits of the multiplicand
    #   k to refer to the bits of the product

    # create the variables corresponding to the input and output wires for the circuit
    a = {i: 'a%d' % i for i in range(num_multiplier_bits)}
    b = {j: 'b%d' % j for j in range(num_multiplicand_bits)}
    p = {k: 'p%d' % k for k in  range(num_multiplier_bits + num_multiplicand_bits)}

    # we will want to store the internal variables somewhere
    AND = defaultdict(dict)  # the output of the AND gate associated with ai, bj is stored in AND[i][j]
    SUM = defaultdict(dict)  # the sum of the ADDER gate associated with ai, bj is stored in SUM[i][j]
    CARRY = defaultdict(dict)  # the carry of the ADDER gate associated with ai, bj is stored in CARRY[i][j]

    # we follow a shift adder
    for i in range(num_multiplier_bits):
        for j in range(num_multiplicand_bits):

            ai = a[i]
            bj = b[j]

            if i == 0 and j == 0:
                # in this case there are no inputs from lower bits, so our only input is the AND
                # gate. And since we only have one bit to add, we don't need an adder, have no
                # carry out
                andij = AND[i][j] = p[0]
                gate = and_gate(ai, bj, andij)
                bqm.update(gate)
                continue

            # we always need an AND gate
            andij = AND[i][j] = 'and%s,%s' % (i, j)
            gate = and_gate(ai, bj, andij)
            bqm.update(gate)

            # the number of inputs will determine the type of adder
            inputs = [andij]

            # determine if there is a carry in
            if i - 1 in CARRY and j in CARRY[i - 1]:
                inputs.append(CARRY[i - 1][j])

            # determine if there is a sum in
            if i - 1 in SUM and j + 1 in SUM[i - 1]:
                inputs.append(SUM[i - 1][j + 1])

            # ok, create adders if necessary
            if len(inputs) == 1:
                # we don't need an adder and we don't have a carry
                SUM[i][j] = andij
            elif len(inputs) == 2:
                # we need a HALFADDER so we have a sum and a carry
                if j == 0:
                    sumij = SUM[i][j] = p[i]
                else:
                    sumij = SUM[i][j] = 'sum%d,%d' % (i, j)

                carryij = CARRY[i][j] = 'carry%d,%d' % (i, j)
                gate = halfadder_gate(inputs[0], inputs[1], sumij, carryij)
                bqm.update(gate)
            else:
                assert len(inputs) == 3, 'unexpected number of inputs'
                # we need a FULLADDER so we have a sum and a carry
                if j == 0:
                    sumij = SUM[i][j] = p[i]
                else:
                    sumij = SUM[i][j] = 'sum%d,%d' % (i, j)

                carryij = CARRY[i][j] = 'carry%d,%d' % (i, j)
                gate = fulladder_gate(inputs[0], inputs[1], inputs[2], sumij, carryij)
                bqm.update(gate)

    # now we have a final row of full adders
    for col in range(num_multiplicand_bits - 1):
        inputs = [CARRY[num_multiplier_bits - 1][col], SUM[num_multiplier_bits - 1][col + 1]]

        if col == 0:
            sumout = p[num_multiplier_bits + col]
            carryout = CARRY[num_multiplier_bits][col] = 'carry%d,%d' % (num_multiplier_bits, col)
            gate = halfadder_gate(inputs[0], inputs[1], sumout, carryout)
            bqm.update(gate)
            continue

        inputs.append(CARRY[num_multiplier_bits][col - 1])

        sumout = p[num_multiplier_bits + col]
        if col < num_multiplicand_bits - 2:
            carryout = CARRY[num_multiplier_bits][col] = 'carry%d,%d' % (num_multiplier_bits, col)
        else:
            carryout = p[num_multiplier_bits + num_multiplicand_bits - 1]

        gate = fulladder_gate(inputs[0], inputs[1], inputs[2], sumout, carryout)
        bqm.update(gate)

    return bqm
