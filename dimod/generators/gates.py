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
from itertools import starmap, product
from typing import Optional

from dimod.binary.binary_quadratic_model import BinaryQuadraticModel, quicksum
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
    """Generate a binary quadratic model with ground states corresponding to an
    AND gate.

    Args:
        in0: Variable label for one of the inputs.
        in1: Variable label for one of the inputs.
        out: Variable label for the output.
        strength: Energy of the lowest-energy infeasible state.

    Returns:
        A binary quadratic model with ground states corresponding to an AND
        gate. The model has three variables and three interactions.

    Examples:
        >>> bqm = dimod.generators.and_gate('x1', 'x2', 'z')
        >>> print(dimod.ExactSolver().sample(bqm).lowest())
          x1 x2  z energy num_oc.
        0  0  0  0    0.0       1
        1  1  0  0    0.0       1
        2  0  1  0    0.0       1
        3  1  1  1    0.0       1
        ['BINARY', 4 rows, 4 samples, 3 variables]

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
    """Generate a binary quadratic model with ground states corresponding to a
    full adder gate.

    Args:
        in0: Variable label for one of the inputs.
        in1: Variable label for one of the inputs.
        in2: Variable label for one of the inputs
        sum_: Variable label for the sum output.
        carry: Variable label for the carry output.
        strength: Energy of the lowest-energy infeasible state.

    Returns:
        A binary quadratic model with ground states corresponding to a full
        adder gate. The model has five variables and ten interactions.

    Examples:
        >>> bqm = dimod.generators.fulladder_gate('a1', 'a2', 'a3', 's', 'c')
        >>> print(dimod.ExactSolver().sample(bqm).lowest())
          a1 a2 a3  c  s energy num_oc.
        0  0  0  0  0  0    0.0       1
        1  0  0  1  0  1    0.0       1
        2  0  1  0  0  1    0.0       1
        3  1  0  0  0  1    0.0       1
        4  1  1  1  1  1    0.0       1
        5  1  0  1  1  0    0.0       1
        6  0  1  1  1  0    0.0       1
        7  1  1  0  1  0    0.0       1
        ['BINARY', 8 rows, 8 samples, 5 variables]

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
    """Generate a binary quadratic model with ground states corresponding to a
    half adder gate.

    Args:
        in0: Variable label for one of the inputs.
        in1: Variable label for one of the inputs.
        sum_: Variable label for the sum output.
        carry: Variable label for the carry output.
        strength: Energy of the lowest-energy infeasible state.

    Returns:
        A binary quadratic model with ground states corresponding to a half
        adder gate. The model has four variables and six interactions.

    Examples:
        >>> bqm = dimod.generators.halfadder_gate('a1', 'a2', 's', 'c')
        >>> print(dimod.ExactSolver().sample(bqm).lowest())
          a1 a2  c  s energy num_oc.
        0  0  0  0  0    0.0       1
        1  0  1  0  1    0.0       1
        2  1  0  0  1    0.0       1
        3  1  1  1  0    0.0       1
        ['BINARY', 4 rows, 4 samples, 4 variables]

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
    """Generate a binary quadratic model with ground states corresponding to an
    OR gate.

    Args:
        in0: Variable label for one of the inputs.
        in1: Variable label for one of the inputs.
        out: Variable label for the output.
        strength: Energy of the lowest-energy infeasible state.

    Returns:
        A binary quadratic model with ground states corresponding to an OR
        gate. The model has three variables and three interactions.

    Examples:
        >>> bqm = dimod.generators.or_gate('x1', 'x2', 'z')
        >>> print(dimod.ExactSolver().sample(bqm).lowest())
          x1 x2  z energy num_oc.
        0  0  0  0    0.0       1
        1  0  1  1    0.0       1
        2  1  1  1    0.0       1
        3  1  0  1    0.0       1
        ['BINARY', 4 rows, 4 samples, 3 variables]

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
    """Generate a binary quadratic model with ground states corresponding to an
    XOR gate.

    Note that it is not possible to construct a binary quadratic model with
    only three variables for an XOR gate.

    Args:
        in0: Variable label for one of the inputs.
        in1: Variable label for one of the inputs.
        out: Variable label for the output.
        aux: Variable label for an auxiliary variable.
        strength: Energy of the lowest-energy infeasible state.

    Returns:
        A binary quadratic model with ground states corresponding to an XOR
        gate. The model has four variables and six interactions.

    Examples:
        >>> bqm = dimod.generators.xor_gate('x1', 'x2', 'z', 'a')
        >>> print(dimod.ExactSolver().sample(bqm).lowest())
           a x1 x2  z energy num_oc.
        0  0  0  0  0    0.0       1
        1  0  0  1  1    0.0       1
        2  0  1  0  1    0.0       1
        3  1  1  1  0    0.0       1
        ['BINARY', 4 rows, 4 samples, 4 variables]

    """
    # the sum of a halfadder is XOR
    return halfadder_gate(in0, in1, out, aux, strength=strength)

def multiplication_circuit(num_arg1_bits: int, num_arg2_bits: Optional[int] = None) -> BinaryQuadraticModel:
    """Generate a binary quadratic model with ground states corresponding to
    a multiplication circuit.

    The generated BQM represents the binary multiplication :math:`ab=p`,
    where the args are binary variables of length ``num_arg1_bits`` and
    ``num_arg2_bits``; for example, :math:`2^ma_{num_arg1_bits} + ... + 4a_2 + 2a_1 + a0`.
    The square below shows a graphic representation of the circuit:

    .. code-block::

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
        num_arg1_bits: Number of bits in the first argument.
        num_arg2_bits: Number of bits in the second argument. If None, set to
            ``num_arg1_bits``.
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
        {'p0': 0, 'p1': 1, 'p2': 1, 'p3': 0}
    """

    if num_arg1_bits < 1:
        raise ValueError("num_arg1_bits must be a positive integer")

    num_arg2_bits = num_arg2_bits or num_arg1_bits

    if num_arg2_bits < 1:
        raise ValueError("the arg2 must have a positive size")

    num_product_bits = num_arg1_bits +num_arg2_bits

    # throughout, we will use the following convention:
    #   i to refer to the bits of arg1
    #   j to refer to the bits of arg2

    def AND(i, j):
        return f'and{i},{j}' if i or j else 'p0'

    def SUM(i, j):
        return (
            f'p{i}'
            if j == 0
            else (
                f'p{i + j}'
                if i == num_arg1_bits - 1
                else f'sum{i},{j}'))

    def CARRY(i, j):
        return (
            f'p{num_product_bits - 1}'
            if i + j == num_product_bits - 2
            else f'carry{i},{j}')

    def gate(i, j):
        inputs = [AND(i, j)]
        bqm = and_gate(f'a{i}', f'b{j}', inputs[0])
        if i > 0:
            if j < num_arg2_bits - 1:
                inputs.append(SUM(i-1, j+1) if i > 1 else AND(0, j+1))
            elif i > 1:
                inputs.append(CARRY(i - 1, j))
            if j > 0:
                inputs.append(CARRY(i, j-1))

        l = len(inputs)
        if l > 1:
            outputs = SUM(i, j), CARRY(i, j)
            bqm.update((halfadder_gate if l == 2 else fulladder_gate)(*inputs, *outputs))

        return bqm

    return quicksum(starmap(gate, product(range(num_arg1_bits),
                                          range(num_arg2_bits))))
