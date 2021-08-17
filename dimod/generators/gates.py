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


from dimod.binary.binary_quadratic_model import BinaryQuadraticModel
from dimod.typing import Variable
from dimod.vartypes import Vartype


__all__ = ['and_gate',
           'fulladder_gate',
           'halfadder_gate',
           'or_gate',
           'xor_gate',
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
