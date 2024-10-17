# Copyright 2024 D-Wave Systems Inc.
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

__all__ = [
    "binary_paint_shop_problem",
    "random_binary_paint_shop_problem",
]

import collections.abc

import numpy.random

import dimod


def binary_paint_shop_problem(
        car_sequence: collections.abc.Sequence[collections.abc.Hashable],
        ) -> dimod.BinaryQuadraticModel:
    """Generates a binary quadratic model representing a binary paint shop problem.

    Ising formulation for Binary Paint Shop Problem:
    - Streif, Michael, et al.
      "Beating classical heuristics for the binary paint shop
      problem with the quantum approximate optimization algorithm."
      Physical Review A 104.1 (2021): 012403.

    Args:
        car_sequence: A list of car labels.

    Returns:
        Ising model where each variable is a car, and the value indicates what color it gets painted first.
    """

    car_counter = {car: 0 for car in car_sequence}
    if len(car_sequence) != 2 * len(car_counter):
        raise ValueError("Car labels are not unique")

    bqm = dimod.BinaryQuadraticModel(vartype=dimod.SPIN)
    for car1, car2 in zip(car_sequence, car_sequence[1:]):
        if car1 != car2:
            bqm.add_quadratic(car1, car2, (-1) ** (car_counter[car1] + car_counter[car2] + 1))
        car_counter[car1] += 1
    return bqm


def random_binary_paint_shop_problem(n_cars: int, seed: int = None) -> dimod.BinaryQuadraticModel:
    """Generates a random binary paint shop problem.

    Binary Paint Shop Problem:
    - Epping, Th, Winfried Hochstättler, and Peter Oertel.
      "Complexity results on a paint shop problem."
      Discrete Applied Mathematics 136.2-3 (2004): 217-226.

    Args:
        n_cars: Number of cars.
        seed: Seed for the random number generator.

    Returns:
        Ising model for the random binary paint shop problem.
    """

    rng = numpy.random.default_rng(seed)
    car_sequence = list(range(n_cars)) * 2
    rng.shuffle(car_sequence)

    return binary_paint_shop_problem(car_sequence)


def sample_to_coloring(
        sample: collections.abc.Mapping,
        car_sequence: collections.abc.Sequence[collections.abc.Hashable],
        ) -> tuple[list, int]:
    """Colors the car sequence based on a bqm sample and returns BPSP objective.

    Args:
        sample: Solution to `binary_paint_shop_problem` mapping variables to color.
        car_sequence: Sequence of appearence of cars.

    Returns:
        car_colors: Colored car sequence.
        color_changes: Number of color changes (objective of BPSP problem).
    """
    visited_cars = set()
    car_colors = []
    for car in car_sequence:
        if car not in visited_cars:
            car_colors.append((sample[car]+1)//2)
        else:
            car_colors.append((-1*sample[car]+1)//2)
        visited_cars.add(car)
    color_changes = sum(abs(col1 - col2) for col1, col2 in zip(car_colors, car_colors[1:]))
    return car_colors, color_changes
