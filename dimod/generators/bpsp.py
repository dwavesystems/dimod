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

import dimod
import numpy.random as rnd

__all__ = ["random_binary_paint_shop_problem"]


def binary_paint_shop_problem(car_sequence):
    """
    Ising formulation for Binary Paint Shop Problem:
    - Streif, Michael, et al.
      "Beating classical heuristics for the binary paint shop
      problem with the quantum approximate optimization algorithm."
      Physical Review A 104.1 (2021): 012403.

    Input:
    - car_sequence: list of car labels
    - seed: Seed for the random number generator

    Returns:
    - BQM: Ising model where each variable is a car, and the value indicates what color it gets painted first
    """

    car_counter = {car: 0 for car in car_sequence}
    assert len(car_sequence) == 2 * len(car_counter)

    bqm = dimod.BQM(vartype=dimod.SPIN)
    for car1, car2 in zip(car_sequence, car_sequence[1:]):
        bqm += (
            (-1) ** (car_counter[car1] + car_counter[car2] + 1)
            * dimod.Spin(car1)
            * dimod.Spin(car2)
        )
        car_counter[car1] += 1

    return bqm


def random_binary_paint_shop_problem(n_cars, seed=None):
    """
    Generates a random binary paint shop problem

    Binary Paint Shop Problem:
    - Epping, Th, Winfried Hochstättler, and Peter Oertel.
      "Complexity results on a paint shop problem."
      Discrete Applied Mathematics 136.2-3 (2004): 217-226.

    Input:

    - n_cars (int): Number of cars
    - seed: Seed for the random number generator
    """

    rnd.seed(seed)
    car_sequence = list(range(n_cars)) * 2
    rnd.shuffle(car_sequence)

    return binary_paint_shop_problem(car_sequence)


def sample_to_coloring(sample, car_sequence):
    """Given a bqm sample, color the car sequence and return BPSP objective
    
    Input:
        - sample: solution to `binary_paint_shop_problem` mapping variables to color
        - car_sequence: sequence of appearence of cars

    Returns:
        - car_colors: colored sequence
        - color_changes: number of color changes (objective of BPSP problem)
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
