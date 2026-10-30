# Copyright 2022 D-Wave Systems Inc.
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

import enum

import dimod
import numpy as np


class Feasibility(enum.Enum):
    Infeasible = 0
    Feasible = 1
    Unknown = 2


class TechniqueFlags(enum.IntFlag):
    None_ = 0
    RemoveRedundantConstraints = 1 << 0
    RemoveSmallBiases = 1 << 1
    DomainPropagation = 1 << 2
    All = 0xffffffffffffffff
    Default = All


class cyPresolver:
    variables: dimod.variables.Variables

    def __init__(self, cqm: dimod.ConstrainedQuadraticModel, move: bool = ...): ...
    def add_techniques(self, techniques: TechniqueFlags) -> TechniqueFlags: ...
    def apply(self) -> bool: ...
    def clear_model(self) -> None: ...
    def copy_model(self) -> dimod.ConstrainedQuadraticModel: ...
    def detach_model(self) -> dimod.ConstrainedQuadraticModel: ...
    def feasibility(self) -> Feasibility: ...
    def normalize(self) -> bool: ...
    def presolve(self, *, time_limit_s: float = float("inf")) -> bool: ...
    def restore_samples(self, samples_like: dimod.typing.SamplesLike) -> np.ndarray: ...
    def set_techniques(self, techniques: TechniqueFlags) -> TechniqueFlags: ...
    def techniques(self) -> TechniqueFlags: ...
