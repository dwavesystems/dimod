// Copyright 2021 D-Wave Systems Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "dimod/binary_quadratic_model.h"

namespace dimod {

/**
 * Fixes the variables of an BinaryQuadraticModel.
 *
 * @param bqm BinaryQuadraticModel to find minimizing variable assignments for
 * @param strict When true, only the variables corresponding to strong persistencies
 *      are fixed. When false, the function tries to fix all the variables
 *      corresponding to strong and weak persistencies. Also when false, variables 
 *      that do not contribute any coefficient to the posiform are set to 1. This
 *      may happen if their bias in the original QUBO was 0 or if they were flushed
 *      to zero when converted to the posiform.
 * @param offset The bqm's offset, used to calculate the lower bound. Defaults to 0.
 */
template <class V, class B>
std::pair<double, std::vector<std::pair<int, int>>>
fixQuboVariables(dimod::BinaryQuadraticModel<B, V> &bqm, bool strict, double offset=0.0);

} // namespace dimod
