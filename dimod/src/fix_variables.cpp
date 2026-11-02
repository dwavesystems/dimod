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

#include "dimod/roof_duality/fix_variables.h"

#include "roof_duality/fix_variables_impl.h"

namespace dimod {

template <class V, class B>
std::pair<double, std::vector<std::pair<int, int>>>
fixQuboVariables(dimod::BinaryQuadraticModel<B, V> &bqm, bool strict, double offset) {
  int num_bqm_variables = bqm.num_variables();
  detail_::PosiformInfo<dimod::BinaryQuadraticModel<B, V>, capacity_type> posiform_info(bqm);
  std::vector<std::pair<int, int>> fixed_variables;
  capacity_type max_flow = fixQuboVariables(posiform_info, num_bqm_variables, strict, fixed_variables);

  // The max_flow added with the constant term of the posiform should be the lower 
  // bound of the posiform, which should be equal to the lower bound of the bqm. 
  // But while creating the implication network and assigning capacities to its 
  // edges, we did not divide the corresponding coefficients of the posiform by 2,
  // thus when we convert the max flow to the minimum value of the posiform, we 
  // need to divide it by 2. The bqm offset also needs to be added since it was 
  // ignored in the posiform to reduce potential numerical errors.
  // See bottom of page 5 after equation 5 of the following paper.
  // Boros, Endre & Hammer, Peter & Tavares, Gabriel. (2006). Preprocessing of
  // unconstrained quadratic binary optimization. RUTCOR Research Report.
  double lower_bound = (posiform_info.getConstant() / posiform_info.getBiasConversionRatio())
                       + ((double)max_flow / (posiform_info.getBiasConversionRatio() * 2))
                       + offset;
  return {lower_bound, fixed_variables};
}

// As with the Presolver in presolve.cpp, we only instantiate what dimod uses:
// cyfix_variables.pyx always converts to BQM<float64, int32>. The long
// instantiation follows presolve.cpp's MSVC note; float is for the C++ tests.
template std::pair<double, std::vector<std::pair<int, int>>>
fixQuboVariables(dimod::BinaryQuadraticModel<double, int>&, bool, double);
template std::pair<double, std::vector<std::pair<int, int>>>
fixQuboVariables(dimod::BinaryQuadraticModel<double, long>&, bool, double);
template std::pair<double, std::vector<std::pair<int, int>>>
fixQuboVariables(dimod::BinaryQuadraticModel<float, int>&, bool, double);

} // namespace dimod
