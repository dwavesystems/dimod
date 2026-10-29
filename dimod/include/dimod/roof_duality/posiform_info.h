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

#ifndef POSIFORM_INFORMATION_HPP_INCLUDED
#define POSIFORM_INFORMATION_HPP_INCLUDED

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * Contains all the information needed to recreate a posiform corresponding to 
 * a BQM. The intention is to reduce the memory footprint as much as possible, 
 * thus requiring some processing before the stored data can be used.
 *
 * For implementation details, see comments in the source code.
 *
 * For details on the algorithm, see : Boros, Endre & Hammer, Peter & Tavares, Gabriel.
 * (2006). Preprocessing of unconstrained quadratic binary optimization. RUTCOR
 * Research Report.
 */
template <class BQM, class coefficient_t> class PosiformInfo {
/*
If linear bias for variable X_i is negative with value -L it means in the
posiform we have the term  L * X_source * X_i' (X_i' being the complement of
X_i). X_source is X_0 in the paper mentioned below. X_0 was a variable added
to the posiform at a later step. We do not store X_0 explicitly as that would
make the indexing complicated, shifting the index of each posiform variable
by 1.

If a quadratic bias between X_i, X_j is negative with value -L with i < j it
means in the posiform we have the term L*X_i*X_j' (X_i' being the complement
of X_i).

For each variable X_i the posiform keeps the iterators in the provided bqm
for biases starting from X_j, (j>i) to X_n.

The linear biases are stored in integral format and can be used directly.
But the quadratic biases are not stored, instead the iterators in the bqm are
stored, thus the convertTo** function must be applied first on the biases
before use.

Note the number of variables and biases exposed correspond to the integral
version of the bqm matrix and may differ from the numbers corresponding
to the floating point based numbers as many biases may be flushed to zeroes.
*/
public:
  using coefficient_type = coefficient_t; // Must be a signed integral type.
  using bias_type = typename BQM::bias_type;
  using quadratic_iterator_type = typename BQM::const_neighborhood_iterator;
  using variable_type = typename BQM::index_type;

  /**
   * Construct a PosiformInfo from a binary quadratic model.
   */
  PosiformInfo(const BQM &bqm);

  /**
   * Get number of posiform variables.
   */
  inline int getNumVariables() { return _num_posiform_variables; }

  /**
   * Get number of posiform variables with a non-zero linear bias.
   */
  inline int getNumLinear() { return _num_linear_integral_biases; }


  /**
   * Get the linear bias of a posiform variable.
   */
  inline coefficient_type getLinear(int posiform_variable) {
    return _linear_integral_biases
        [_posiform_to_bqm_variable_map[posiform_variable]];
  }

  /**
   * Get number of quadratic terms a posiform variable contributes in.
   */
  inline int getNumQuadratic(int posiform_variable) {
    return _num_quadratic_integral_biases
        [_posiform_to_bqm_variable_map[posiform_variable]];
  }

  // For iterating over the quadratic biases, we need the
  // convertToPosiformCoefficient and mapVariableQuboToPosiform functions, since
  // the iterators belong to the bqm.

  /**
   * Get the neighbors of a posiform variable in the corresponding BQM.
   */
  inline std::pair<quadratic_iterator_type, quadratic_iterator_type>
  getQuadratic(int posiform_variable) {
    return _quadratic_iterators
        [_posiform_to_bqm_variable_map[posiform_variable]];
  }

  /**
   * Map a QUBO variable to a posiform variable. The number of QUBO variables 
   * and posiform variables may differ since the coefficient for a QUBO variable 
   * may turn out to be zero in a posiform. This may occur if the coefficient is 
   * flushed to zero during the conversion or if the variable did not have non-zero 
   * linear/quadratic biases in the QUBO.
   */
  inline int mapVariableQuboToPosiform(int bqm_variable) {
    if (_bqm_to_posiform_variable_map.count(bqm_variable) == 0) {
      return -1;
    } else {
      return _bqm_to_posiform_variable_map[bqm_variable];
    }
  }

  /**
   * Map a posiform variable to a QUBO variable. The number of QUBO variables 
   * and posiform variables may differ since the coefficient for a QUBO variable 
   * may turn out to be zero in a posiform. This may occur if the coefficient is 
   * flushed to zero during the conversion or if the variable did not have non-zero 
   * linear/quadratic biases in the QUBO.
   */
  inline int mapVariablePosiformToQubo(int posiform_variable) {
    return _posiform_to_bqm_variable_map[posiform_variable];
  }

  /**
   * Convert a QUBO coefficient to a posiform coefficient.
   */
  inline coefficient_type convertToPosiformCoefficient(bias_type bqm_bias) {
    return static_cast<coefficient_type>(bqm_bias * _bias_conversion_ratio);
  }

  /**
   * Return the value by which bqm biases are multiplied to get posiform coefficients.
   */
  inline double getBiasConversionRatio() {
	return _bias_conversion_ratio;
  }

  /**
   * Return the value of the constant term of the posiform.
   */
  inline double getConstant() {
	return _constant_posiform;
  }

  /**
   * Print out posiform details.
   */
  void print();

private:
  double _max_absolute_value;
  double _bias_conversion_ratio;
  coefficient_type _constant_posiform;
  double _posiform_linear_sum_non_integral;
  coefficient_type _posiform_linear_sum_integral;
  int _num_bqm_variables;
  int _num_posiform_variables;
  int _num_linear_integral_biases;
  std::vector<int> _num_quadratic_integral_biases;
  std::vector<int> _posiform_to_bqm_variable_map;
  std::unordered_map<int, int> _bqm_to_posiform_variable_map;
  std::vector<double> _linear_double_biases;
  std::vector<coefficient_type> _linear_integral_biases;
  std::vector<std::pair<quadratic_iterator_type, quadratic_iterator_type>>
      _quadratic_iterators;
};

template <class BQM, class coefficient_t>
PosiformInfo<BQM, coefficient_t>::PosiformInfo(const BQM &bqm) {
  assert(std::is_integral<coefficient_type>::value &&
         std::is_signed<coefficient_type>::value &&
         "Posiform must have signed, integral type coefficients");
  _constant_posiform = 0;
  _max_absolute_value = 0;
  _num_linear_integral_biases = 0;
  _num_bqm_variables = bqm.num_variables();
  _quadratic_iterators.resize(_num_bqm_variables);
  _linear_double_biases.resize(_num_bqm_variables, 0);
  _linear_integral_biases.resize(_num_bqm_variables, 0);
  _num_quadratic_integral_biases.resize(_num_bqm_variables, 0);

  // Apart from finding the maximum absolute value in the bqm, we must
  // consider the sum of the absolute values of linear values found by using
  // non integral bqm-biases for calculating the conversion ratio. As that
  // value converted to integral type is the upper bound for max flow. This is
  // because the linear values correspond to the capacities of the edges in
  // the implication network connecting to source/sink. Thus we calculate the
  // linear values of posiform in double format. Note in the posiform all
  // linear values are positive, here we keep the sign to indicate whether
  // X_source is connected to X_i or X_i'.
  for (int bqm_variable = 0; bqm_variable < _num_bqm_variables;
       bqm_variable++) {
    auto bqm_linear = bqm.linear(bqm_variable);
    auto bqm_linear_abs = std::fabs(bqm_linear);
    _linear_double_biases[bqm_variable] = bqm_linear;
    if (_max_absolute_value < bqm_linear_abs) {
      _max_absolute_value = bqm_linear_abs;
    }
    auto span =
            std::make_pair(std::lower_bound(bqm.cbegin_neighborhood(bqm_variable),
                                            bqm.cend_neighborhood(bqm_variable), bqm_variable + 1),
                           bqm.cend_neighborhood(bqm_variable));
    _quadratic_iterators[bqm_variable] = span;
    if (span.first != span.second) {
      for (auto it_end = span.second; span.first != it_end; span.first++) {
        auto bqm_quadratic = span.first->bias;
        auto bqm_quadratic_abs = std::fabs(bqm_quadratic);
        if (bqm_quadratic < 0) {
          _linear_double_biases[bqm_variable] += bqm_quadratic;
        }
        if (_max_absolute_value < bqm_quadratic_abs) {
          _max_absolute_value = bqm_quadratic_abs;
        }
      }
    }
  }

  // See comment above regarding calculating conversion ratio. We do not know
  // the conversion ratio yet so we consider all the linear values, including
  // those which will be flushed to zero and will not be in the posiform. Hence
  // we are summing up the linear coefficients for the posiform over all the bqm
  // variables, not the posiform variables.
  _posiform_linear_sum_non_integral = 0;
  for (std::size_t bqm_variable = 0; bqm_variable < _linear_double_biases.size();
       bqm_variable++) {
    _posiform_linear_sum_non_integral +=
        std::fabs(_linear_double_biases[bqm_variable]);
  }

  // Consider the upper limit of max-flow in implication graph for calculating
  // conversion ratio.
  if (_max_absolute_value < _posiform_linear_sum_non_integral) {
    _max_absolute_value = _posiform_linear_sum_non_integral;
  }

  if (_max_absolute_value != 0) {
     _bias_conversion_ratio =
      static_cast<double>(std::numeric_limits<coefficient_type>::max()) /
      _max_absolute_value;

    // We should divide the ratio by at least 2 to avoid overflow because we do
    // not divide by 2 when we make the residual network symmetric. In that step
    // the total flow is effectively doubled along with capacities. Divinding just
    // by 2 introduced overflow, thus we divide by a number larger than 2.
    // TODO : Find the theoretical optimal number for division, for now we divide
    // by 4 to be safe.
    _bias_conversion_ratio /= 4;

    for (int bqm_variable = 0; bqm_variable < _num_bqm_variables;
        bqm_variable++) {
      _linear_integral_biases[bqm_variable] =
          convertToPosiformCoefficient(bqm.linear(bqm_variable));
      int num_nonZero_quadratic_biases_in_upper_triangular_row = 0;
      auto it = _quadratic_iterators[bqm_variable].first;
      auto it_end = _quadratic_iterators[bqm_variable].second;
      for (; it != it_end; it++) {
        // Note the checks for zero below must be done after conversion since some
        // values may get flushed to zero, we want the linear values and quadratic
        // values adhere to the same conversion protocal and be consistent.
        // Otherwise in the implication graph, there may be paths with flow values
        // of small capacities corresponding to numerical errors, affecting the
        // final result. For this same reason we do not use the already computed
        // linear values for the posiform in double format.
        auto bias_quadratic_integral = convertToPosiformCoefficient(it->bias);
        if (bias_quadratic_integral) {
          _num_quadratic_integral_biases[it->v]++;
          if (bias_quadratic_integral < 0) {
            _linear_integral_biases[bqm_variable] += bias_quadratic_integral;
          }
          num_nonZero_quadratic_biases_in_upper_triangular_row++;
        }
      }
      _num_quadratic_integral_biases[bqm_variable] +=
          num_nonZero_quadratic_biases_in_upper_triangular_row;
    }

    _posiform_linear_sum_integral = 0; // For debugging purposes.
    for (std::size_t bqm_variable = 0; bqm_variable < _linear_integral_biases.size();
        bqm_variable++) {
      if (_linear_integral_biases[bqm_variable]) {
        _num_linear_integral_biases++;
        _posiform_linear_sum_integral +=
            std::llabs(_linear_integral_biases[bqm_variable]);
      }
      if (_linear_integral_biases[bqm_variable] < 0) {
        _constant_posiform += _linear_integral_biases[bqm_variable];
      }
    }

    _posiform_to_bqm_variable_map.reserve(_num_bqm_variables);
    _num_posiform_variables = 0;
    for (int bqm_variable = 0; bqm_variable < _num_bqm_variables;
        bqm_variable++) {
      if (_linear_integral_biases[bqm_variable] ||
          _num_quadratic_integral_biases[bqm_variable]) {
        _posiform_to_bqm_variable_map.push_back(bqm_variable);
        _bqm_to_posiform_variable_map.insert(
            {bqm_variable, _num_posiform_variables});
        _num_posiform_variables++;
      }
    }
  }
  else {
    // All biases in bqm are 0, so the resulting posiform won't have any term in it.
    _num_posiform_variables = 0;
    _constant_posiform = 0;
    _bias_conversion_ratio = 1;
    _posiform_linear_sum_integral = 0;
    _num_linear_integral_biases = 0;
  }
}

template <class BQM, class coefficient_t>
void PosiformInfo<BQM, coefficient_t>::print() {
  std::cout << std::endl;
  std::cout << "Posiform Information : " << std::endl << std::endl;
  std::cout << "Number of BQM Variables : " << _num_bqm_variables << std::endl;
  std::cout << "Constant : " << _constant_posiform << std::endl;
  std::cout << "Maximum Absolute Value : " << _max_absolute_value << std::endl;
  std::cout << "Numeric Limit of Coefficient Type "
            << std::numeric_limits<coefficient_type>::max() << std::endl;
  std::cout << "Linear Sum in double Format : "
            << _posiform_linear_sum_non_integral << std::endl;
  std::cout << "Linear Sum Converted to Integral Type : "
            << convertToPosiformCoefficient(_posiform_linear_sum_non_integral)
            << std::endl;
  std::cout << "Linear Sum After Summing in Integral Type : "
            << _posiform_linear_sum_integral << std::endl;
  std::cout << "Ratio Chosen : " << _bias_conversion_ratio << std::endl;
  std::cout << std::endl;

  // Printing out in convoluted manner, to verify the mappings.
  // Posiform variables should be serially numbered, 0,1,2....
  std::cout << "Used Variables (bqm --> posiform) : "
            << _posiform_to_bqm_variable_map.size() << std::endl;
  for (int i = 0; i < _posiform_to_bqm_variable_map.size(); i++) {
    std::cout << _posiform_to_bqm_variable_map[i] << " --> "
              << _bqm_to_posiform_variable_map[_posiform_to_bqm_variable_map[i]]
              << std::endl;
  }

  std::cout << std::endl;
  std::cout << "Linear (posiform, bqm, value) : " << std::endl;
  for (int bqm_variable = 0; bqm_variable < _num_bqm_variables;
       bqm_variable++) {
    if (_linear_integral_biases[bqm_variable]) {
      std::cout << _bqm_to_posiform_variable_map[bqm_variable] << ", "
                << bqm_variable << ", " << _linear_integral_biases[bqm_variable]
                << std::endl;
    }
  }

  std::cout << std::endl;
  std::cout << "Quadratic (posiform-posiform, bqm-bqm, value): " << std::endl;
  for (int bqm_variable = 0; bqm_variable < _num_bqm_variables;
       bqm_variable++) {
    auto it = _quadratic_iterators[bqm_variable].first;
    auto it_end = _quadratic_iterators[bqm_variable].second;
    for (; it != it_end; it++) {
      std::cout << _bqm_to_posiform_variable_map[bqm_variable] << " "
                << _bqm_to_posiform_variable_map[it->v] << ", ";
      std::cout << bqm_variable << " " << it->v << ",  "
                << convertToPosiformCoefficient(it->bias) << std::endl;
    }
  }
  std::cout << std::endl;
}

#endif // POSIFORM_INFORMATION_INCLUDED
