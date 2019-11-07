// Copyright 2019 D-Wave Systems Inc.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#include <utility>
#include <vector>

#ifndef DIMOD_BQM_SRC_ADJVECTOR_H_
#define DIMOD_BQM_SRC_ADJVECTOR_H_

namespace dimod {

template<typename VarIndex, typename Bias>
using VectorNeighbourhood = typename std::vector<std::pair<VarIndex, Bias>>;

template<typename VarIndex, typename Bias>
using AdjVectorBQM = typename std::vector<
    std::pair<VectorNeighbourhood<VarIndex, Bias>, Bias>>;

// Read the BQM

template<typename V, typename B>
std::size_t num_variables(const AdjVectorBQM<V, B>&);

template<typename V, typename B>
std::size_t num_interactions(const AdjVectorBQM<V, B>&);

template<typename V, typename B>
B get_linear(const AdjVectorBQM<V, B>&, V);

template<typename V, typename B>
std::pair<B, bool> get_quadratic(const AdjVectorBQM<V, B>&, V, V);

template<typename V, typename B>
std::size_t degree(const AdjVectorBQM<V, B>&, V);

template<typename V, typename B>
std::pair<typename VectorNeighbourhood<V, B>::const_iterator,
            typename VectorNeighbourhood<V, B>::const_iterator>
neighborhood(const AdjVectorBQM<V, B>&, V);

// todo: variable_iterator
// todo: interaction_iterator

// Change the values in the BQM

template<typename V, typename B>
void set_linear(AdjVectorBQM<V, B>&, V, B);

template<typename V, typename B>
bool set_quadratic(AdjVectorBQM<V, B>&, V, V, B);

// Change the structure of the BQM

template<typename V, typename B>
V add_variable(AdjVectorBQM<V, B>&);

template<typename V, typename B>
bool add_interaction(AdjVectorBQM<V, B>&, V, V);

template<typename V, typename B>
V pop_variable(AdjVectorBQM<V, B>&);

template<typename V, typename B>
bool remove_interaction(AdjVectorBQM<V, B>&, V, V);
}  // namespace dimod

#endif  // DIMOD_BQM_SRC_ADJVECTOR_H_
