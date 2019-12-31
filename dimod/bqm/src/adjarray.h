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

#ifndef DIMOD_BQM_SRC_ADJARRAY_H_
#define DIMOD_BQM_SRC_ADJARRAY_H_

#include <utility>
#include <vector>

namespace dimod {

// developer note: In the future, we probably want to allow the user to
// also configure the type of the out var index rather than hardcoding it to
// size_t as we do here.
template<typename Bias>
using AdjArrayInVars = typename std::vector<std::pair<std::size_t, Bias>>;

template<typename VarIndex, typename Bias>
using AdjArrayOutVars = typename std::vector<std::pair<VarIndex, Bias>>;

template<typename VarIndex, typename Bias>
using AdjArrayBQM = typename std::pair<AdjArrayInVars<Bias>,
                                       AdjArrayOutVars<VarIndex, Bias>>;

// Read the BQM

template<typename V, typename B>
std::size_t num_variables(const AdjArrayBQM<V, B>&);

template<typename V, typename B>
std::size_t num_interactions(const AdjArrayBQM<V, B>&);

template<typename V, typename B>
B get_linear(const AdjArrayBQM<V, B>&, V);

template<typename V, typename B>
std::pair<B, bool> get_quadratic(const AdjArrayBQM<V, B>&, V, V);

template<typename V, typename B>
std::size_t degree(const AdjArrayBQM<V, B>&, V);

template<typename V, typename B>
std::pair<typename AdjArrayOutVars<V, B>::iterator,
            typename AdjArrayOutVars<V, B>::iterator>
neighborhood(AdjArrayBQM<V, B>&, V, bool upper_triangular = false);

// todo: variable_iterator
// todo: interaction_iterator
// todo: neighbour_iterator

// Change the values in the BQM

template<typename V, typename B>
void set_linear(AdjArrayBQM<V, B>&, V, B);

template<typename V, typename B>
bool set_quadratic(AdjArrayBQM<V, B>&, V, V, B);

template<typename V, typename B, class BQM>
void copy_bqm(BQM &bqm, AdjArrayBQM<V, B> &bqm_copy);

}  // namespace dimod

#endif  // DIMOD_BQM_SRC_ADJARRAY_H_
