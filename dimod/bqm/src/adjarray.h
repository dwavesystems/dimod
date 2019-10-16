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

#ifndef DIMOD_BQM_SRC_ADJARRAY_H_
#define DIMOD_BQM_SRC_ADJARRAY_H_

namespace dimod {

    template<typename Bias>
    using AdjArrayInVars = typename std::vector<std::pair<std::size_t, Bias>>;

    template<typename VarIndex, typename Bias>
    using AdjArrayOutVars = typename std::vector<std::pair<VarIndex, Bias>>;

    // Read the BQM

    template<typename V, typename B>
    std::size_t num_variables(const AdjArrayInVars<B>&,
                              const AdjArrayOutVars<V, B>&);

    template<typename V, typename B>
    std::size_t num_interactions(const AdjArrayInVars<B>&,
                                 const AdjArrayOutVars<V, B>&);

    template<typename V, typename B>
    B get_linear(const AdjArrayInVars<B>&, const AdjArrayOutVars<V, B>&, V);

    template<typename V, typename B>
    std::pair<B, bool> get_quadratic(const AdjArrayInVars<B>&,
                                     const AdjArrayOutVars<V, B>&,
                                     V, V);

    template<typename V, typename B>
    std::pair<typename AdjArrayOutVars<V, B>::const_iterator,
              typename AdjArrayOutVars<V, B>::const_iterator>
    neighborhood(const AdjArrayInVars<B>&, const AdjArrayOutVars<V, B>&, V);

    // todo: variable_iterator
    // todo: interaction_iterator
    // todo: neighbour_iterator

    // Change the values in the BQM

    template<typename V, typename B>
    void set_linear(AdjArrayInVars<B>&, AdjArrayOutVars<V, B>&, V, B);

    template<typename V, typename B>
    bool set_quadratic(AdjArrayInVars<B>&, AdjArrayOutVars<V, B>&, V, V, B);

}  // namespace dimod

#endif  // DIMOD_BQM_SRC_ADJARRAY_H_
