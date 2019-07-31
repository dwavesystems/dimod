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

#include <map>
#include <utility>
#include <vector>

#ifndef ADJMAP_H
#define ADJMAP_H

namespace dimod {

    template<typename VarIndex, typename Bias>
    using Neighbourhood = typename std::map<VarIndex, Bias>;

    template<typename VarIndex, typename Bias>
    using AdjMapBQM = typename std::vector<std::pair<Neighbourhood<VarIndex, Bias>, Bias>>;

    // Read the BQM

    template<typename V, typename B>
    size_t num_variables(AdjMapBQM<V, B>&);

    template<typename V, typename B>
    size_t num_interactions(AdjMapBQM<V, B>&);

    template<typename V, typename B>
    B get_linear(AdjMapBQM<V, B>&, V);

    template<typename V, typename B>
    B get_quadratic(AdjMapBQM<V, B>&, V, V);

    // Change the values in the BQM

    template<typename V, typename B>
    void set_linear(AdjMapBQM<V, B>&, V, B);

    template<typename V, typename B>
    void set_quadratic(AdjMapBQM<V, B>&, V, V, B);

    // Change the structure of the BQM

    template<typename V, typename B>
    V add_variable(AdjMapBQM<V, B>&);

    template<typename V, typename B>
    bool add_interaction(AdjMapBQM<V, B>&, V, V);

    template<typename V, typename B>
    V pop_variable(AdjMapBQM<V, B>&);


}

#endif  // ADJMAP_H
