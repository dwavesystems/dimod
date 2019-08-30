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


    // developer note: while we continue to support python2.7 we are stuck
    // using visual studio 9.0 and consequently don't have access to template
    // aliases. For now we'll just be more verbose but here they are for
    // reference:
    //
    // template<typename VarIndex, typename Bias>
    // using Neighbourhood = typename std::vector<std::pair<VarIndex, Bias>>;
    //
    // template<typename VarIndex, typename Bias>
    // using AdjVectorBQM = typename std::vector<
    //     std::pair<Neighbourhood<VarIndex, Bias>, Bias>>;

    // Read the BQM

    template<typename V, typename B>
    std::size_t num_variables(
        const std::vector<std::pair<std::vector<std::pair<V, B>>, B>>&);

    template<typename V, typename B>
    std::size_t num_interactions(
        const std::vector<std::pair<std::vector<std::pair<V, B>>, B>>&);

    template<typename V, typename B>
    B get_linear(
        const std::vector<std::pair<std::vector<std::pair<V, B>>, B>>&, V);

    template<typename V, typename B>
    std::pair<B, bool> get_quadratic(
        const std::vector<std::pair<std::vector<std::pair<V, B>>, B>>&, V, V);

    // todo: variable_iterator
    // todo: interaction_iterator
    // todo: neighbour_iterator

    // Change the values in the BQM

    template<typename V, typename B>
    void set_linear(
        std::vector<std::pair<std::vector<std::pair<V, B>>, B>>&, V, B);

    template<typename V, typename B>
    bool set_quadratic(
        std::vector<std::pair<std::vector<std::pair<V, B>>, B>>&, V, V, B);

    // Change the structure of the BQM

    template<typename V, typename B>
    V add_variable(std::vector<std::pair<std::vector<std::pair<V, B>>, B>>&);

    template<typename V, typename B>
    bool add_interaction(
        std::vector<std::pair<std::vector<std::pair<V, B>>, B>>&, V, V);

    template<typename V, typename B>
    V pop_variable(std::vector<std::pair<std::vector<std::pair<V, B>>, B>>&);

    template<typename V, typename B>
    bool remove_interaction(
        std::vector<std::pair<std::vector<std::pair<V, B>>, B>>&, V, V);
}  // namespace dimod

#endif  // DIMOD_BQM_SRC_ADJVECTOR_H_
