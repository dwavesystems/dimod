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

#include "src/adjarray.h"

#include <algorithm>
#include <utility>

namespace dimod {

// Read the BQM

template<typename VarIndex, typename Bias>
std::size_t num_variables(const AdjArrayBQM<VarIndex, Bias> &bqm) {
    return bqm.first.size();
}

template<typename VarIndex, typename Bias>
std::size_t num_interactions(const AdjArrayBQM<VarIndex, Bias> &bqm) {
    return bqm.second.size() / 2;
}

template<typename VarIndex, typename Bias>
Bias get_linear(const AdjArrayBQM<VarIndex, Bias> &bqm,
                VarIndex v) {
    assert(v >= 0 && v < bqm.first.size());
    return bqm.first[v].second;
}

// compare only the first value in the pair
template<typename VarIndex, typename Bias>
bool pair_lt(const std::pair<VarIndex, Bias> a,
                const std::pair<VarIndex, Bias> b) {
    return a.first < b.first;
}

template<typename VarIndex, typename Bias>
std::pair<Bias, bool> get_quadratic(const AdjArrayBQM<VarIndex, Bias> &bqm,
                                    VarIndex u, VarIndex v) {
    assert(u >= 0 && u < bqm.first.size());
    assert(v >= 0 && v < bqm.first.size());
    assert(u != v);

    if (v < u) std::swap(u, v);

    std::size_t start = bqm.first[u].first;
    std::size_t end = bqm.first[u+1].first;  // safe if u < v and asserts above

    const std::pair<VarIndex, Bias> target(v, 0);

    typename AdjArrayOutVars<VarIndex, Bias>::const_iterator low;
    low = std::lower_bound(bqm.second.begin()+start, bqm.second.begin()+end,
                            target, pair_lt<VarIndex, Bias>);

    if (low == bqm.second.begin()+end)
        return std::make_pair(0, false);
    return std::make_pair((*low).second, true);
}

// Iterate over the neighbourhood of variable u.
//
// Returns iterators to beginning and end of neighborhood.
// upper_triangular will gives the start iterator at the first index > u
template<typename VarIndex, typename Bias>
std::pair<typename AdjArrayOutVars<VarIndex, Bias>::const_iterator,
          typename AdjArrayOutVars<VarIndex, Bias>::const_iterator>
neighborhood(const AdjArrayBQM<VarIndex, Bias> &bqm, VarIndex u,
             bool upper_triangular) {
    typename AdjArrayOutVars<VarIndex, Bias>::const_iterator start, stop;

    if (u == bqm.first.size() - 1) {
        // last element so ends at the end out bqm.second
        stop = bqm.second.end();
    } else {
        stop = bqm.second.begin() + bqm.first[u+1].first;
    }

    start = bqm.second.begin() + bqm.first[u].first;
    if (upper_triangular) {
        // binary search to find the index of the starting position
        const std::pair<VarIndex, Bias> target(u, 0);
        start = std::lower_bound(start, stop, target, pair_lt<VarIndex, Bias>);
    }

    // random access iterators
    return std::make_pair(start, stop);
}

// Change the values in the BQM

template<typename VarIndex, typename Bias>
void set_linear(AdjArrayBQM<VarIndex, Bias> &bqm,
                VarIndex v, Bias b) {
    assert(v >= 0 && v < bqm.first.size());
    bqm.first[v].second = b;
}

// Q: Should we do something else if the user tries to set a non-existant
//    quadratic bias? Error/segfault?
template<typename VarIndex, typename Bias>
bool set_quadratic(AdjArrayBQM<VarIndex, Bias> &bqm,
                   VarIndex u, VarIndex v, Bias b) {
    assert(u >= 0 && u < bqm.first.size());
    assert(v >= 0 && v < bqm.first.size());
    assert(u != v);

    typename AdjArrayOutVars<VarIndex, Bias>::iterator low;
    std::size_t start, end;

    // interaction (u, v)
    start = bqm.first[u].first;
    if (u < bqm.first.size() - 1) {
        end = bqm.first[u+1].first;
    } else {
        end = bqm.second.size();
    }

    const std::pair<VarIndex, Bias> v_target(v, 0);

    low = std::lower_bound(bqm.second.begin()+start, bqm.second.begin()+end,
                            v_target, pair_lt<VarIndex, Bias>);

    // if the edge does not exist, just return false here
    if (low == bqm.second.begin()+end)
        return false;

    (*low).second = b;

    // interaction (v, u)
    start = bqm.first[v].first;
    if (v < bqm.first.size() - 1) {
        end = bqm.first[v+1].first;
    } else {
        end = bqm.second.size();
    }

    const std::pair<VarIndex, Bias> u_target(u, 0);

    low = std::lower_bound(bqm.second.begin()+start, bqm.second.begin()+end,
                            u_target, pair_lt<VarIndex, Bias>);

    // we already know that this exists
    (*low).second = b;

    return true;
}

}  // namespace dimod
