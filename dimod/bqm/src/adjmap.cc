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

#include "src/adjmap.h"

#include <utility>


namespace dimod {

// Read the BQM

template<typename VarIndex, typename Bias>
std::size_t num_variables(const AdjMapBQM<VarIndex, Bias> &bqm) {
    return bqm.size();
}

template<typename VarIndex, typename Bias>
std::size_t num_interactions(const AdjMapBQM<VarIndex, Bias> &bqm) {
    std::size_t count = 0;
    for (typename AdjMapBQM<VarIndex, Bias>::const_iterator it = bqm.begin();
            it != bqm.end(); ++it) {
        count += (*it).first.size();
    }
    return count / 2;
}

template<typename VarIndex, typename Bias>
Bias get_linear(const AdjMapBQM<VarIndex, Bias> &bqm, VarIndex v) {
    assert(v >= 0 && v < bqm.size());
    return bqm[v].second;
}

template<typename VarIndex, typename Bias>
std::pair<Bias, bool> get_quadratic(const AdjMapBQM<VarIndex, Bias> &bqm,
                                    VarIndex u, VarIndex v) {
    assert(u >= 0 && u < bqm.size());
    assert(v >= 0 && v < bqm.size());
    assert(u != v);

    // we could potentially search the shorter of the two neighbourhoods
    typename std::map<VarIndex, Bias>::const_iterator it;
    it = bqm[u].first.find(v);

    if (it == bqm[u].first.end())
        return std::make_pair(0, false);

    return std::make_pair((*it).second, true);
}

template<typename VarIndex, typename Bias>
std::pair<typename MapNeighbourhood<VarIndex, Bias>::iterator,
          typename MapNeighbourhood<VarIndex, Bias>::iterator>
neighborhood(AdjMapBQM<VarIndex, Bias> &bqm, VarIndex v,
             bool upper_triangular) {
    typename MapNeighbourhood<VarIndex, Bias>::iterator start;

    if (upper_triangular) {
        // std::pair<VarIndex, Bias> target(v, 0);
        start = bqm[v].first.lower_bound(v);
    } else {
        start = bqm[v].first.begin();
    }
    return std::make_pair(start, bqm[v].first.end());
}

template<typename VarIndex, typename Bias>
std::size_t degree(const AdjMapBQM<VarIndex, Bias> &bqm, VarIndex v) {
    return bqm[v].first.size();
}

// todo: variable_iterator
// todo: interaction_iterator

// Change the values in the BQM

template<typename VarIndex, typename Bias>
void set_linear(AdjMapBQM<VarIndex, Bias> &bqm, VarIndex v, Bias b) {
    assert(v >= 0 && v < bqm.size());
    bqm[v].second = b;
}

// todo: decide and document what happens when u == v
template<typename VarIndex, typename Bias>
void set_quadratic(AdjMapBQM<VarIndex, Bias> &bqm,
                    VarIndex u, VarIndex v, Bias b) {
    assert(u >= 0 && u < bqm.size());
    assert(v >= 0 && v < bqm.size());
    assert(u != v);

    bqm[u].first[v] = b;
    bqm[v].first[u] = b;
}

// Change the structure of the BQM

template<typename VarIndex, typename Bias>
VarIndex add_variable(AdjMapBQM<VarIndex, Bias> &bqm) {
    bqm.push_back(std::make_pair(std::map<VarIndex, Bias>(), 0));
    return bqm.size() - 1;
}

template<typename VarIndex, typename Bias>
bool add_interaction(AdjMapBQM<VarIndex, Bias> &bqm,
                        VarIndex u, VarIndex v) {
    assert(u >= 0 && u < bqm.size());
    assert(v >= 0 && v < bqm.size());
    assert(u != v);

    std::pair<typename std::map<VarIndex, Bias>::iterator, bool> ret_u
        = bqm[u].first.insert(std::make_pair(v, 0));
    std::pair<typename std::map<VarIndex, Bias>::iterator, bool> ret_v
        = bqm[v].first.insert(std::make_pair(u, 0));

    // sanity check that they both were present/not
    assert(ret_u.second == ret_v.second);

    return ret_u.second;
}

template<typename VarIndex, typename Bias>
VarIndex pop_variable(AdjMapBQM<VarIndex, Bias> &bqm) {
    assert(bqm.size() > 0);  // undefined for empty

    VarIndex v = bqm.size() - 1;

    for (typename std::map<VarIndex, Bias>::iterator
            it=bqm[v].first.begin(); it != (*bqm.rbegin()).first.end(); it++) {
        bqm[(*it).first].first.erase(v);
    }

    bqm.pop_back();

    return v;
}

// todo: decide and document what happens when u == v
template<typename VarIndex, typename Bias>
bool remove_interaction(AdjMapBQM<VarIndex, Bias> &bqm,
                        VarIndex u, VarIndex v) {
    assert(u >= 0 && u < bqm.size());
    assert(v >= 0 && v < bqm.size());
    assert(u != v);

    bool uv_removed = bqm[u].first.erase(v);
    bool vu_removed = bqm[v].first.erase(u);

    assert(uv_removed == vu_removed);  // should always match

    return uv_removed || vu_removed;
}
}  // namespace dimod
