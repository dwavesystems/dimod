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

#include "src/adjvector.h"

#include <algorithm>


namespace dimod {

// Read the BQM

template<typename VarIndex, typename Bias>
std::size_t num_variables(const std::vector<std::pair<std::vector<
                          std::pair<VarIndex, Bias>>, Bias>> &bqm) {
    return bqm.size();
}

template<typename VarIndex, typename Bias>
std::size_t num_interactions(const std::vector<std::pair<std::vector<
                             std::pair<VarIndex, Bias>>, Bias>> &bqm) {
    std::size_t count = 0;
    for (typename std::vector<std::pair<std::vector<std::pair<VarIndex,
         Bias>>, Bias>>::const_iterator it = bqm.begin();
         it != bqm.end(); ++it) {
        count += (*it).first.size();
    }
    return count / 2;
}

template<typename VarIndex, typename Bias>
Bias get_linear(const std::vector<std::pair<std::vector<std::pair<VarIndex,
                Bias>>, Bias>> &bqm, VarIndex v) {
    assert(v >= 0 && v < bqm.size());
    return bqm[v].second;
}

template<typename VarIndex, typename Bias>
std::pair<Bias, bool> get_quadratic(const std::vector<std::pair<std::vector<
                                    std::pair<VarIndex, Bias>>, Bias>> &bqm,
                                    VarIndex u, VarIndex v) {
    assert(u >= 0 && u < bqm.size());
    assert(v >= 0 && v < bqm.size());
    assert(u != v);

    const std::pair<VarIndex, Bias> target(v, 0);

    // binary search on the vector, matching only the first variable, we could
    // search the smaller of the two neighbourhoods
    typename std::vector<std::pair<VarIndex, Bias>>::const_iterator it;
    it = std::lower_bound(bqm[u].first.begin(), bqm[u].first.end(),
                          target, pair_lt<VarIndex, Bias>);

    if (it == bqm[u].first.end() || (*it).first != v)
        return std::make_pair(0, false);

    return std::make_pair((*it).second, true);
}

// Change the values in the BQM
template<typename VarIndex, typename Bias>
void set_linear(std::vector<std::pair<std::vector<std::pair<VarIndex, Bias>>,
                Bias>> &bqm,
                VarIndex v, Bias b) {
    assert(v >= 0 && v < bqm.size());
    bqm[v].second = b;
}

template<typename VarIndex, typename Bias>
bool set_quadratic(std::vector<std::pair<std::vector<std::pair<VarIndex, Bias>>,
                   Bias>> &bqm,
                   VarIndex u, VarIndex v, Bias b) {
    assert(u >= 0 && u < bqm.size());
    assert(v >= 0 && v < bqm.size());
    assert(u != v);

    typename std::vector<std::pair<VarIndex, Bias>>::iterator it;
    bool uv_exists, vu_exists;

    // u, v
    const std::pair<VarIndex, Bias> target_uv(v, b);
    it = std::lower_bound(bqm[u].first.begin(), bqm[u].first.end(),
                          target_uv, pair_lt<VarIndex, Bias>);
    if (it == bqm[u].first.end() || (*it).first != v) {
        bqm[u].first.insert(it, target_uv);
        uv_exists = false;
    } else {
        (*it).second = b;
        uv_exists = true;
    }

    // v, u
    const std::pair<VarIndex, Bias> target_vu(u, b);
    it = std::lower_bound(bqm[v].first.begin(), bqm[v].first.end(),
                          target_vu, pair_lt<VarIndex, Bias>);
    if (it == bqm[v].first.end() || (*it).first != u) {
        bqm[v].first.insert(it, target_vu);
        vu_exists = false;
    } else {
        (*it).second = b;
        vu_exists = true;
    }

    assert(uv_exists == vu_exists);  // sanity check

    return (uv_exists && vu_exists);
}

// Change the structure of the BQM

template<typename VarIndex, typename Bias>
VarIndex add_variable(std::vector<std::pair<std::vector<std::pair<VarIndex,
                      Bias>>, Bias>> &bqm) {
    bqm.resize(num_variables(bqm)+1);
    return num_variables(bqm)-1;
}

// todo: this should do nothing if the interaction is already present rather
// than overriding with 0
template<typename VarIndex, typename Bias>
bool add_interaction(std::vector<std::pair<std::vector<std::pair<VarIndex,
                     Bias>>, Bias>> &bqm,
                     VarIndex u, VarIndex v, Bias b) {
    return set_quadratic(bqm, u, v, 0);
}

template<typename VarIndex, typename Bias>
VarIndex pop_variable(std::vector<std::pair<std::vector<std::pair<VarIndex,
                      Bias>>, Bias>> &bqm) {
    assert(bqm.size() > 0);  // undefined for empty


    VarIndex v = bqm.size() - 1;

    // remove v from any of it's neighbour's associated vectors
    VarIndex u;
    typename std::pair<VarIndex, Bias> target(v, 0);
    typename std::vector<std::pair<VarIndex, Bias>>::iterator it;
    for (it = bqm[v].first.begin(); it != bqm[v].first.end(); it++) {
        u = (*it).first;
        bqm[u].first.erase(std::lower_bound(bqm[u].first.begin(),
                                            bqm[u].first.end(),
                                            target, pair_lt<VarIndex, Bias>));
    }

    bqm.pop_back();

    return v;
}

template<typename VarIndex, typename Bias>
bool remove_interaction(std::vector<std::pair<std::vector<std::pair<VarIndex,
                      Bias>>, Bias>> &bqm, VarIndex u, VarIndex v) {
    assert(u >= 0 && u < bqm.size());
    assert(v >= 0 && v < bqm.size());
    assert(u != v);

    typename std::vector<std::pair<VarIndex, Bias>>::iterator it;
    bool uv_exists, vu_exists;

    // u, v
    const std::pair<VarIndex, Bias> target_uv(v, 0);
    it = std::lower_bound(bqm[u].first.begin(), bqm[u].first.end(),
                          target_uv, pair_lt<VarIndex, Bias>);
    if (it == bqm[u].first.end() || (*it).first != v) {
        uv_exists = false;
    } else {
        bqm[u].first.erase(it);
        uv_exists = true;
    }

    // v, u
    const std::pair<VarIndex, Bias> target_vu(u, 0);
    it = std::lower_bound(bqm[v].first.begin(), bqm[v].first.end(),
                          target_vu, pair_lt<VarIndex, Bias>);
    if (it == bqm[v].first.end() || (*it).first != u) {
        vu_exists = false;
    } else {
        bqm[v].first.erase(it);
        vu_exists = true;
    }

    assert(uv_exists == vu_exists);  // sanity check

    return (uv_exists && vu_exists);
}
}  // namespace dimod
