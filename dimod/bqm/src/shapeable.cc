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
#include "src/shapeable.h"

#include <algorithm>
#include <tuple>


namespace dimod {

// Utilities for other functions

template<class VarIndex, class Bias>
bool cmp_v(const std::pair<VarIndex, Bias> ub, VarIndex v) {
    return ub.first < v;
}

// Returns at iterator pointing to the first pair in the neighborhood in which
// the variable is not less than `v`.
//
// Assumes that the neighborhood is sorted.
template<class VarIndex, class Bias>
inline typename VectorNeighborhood<VarIndex, Bias>::const_iterator
find_outvar(const VectorNeighborhood<VarIndex, Bias> &neighborhood, VarIndex v){
    return std::lower_bound(neighborhood.begin(), neighborhood.end(), v,
                            cmp_v<VarIndex, Bias>);
}
template<class VarIndex, class Bias>
inline typename VectorNeighborhood<VarIndex, Bias>::iterator
find_outvar(VectorNeighborhood<VarIndex, Bias> &neighborhood, VarIndex v){
    return std::lower_bound(neighborhood.begin(), neighborhood.end(), v,
                            cmp_v<VarIndex, Bias>);
}

// Returns at iterator pointing to the pair containing `v`, otherwise it will
// return an iterator to `neighborhood::end`.
template<class VarIndex, class Bias>
inline typename MapNeighborhood<VarIndex, Bias>::const_iterator
find_outvar(const MapNeighborhood<VarIndex, Bias> &neighborhood, VarIndex v){
    return neighborhood.find(v);
}
template<class VarIndex, class Bias>
inline typename MapNeighborhood<VarIndex, Bias>::iterator
find_outvar(MapNeighborhood<VarIndex, Bias> &neighborhood, VarIndex v){
    return neighborhood.find(v);
}


template<typename VarIndex, typename Bias>
std::pair<typename std::vector<std::pair<VarIndex, Bias>>::const_iterator, bool>
    directed_edge_iterator(const AdjVectorBQM<VarIndex, Bias> &bqm,
                           VarIndex u, VarIndex v) {
    assert(u >= 0 && u < bqm.size());
    assert(v >= 0 && v < bqm.size());
    assert(u != v);

    const std::pair<VarIndex, Bias> target(v, 0);

    typename std::vector<std::pair<VarIndex, Bias>>::const_iterator it;
    it = std::lower_bound(bqm[u].first.begin(), bqm[u].first.end(),
                          target, pair_lt<VarIndex, Bias>);

    return std::make_pair(it, !(it == bqm[u].first.end() || it->first != v));
}

template<typename VarIndex, typename Bias>
std::pair<typename std::vector<std::pair<VarIndex, Bias>>::iterator, bool>
    directed_edge_iterator(AdjVectorBQM<VarIndex, Bias> &bqm,
                           VarIndex u, VarIndex v) {
    assert(u >= 0 && u < bqm.size());
    assert(v >= 0 && v < bqm.size());
    assert(u != v);

    std::pair<VarIndex, Bias> target(v, 0);

    typename std::vector<std::pair<VarIndex, Bias>>::iterator it;
    it = std::lower_bound(bqm[u].first.begin(), bqm[u].first.end(),
                          target, pair_lt<VarIndex, Bias>);

    return std::make_pair(it, !(it == bqm[u].first.end() || (*it).first != v));
}


// Construction


// copy `bqm` into `bqm_copy`.
template<typename VarIndex, typename Bias, class BQM>
void copy_bqm(BQM &bqm, AdjVectorBQM<VarIndex, Bias> &bqm_copy) {

    bqm_copy.resize(num_variables(bqm));

    for (VarIndex v = 0; v < num_variables(bqm); ++v) {
        set_linear(bqm_copy, v, get_linear(bqm, v));

        // in case there is anything already in there
        bqm_copy[v].first.clear();

        // we know how much space we'll need
        bqm_copy[v].first.reserve(degree(bqm, v));

        auto span = neighborhood(bqm, v);
        bqm_copy[v].first.insert(bqm_copy[v].first.begin(), span.first, span.second);
    }
}

template<typename VarIndex, typename Bias, class BQM>
void copy_bqm(BQM &bqm, AdjMapBQM<VarIndex, Bias> &bqm_copy) {

    bqm_copy.resize(num_variables(bqm));

    for (VarIndex v = 0; v < num_variables(bqm); v++) {
        set_linear(bqm_copy, v, get_linear(bqm, v));

        // in case there is anything already in there
        bqm_copy[v].first.clear();

        auto span = neighborhood(bqm, v);
        bqm_copy[v].first.insert(span.first, span.second);
    }
}


// Read the BQM


template<class Neighborhood, class B>
std::size_t num_variables(const ShapeableBQM<Neighborhood, B> &bqm) {
    return bqm.size();
}


template<class Neighborhood, class B>
std::size_t num_interactions(const ShapeableBQM<Neighborhood, B> &bqm) {
    std::size_t count = 0;
    for (auto it = bqm.begin(); it != bqm.end(); ++it) {
        count += (*it).first.size();
    }
    return count / 2;
}


template<class Neighborhood, class VarIndex, class Bias>
Bias get_linear(const ShapeableBQM<Neighborhood, Bias> &bqm, VarIndex v) {
    assert(v >= 0 && v < bqm.size());
    return bqm[v].second;
}


template<class Neighborhood, class VarIndex, class Bias>
std::pair<Bias, bool> get_quadratic(const ShapeableBQM<Neighborhood, Bias> &bqm,
                                    VarIndex u, VarIndex v) {
    assert(u >= 0 && u < bqm.size());
    assert(v >= 0 && v < bqm.size());
    assert(u != v);

    // todo: we could search the shorter of the two neighborhoods

    auto it = find_outvar(bqm[u].first, v);

    if (it == bqm[u].first.end() || (*it).first != v)
        return std::make_pair(0, false);

    return std::make_pair(it->second, true);
}


template<class Neighborhood, class VarIndex, class Bias>
std::size_t degree(const ShapeableBQM<Neighborhood, Bias> &bqm, VarIndex v) {
    return bqm[v].first.size();
}


// see note in header about why we need these two different functions

template<class VarIndex, class Bias>
std::pair<typename VectorNeighborhood<VarIndex, Bias>::iterator,
          typename VectorNeighborhood<VarIndex, Bias>::iterator>
neighborhood(AdjVectorBQM<VarIndex, Bias> &bqm, VarIndex v) {
    assert(v >= 0 && v < bqm.size());
    return std::make_pair(bqm[v].first.begin(), bqm[v].first.end());
}

template<typename VarIndex, typename Bias>
std::pair<typename MapNeighborhood<VarIndex, Bias>::iterator,
          typename MapNeighborhood<VarIndex, Bias>::iterator>
neighborhood(AdjMapBQM<VarIndex, Bias> &bqm, VarIndex v) {
    assert(v >= 0 && v < bqm.size());
    return std::make_pair(bqm[v].first.begin(), bqm[v].first.end());
}


template<class Neighborhood, class VarIndex, class Bias>
void set_linear(ShapeableBQM<Neighborhood, Bias> &bqm, VarIndex v, Bias b) {
    assert(v >= 0 && v < bqm.size());
    bqm[v].second = b;
}


template<typename VarIndex, typename Bias>
void set_quadratic(AdjMapBQM<VarIndex, Bias> &bqm,
                    VarIndex u, VarIndex v, Bias b) {
    assert(u >= 0 && u < bqm.size());
    assert(v >= 0 && v < bqm.size());
    assert(u != v);

    bqm[u].first[v] = b;
    bqm[v].first[u] = b;
}

template<typename VarIndex, typename Bias>
void set_quadratic(AdjVectorBQM<VarIndex, Bias> &bqm,
                   VarIndex u, VarIndex v, Bias b) {
    assert(u >= 0 && u < bqm.size());
    assert(v >= 0 && v < bqm.size());
    assert(u != v);

    auto vit = find_outvar(bqm[u].first, v);
    bool uv_exists = !(vit == bqm[u].first.end() || vit->first != v);
    if (uv_exists) {
        vit->second = b;
    } else {
        bqm[u].first.insert(vit, std::make_pair(v, b));
    }

    auto uit = find_outvar(bqm[v].first, u);
    bool vu_exists = !(uit == bqm[v].first.end() || (*uit).first != u);
    if (vu_exists) {
        uit->second = b;
    } else {
        bqm[v].first.insert(uit, std::make_pair(u, b));
    }

    assert(uv_exists == vu_exists);  // sanity check
}


// Change the structure of the BQM


template<class Neighborhood, class Bias>
std::size_t add_variable(ShapeableBQM<Neighborhood, Bias> &bqm) {
    bqm.resize(bqm.size()+1);
    return bqm.size()-1;
}


template<class Neighborhood, class Bias>
std::size_t pop_variable(ShapeableBQM<Neighborhood, Bias> &bqm) {
    assert(bqm.size() > 0);  // undefined for empty

    typename Neighborhood::value_type::first_type v = bqm.size() - 1;

    // remove v from any of it's neighbor's associated vectors. We could use
    // remove_interaction but that acts bi-directionally which is not needed
    for (auto it = bqm[v].first.begin(); it != bqm[v].first.end(); it++) {
        auto u = (*it).first;
        bqm[u].first.erase(find_outvar(bqm[u].first, v));
    }

    bqm.pop_back();

    return bqm.size();
}


template<typename VarIndex, typename Bias>
bool remove_interaction(AdjMapBQM<VarIndex, Bias> &bqm,
                        VarIndex u, VarIndex v) {
    assert(u >= 0 && u < bqm.size());
    assert(v >= 0 && v < bqm.size());
    assert(u != v);

    bool uv_removed = bqm[u].first.erase(v);
    bool vu_removed = bqm[v].first.erase(u);

    assert(uv_removed == vu_removed);  // sanity check

    return uv_removed || vu_removed;
}

template<typename VarIndex, typename Bias>
bool remove_interaction(AdjVectorBQM<VarIndex, Bias> &bqm, VarIndex u, VarIndex v) {
    assert(u >= 0 && u < bqm.size());
    assert(v >= 0 && v < bqm.size());
    assert(u != v);

    auto vit = find_outvar(bqm[u].first, v);
    bool uv_exists = !(vit == bqm[u].first.end() || vit->first != v);
    if (uv_exists) 
        bqm[u].first.erase(vit);

    auto uit = find_outvar(bqm[v].first, u);
    bool vu_exists = !(uit == bqm[v].first.end() || uit->first != u);
    if (vu_exists) 
        bqm[v].first.erase(uit);

    assert(uv_exists == vu_exists);  // sanity check

    return (uv_exists && vu_exists);
}
}  // namespace dimod
