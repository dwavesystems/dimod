// Copyright 2020 D-Wave Systems Inc.
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

#ifndef DIMOD_ADJMAPBQM_H_
#define DIMOD_ADJMAPBQM_H_

#include <map>
#include <utility>
#include <vector>

#include "dimod/utils.h"

namespace dimod {

template<class V, class B>
class AdjMapBQM {
 public:
    using bias_type = B;
    using variable_type = V;
    using size_type = std::size_t;

    using outvars_iterator = typename std::map<V, B>::iterator;
    using const_outvars_iterator = typename std::map<V, B>::const_iterator;

    // in the future we'd probably like to make this protected
    std::vector<std::pair<std::map<V, B>, B>> adj;

    AdjMapBQM() {}

    template<class BQM>
    explicit AdjMapBQM(const BQM &bqm) {
        adj.resize(bqm.num_variables());

        for (variable_type v = 0; v < bqm.num_variables(); ++v) {
            linear(v) = bqm.linear(v);

            auto span = bqm.neighborhood(v);
            adj[v].first.insert(span.first, span.second);
        }
    }

    /**
     * Construct a BQM from a dense array.
     *
     * @param dense An array containing the biases. Assumed to contain
     *     `num_variables`^2 elements. The upper and lower triangle are summed.
     * @param num_variables The number of variables. 
     */
    template<class B2>
    AdjMapBQM(const B2 dense[], size_type num_variables,
              bool ignore_diagonal = false) {
        // we know how big our linear is going to be
        adj.resize(num_variables);

        bias_type qbias;

        if (!ignore_diagonal) {
            for (size_type v = 0; v < num_variables; ++v) {
                adj[v].second = dense[v*(num_variables+1)];
            }
        }

        for (size_type u = 0; u < num_variables; ++u) {
            for (size_type v = u + 1; v < num_variables; ++v) {
                qbias = dense[u*num_variables+v] + dense[v*num_variables+u];

                if (qbias != 0) {
                    adj[u].first.emplace_hint(adj[u].first.end(), v, qbias);
                    adj[v].first.emplace_hint(adj[v].first.end(), u, qbias);
                }
            }
        }
    }

    /// Add one (disconnected) variable to the BQM and return its index.
    variable_type add_variable() {
        adj.resize(adj.size()+1);
        return adj.size()-1;
    }

    /// Get the degree of variable `v`.
    size_type degree(variable_type v) const {
        return adj[v].first.size();
    }

    [[deprecated("Use AdjMapBQM::linear(v)")]]
    bias_type get_linear(variable_type v) const {
        return linear(v);
    }

    std::pair<bias_type, bool>
    get_quadratic(variable_type u, variable_type v) const {
        assert(u >= 0 && u < adj.size());
        assert(v >= 0 && v < adj.size());
        assert(u != v);

        auto it = adj[u].first.find(v);

        if (it == adj[u].first.end() || it->first != v)
            return std::make_pair(0, false);

        return std::make_pair(it->second, true);
    }

    bias_type& linear(variable_type v) {
        assert(v >= 0 && v < adj.size());
        return adj[v].second;
    }

    const bias_type& linear(variable_type v) const {
        assert(v >= 0 && v < adj.size());
        return adj[v].second;
    }

    std::pair<outvars_iterator, outvars_iterator>
    neighborhood(variable_type u) {
        assert(u >= 0 && u < adj.size());
        return std::make_pair(adj[u].first.begin(), adj[u].first.end());
    }

    std::pair<const_outvars_iterator, const_outvars_iterator>
    neighborhood(variable_type u) const {
        assert(u >= 0 && u < adj.size());
        return std::make_pair(adj[u].first.cbegin(), adj[u].first.cend());
    }

    /**
     * The neighborhood of variable `v`.
     * 
     * @param A variable `v`.
     * @param The neighborhood will start with the first out variable that
     * does not compare less than `start`.
     *
     * @returns A pair of iterators pointing to the start and end of the
     *     neighborhood.
     */
    std::pair<const_outvars_iterator, const_outvars_iterator>
    neighborhood(variable_type v, variable_type start) const {
        return std::make_pair(adj[v].first.lower_bound(start),
                              adj[v].first.cend());
    }

    size_type num_variables() const {
        return adj.size();
    }

    size_type num_interactions() const {
        size_type count = 0;
        for (auto it = adj.begin(); it != adj.end(); ++it)
            count += it->first.size();
        return count / 2;
    }

    variable_type pop_variable() {
        assert(adj.size() > 0);

        variable_type v = adj.size() - 1;

        // remove v from all of its neighbor's neighborhoods
        for (auto it = adj[v].first.cbegin(); it != adj[v].first.cend(); ++it)
            adj[it->first].first.erase(v);

        adj.pop_back();

        return adj.size();
    }

    bool remove_interaction(variable_type u, variable_type v) {
        assert(u >= 0 && u < adj.size());
        assert(v >= 0 && v < adj.size());

        if (adj[u].first.erase(v) > 0) {
            adj[v].first.erase(u);
            return true;
        }

        return false;
    }

    [[deprecated("Use AdjMapBQM::linear(v)")]]
    void set_linear(variable_type v, bias_type b) {
        assert(v >= 0 && v < adj.size());
        linear(v) = b;
    }

    bool set_quadratic(variable_type u, variable_type v, bias_type b) {
        assert(u >= 0 && u < adj.size());
        assert(v >= 0 && v < adj.size());
        assert(u != v);

        adj[u].first[v] = b;
        adj[v].first[u] = b;

        // to be consistent with AdjArrayBQM, we return whether the value was
        // set
        return true;
    }
};

}  // namespace dimod

#endif  // DIMOD_ADJMAPBQM_H_
