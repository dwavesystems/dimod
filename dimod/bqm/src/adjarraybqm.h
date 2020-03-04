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

#ifndef DIMOD_BQM_SRC_ADJARRAYBQM_H_
#define DIMOD_BQM_SRC_ADJARRAYBQM_H_

#include <algorithm>
#include <utility>
#include <vector>

namespace dimod {

template<class V, class B, class N = std::size_t>
class AdjArrayBQM {
public:
    typedef B bias_type;
    typedef N neighborhood_type;
    typedef V variable_type;
    typedef std::size_t size_type;

    std::vector<std::pair<N, B>> invars;
    std::vector<std::pair<V, B>> outvars;

    typedef typename std::vector<std::pair<N, B>>::iterator invars_iterator;
    typedef typename std::vector<std::pair<V, B>>::iterator outvars_iterator;
    typedef typename std::vector<std::pair<V, B>>::const_iterator const_outvars_iterator;

    AdjArrayBQM() {}

    // can we specify this slightly better?
    template<class BQM>
    AdjArrayBQM(BQM &bqm) {

        // We know how big we'll need to be. Note that num_interactions is
        // O(|V|) for the shapeable bqms. Testing shows it's faster to do it
        // though.
        invars.reserve(bqm.num_variables());
        outvars.reserve(2*bqm.num_interactions());

        for (auto v = 0; v < bqm.num_variables(); ++v) {
            invars.emplace_back(outvars.size(), bqm.get_linear(v));

            auto span = bqm.neighborhood(v);
            outvars.insert(outvars.end(), span.first, span.second);
        }
    }

    ~AdjArrayBQM() {}

    size_type num_interactions() const {
        return outvars.size() / 2;
    }

    size_type num_variables() const {
        return invars.size();
    }

    bias_type get_linear(variable_type v) const {
        return invars[v].second;
    }

    std::pair<bias_type, bool>
    get_quadratic(variable_type u, variable_type v) const {
        assert(u >= 0 && u < invars.size());
        assert(v >= 0 && v < invars.size());
        assert(u != v);

        auto span = neighborhood(u);
        auto low = std::lower_bound(span.first, span.second, v,
                                    [](const std::pair<V, B> ub, V v) {return ub.first < v;});

        if (low == span.second)
            return std::make_pair(0, false);
        return std::make_pair(low->second, true);
    }

    size_type degree(variable_type v) const {
        assert(v >= 0 && v < invars.size());

        // need to check the case that v is the last variable
        if ((unsigned) v == invars.size() - 1)
            return outvars.size() - invars[v].first;

        return invars[v+1].first - invars[v].first;
    }

    std::pair<outvars_iterator, outvars_iterator>
    neighborhood(variable_type u) {
        assert(u >= 0 && u < invars.size());

        outvars_iterator end;
        if (((unsigned) u == invars.size())) {
            end = outvars.end();
        } else {
            end = outvars.begin() + invars[u+1].first;
        }

        return std::make_pair(outvars.begin() + invars[u].first, end);
    }

    std::pair<const_outvars_iterator, const_outvars_iterator>
    neighborhood(variable_type u) const {
        assert(u >= 0 && u < invars.size());

        const_outvars_iterator end;
        if (((unsigned) u == invars.size())) {
            end = outvars.cend();
        } else {
            end = outvars.cbegin() + invars[u+1].first;
        }

        return std::make_pair(outvars.cbegin() + invars[u].first, end);
    }

    void set_linear(variable_type v, bias_type b) {
        assert(v >= 0 && v < invars.size());
        invars[v].second = b;
    }

    bool set_quadratic(variable_type u, variable_type v, bias_type b) {
        assert(u >= 0 && u < invars.size());
        assert(v >= 0 && v < invars.size());
        assert(u != v);

        auto span = neighborhood(u);

        auto low = std::lower_bound(span.first, span.second, v,
                                    [](const std::pair<V, B> ub, V v) {return ub.first < v;});

        // if u, v does not exist when we are done
        if (low == span.second) return false;

        low->second = b;

        span = neighborhood(v);
        low = std::lower_bound(span.first, span.second, u,
                              [](const std::pair<V, B> ub, V v) {return ub.first < v;});

        low->second = b;  // can rely on it existing

        return true;
    }

};

}  // namespace dimod

#endif  // DIMOD_BQM_SRC_ADJARRAYBQM_H_
