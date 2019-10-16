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
    std::size_t num_variables(const AdjArrayInVars<Bias> &invars,
                              const AdjArrayOutVars<VarIndex, Bias> &outvars) {
        return invars.size();
    }

    template<typename VarIndex, typename Bias>
    std::size_t num_interactions(const AdjArrayInVars<Bias> &invars,
                                 const AdjArrayOutVars<VarIndex, Bias>
                                &outvars) {
        return outvars.size() / 2;
    }

    template<typename VarIndex, typename Bias>
    Bias get_linear(const AdjArrayInVars<Bias> &invars,
                    const AdjArrayOutVars<VarIndex, Bias> &outvars,
                    VarIndex v) {
        assert(v >= 0 && v < invars.size());
        return invars[v].second;
    }

    // compare only the first value in the pair
    template<typename VarIndex, typename Bias>
    bool pair_lt(const std::pair<VarIndex, Bias> a,
                 const std::pair<VarIndex, Bias> b) {
        return a.first < b.first;
    }

    template<typename VarIndex, typename Bias>
    std::pair<Bias, bool> get_quadratic(const AdjArrayInVars<Bias> &invars,
                                        const AdjArrayOutVars<VarIndex, Bias>
                                        &outvars,
                                        VarIndex u, VarIndex v) {
        assert(u >= 0 && u < invars.size());
        assert(v >= 0 && v < invars.size());
        assert(u != v);

        if (v < u) std::swap(u, v);

        std::size_t start = invars[u].first;
        std::size_t end = invars[u+1].first;  // safe if u < v and asserts above

        const std::pair<VarIndex, Bias> target(v, 0);

        typename AdjArrayOutVars<VarIndex, Bias>::const_iterator low;
        low = std::lower_bound(outvars.begin()+start, outvars.begin()+end,
                               target, pair_lt<VarIndex, Bias>);

        if (low == outvars.begin()+end)
            return std::make_pair(0, false);
        return std::make_pair((*low).second, true);
    }

    // Iterate over the neighbourhood of variable u.
    //
    // Returns iterators to beginning and end of neighborhood.
    template<typename VarIndex, typename Bias>
    std::pair<typename AdjArrayOutVars<VarIndex, Bias>::const_iterator,
              typename AdjArrayOutVars<VarIndex, Bias>::const_iterator>
    neighborhood(const AdjArrayInVars<Bias> &invars,
                 const AdjArrayOutVars<VarIndex, Bias> &outvars,
                 VarIndex u) {
        std::size_t start, stop;

        start = invars[u].first;

        if (u == invars.size() - 1) {
            // last element so ends at the end out outvars
            stop = outvars.size();
        } else {
            stop = invars[u+1].first;
        }

        // random access iterators
        return std::make_pair(outvars.begin() + start, outvars.begin() + stop);
    }

    // Change the values in the BQM

    template<typename VarIndex, typename Bias>
    void set_linear(AdjArrayInVars<Bias> &invars,
                    AdjArrayOutVars<VarIndex, Bias> &outvars,
                    VarIndex v, Bias b) {
        assert(v >= 0 && v < invars.size());
        invars[v].second = b;
    }

    // Q: Should we do something else if the user tries to set a non-existant
    //    quadratic bias? Error/segfault?
    template<typename VarIndex, typename Bias>
    bool set_quadratic(AdjArrayInVars<Bias> &invars,
                       AdjArrayOutVars<VarIndex, Bias> &outvars,
                       VarIndex u, VarIndex v, Bias b) {
        assert(u >= 0 && u < invars.size());
        assert(v >= 0 && v < invars.size());
        assert(u != v);

        typename AdjArrayOutVars<VarIndex, Bias>::iterator low;
        std::size_t start, end;

        // interaction (u, v)
        start = invars[u].first;
        if (u < invars.size() - 1) {
            end = invars[u+1].first;
        } else {
            end = outvars.size();
        }

        const std::pair<VarIndex, Bias> v_target(v, 0);

        low = std::lower_bound(outvars.begin()+start, outvars.begin()+end,
                               v_target, pair_lt<VarIndex, Bias>);

        // if the edge does not exist, just return false here
        if (low == outvars.begin()+end)
            return false;

        (*low).second = b;

        // interaction (v, u)
        start = invars[v].first;
        if (v < invars.size() - 1) {
            end = invars[v+1].first;
        } else {
            end = outvars.size();
        }

        const std::pair<VarIndex, Bias> u_target(u, 0);

        low = std::lower_bound(outvars.begin()+start, outvars.begin()+end,
                               u_target, pair_lt<VarIndex, Bias>);

        // we already know that this exists
        (*low).second = b;

        return true;
    }

}  // namespace dimod
