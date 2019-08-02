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
    Bias get_quadratic(const AdjArrayInVars<Bias> &invars,
                       const AdjArrayOutVars<VarIndex, Bias> &outvars,
                       VarIndex u, VarIndex v) {
        assert(u >= 0 && u < invars.size());
        assert(v >= 0 && v < invars.size());
        assert(u != v);

        if (v < u) std::swap(u, v);

        std::size_t start = invars[u].first;
        std::size_t end = invars[u+1].first;  // safe if u < v and asserts above

        const std::pair<VarIndex, Bias> target(v, 0);

        typename std::vector<std::pair<VarIndex, Bias>>::const_iterator low;
        low = std::lower_bound(outvars.begin()+start, outvars.begin()+end,
                               target, pair_lt<VarIndex, Bias>);

        if (low == outvars.end())
            return 0;  // do we want to raise?
        return (*low).second;
    }
}  // namespace dimod
