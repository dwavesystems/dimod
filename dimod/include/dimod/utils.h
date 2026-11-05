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

#pragma once

#include <algorithm>
#include <ranges>
#include <tuple>
#include <vector>

namespace dimod {
namespace utils {

// Remove all elements in the range defined by vfirst to vlast at indices
// specified by ifirst to ilast.
// All iterators must be forward iterators
// Indices must be non-negative, sorted, and unique.
template <class ValueIter, class IndexIter>
ValueIter remove_by_index(ValueIter vfirst, ValueIter vlast, IndexIter ifirst, IndexIter ilast) {
    assert(std::is_sorted(ifirst, ilast));
    assert((ifirst == ilast || *ifirst >= 0));

    using value_type = typename std::iterator_traits<ValueIter>::value_type;

    typename std::iterator_traits<IndexIter>::value_type loc = 0;  // location in the values
    IndexIter it = ifirst;
    auto pred = [&](const value_type&) {
        if (it != ilast && *it == loc) {
            ++loc;
            ++it;
            return true;
        } else {
            ++loc;
            return false;
        }
    };

    // relies on this being executed sequentially
    return std::remove_if(vfirst, vlast, pred);
}

    /**
     * Sort two vectors, using `control` to provide the ordering.
     *
     * Note that this only sorts by the `control`, the values of `response`
     * are ignored.
     */
    template <class T1, class T2>
    void zip_sort(std::vector<T1>& control, std::vector<T2>& response) {
        assert(control.size() == response.size());
        std::ranges::sort(std::views::zip(control, response), {},
                          [](const auto& cr) { return std::get<0>(cr); });
    }

}  // namespace utils
}  // namespace dimod
