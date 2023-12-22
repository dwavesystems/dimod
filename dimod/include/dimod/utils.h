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
#include <limits>
#include <utility>
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

    // zip_sort is a modification of the code found here :
    // https://www.geeksforgeeks.org/iterative-quick-sort/

    /**
     * Sort two vectors, using `control` to provide the ordering.
     *
     * Note that this only sorts by the `control`, the values of `response`
     * are ignored.
     */
    template <class T1, class T2>
    void zip_sort(std::vector<T1>& control, std::vector<T2>& response) {
        assert(control.size() == response.size());
        std::size_t length = control.size();

        if (length < 2) {
            return;
        } else if (length > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
            throw std::logic_error(
                    "Length of arrays are too big for sorting. Numerical "
                    "overflow will occur.");
        }

        std::int64_t* stack = (std::int64_t*)malloc(length * sizeof(std::int64_t));
        std::int64_t top = -1;
        std::int64_t low = 0;
        std::int64_t high = length - 1;

        stack[++top] = low;
        stack[++top] = high;

        while (top >= 0) {
            high = stack[top--];
            low = stack[top--];

            std::int64_t pivot_choice_index = (low + high + 1) / 2;
            if (pivot_choice_index != high) {
                T1 control_temp = control[pivot_choice_index];
                control[pivot_choice_index] = control[high];
                control[high] = control_temp;

                T2 response_temp = response[pivot_choice_index];
                response[pivot_choice_index] = response[high];
                response[high] = response_temp;
            }

            T1 pivot_element = control[high];
            std::int64_t i = low - 1;

            for (std::int64_t j = low; j <= high - 1; j++) {
                if (control[j] <= pivot_element) {
                    i++;
                    if (i != j) {
                        T1 control_temp = control[i];
                        control[i] = control[j];
                        control[j] = control_temp;

                        T2 response_temp = response[i];
                        response[i] = response[j];
                        response[j] = response_temp;
                    }
                }
            }

            // Pivot position.
            std::int64_t p = i + 1;

            T1 control_temp = control[p];
            control[p] = control[high];
            control[high] = control_temp;

            T2 response_temp = response[p];
            response[p] = response[high];
            response[high] = response_temp;

            if (p - 1 > low) {
                stack[++top] = low;
                stack[++top] = p - 1;
            }

            if (p + 1 < high) {
                stack[++top] = p + 1;
                stack[++top] = high;
            }
        }

        free(stack);
    }

}  // namespace utils
}  // namespace dimod
