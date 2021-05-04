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
#include <climits>
#include <utility>
#include <vector>

#define BLOCK_SIZE 64  // Block size for cache blocking.

namespace dimod {
namespace utils {

    template <class V, class B>
    bool comp_v(std::pair<V, B> ub, V v) {
        return ub.first < v;
    }

    template <class T1, class T2>
    void zip_sort(std::vector<T1> &keys, std::vector<T2> &values) {
        std::size_t size = std::min(keys.size(), values.size());

        std::vector<std::pair<T1, T2>> zipped;
        zipped.reserve(size);

        for (std::size_t i = 0; i < size; ++i) {
            zipped.emplace_back(keys[i], values[i]);
        }

        std::sort(zipped.begin(), zipped.end());

        for (std::size_t i = 0; i < size; ++i) {
            keys[i] = zipped[i].first;
            values[i] = zipped[i].second;
        }
    }

    // This code is a modification of the code found here :
    // https://www.geeksforgeeks.org/iterative-quick-sort/
    template <class C_t, class R_t>
    void sort_vector_pairs(std::vector<C_t>& control,
                           std::vector<R_t>& response) {
        assert(control.size() == response.size());
        size_t length = control.size();

        if (length < 2) {
            return;
        } else if (length > LLONG_MAX) {
            throw std::logic_error(
                    "Length of arrays are too big for sorting. Numerical "
                    "overflow will occur.");
        }

        long long int* stack =
                (long long int*)malloc(length * sizeof(long long int));
        long long int top = -1;
        long long int low = 0;
        long long int high = length - 1;

        stack[++top] = low;
        stack[++top] = high;

        while (top >= 0) {
            high = stack[top--];
            low = stack[top--];

            long long int pivot_choice_index = (low + high + 1) / 2;
            if (pivot_choice_index != high) {
                C_t control_temp = control[pivot_choice_index];
                control[pivot_choice_index] = control[high];
                control[high] = control_temp;
            }

            C_t pivot_element = control[high];
            long long int i = low - 1;

            for (long long int j = low; j <= high - 1; j++) {
                if (control[j] <= pivot_element) {
                    i++;
                    if (i != j) {
                        C_t control_temp = control[i];
                        control[i] = control[j];
                        control[j] = control_temp;

                        R_t response_temp = response[i];
                        response[i] = response[j];
                        response[j] = response_temp;
                    }
                }
            }

            // Pivot position.
            long long int p = i + 1;

            C_t control_temp = control[p];
            control[p] = control[high];
            control[high] = control_temp;

            R_t response_temp = response[p];
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
