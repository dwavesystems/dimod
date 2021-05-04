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

}  // namespace utils
}  // namespace dimod
