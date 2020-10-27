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

#ifndef DIMOD_UTILS_H_
#define DIMOD_UTILS_H_

#include <utility>

#define BLOCK_SIZE 64  // Block size for cache blocking.

namespace dimod {
namespace utils {

    template <class V, class B>
    bool comp_v(std::pair<V, B> ub, V v) {
        return ub.first < v;
    }

}  // namespace utils
}  // namespace dimod

#endif  // DIMOD_UTILS_H_
