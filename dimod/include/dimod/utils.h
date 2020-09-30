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

#define BLOCK_SIZE 64 // Block size for cache blocking.
#define CACHE_LINE_SIZE 64 // General cache line size in bytes.

namespace dimod {
namespace utils {

template<class V, class B>
bool comp_v(std::pair<V, B> ub, V v) {
    return ub.first < v;
}

// The aligned_malloc and aligned_free functions were written with the help of this link:
// https://stackoverflow.com/questions/38088732/explanation-to-aligned-malloc-implementation

// Allocate memory and make sure the returned pointer address
// is a multiple of the given alignment
void* aligned_malloc(size_t required_bytes, size_t alignment = 0) {
    if (!alignment) {
        alignment = CACHE_LINE_SIZE;
    }

    void* p1;   // original pointer
    void** p2;  // aligned pointer
    int extra_bytes = alignment - 1 + sizeof(void*);

    if ((p1 = (void*)malloc(required_bytes + extra_bytes)) == NULL) {
        return NULL;
    }
    p2 = (void**)(alignment * (((size_t)(p1) + extra_bytes) / alignment));
    p2[-1] = p1;
    return p2;
}

// Corresponding aligned free for the aligned malloc
void aligned_free(void* p) { free(((void**)p)[-1]); }

// Allocate memory and fill it with zeroes but also make sure
// the returned address is a multiple of the given alignment
void* aligned_calloc(size_t num, size_t size, size_t alignment = 0) {
    if (!alignment) {
        alignment = CACHE_LINE_SIZE;
    }

    size_t required_bytes = num * size;
    void* ptr = aligned_malloc(required_bytes, alignment);
    long long int* ptr_ll = (long long int*)ptr;
    size_t numfill_by_ll = required_bytes / sizeof(long long int);

    #pragma omp parallel for schedule(static)
    for (size_t i_ll = 0; i_ll < numfill_by_ll; i_ll++) {
        ptr_ll[i_ll] = 0;
    }

    char* ptr_char = (char*)(ptr_ll + numfill_by_ll);
    size_t bytes_left = required_bytes - (numfill_by_ll * sizeof(long long int));

    for (size_t i_char = 0; i_char < bytes_left; i_char++) {
        ptr_char[i_char] = 0;
    }

    return ptr;
}

}  // namespace utils
}  // namespace dimod

#endif  // DIMOD_UTILS_H_
