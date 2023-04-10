// Copyright 2021 D-Wave Systems Inc.
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

#include <cstdint>
#include <stdexcept>
#include <limits>

namespace dimod {

/// Encode the domain of a variable.
enum Vartype {
    BINARY,   ///< Variables that are either 0 or 1.
    SPIN,     ///< Variables that are either -1 or 1.
    INTEGER,  ///< Variables that are integer valued.
    REAL      ///< Variables that are real valued.
};

/// Compile-time limits by variable type.
template <class Bias, Vartype vartype>
class vartype_limits {};

template <class Bias>
class vartype_limits<Bias, Vartype::BINARY> {
 public:
    static constexpr Bias default_max() noexcept { return 1; }
    static constexpr Bias default_min() noexcept { return 0; }
    static constexpr Bias max() noexcept { return 1; }
    static constexpr Bias min() noexcept { return 0; }
};

template <class Bias>
class vartype_limits<Bias, Vartype::SPIN> {
 public:
    static constexpr Bias default_max() noexcept { return +1; }
    static constexpr Bias default_min() noexcept { return -1; }
    static constexpr Bias max() noexcept { return +1; }
    static constexpr Bias min() noexcept { return -1; }
};

template <class Bias>
class vartype_limits<Bias, Vartype::INTEGER> {
 public:
    static constexpr Bias default_max() noexcept { return max(); }
    static constexpr Bias default_min() noexcept { return 0; }
    static constexpr Bias max() noexcept {
        return ((std::int64_t)1 << (std::numeric_limits<Bias>::digits)) - 1;
    }
    static constexpr Bias min() noexcept { return -max(); }
};
template <class Bias>
class vartype_limits<Bias, Vartype::REAL> {
 public:
    static constexpr Bias default_max() noexcept { return max(); }
    static constexpr Bias default_min() noexcept { return 0; }
    static constexpr Bias max() noexcept { return 1e30; }
    static constexpr Bias min() noexcept { return -1e30; }
};

/// Runtime limits by variable type.
template <class Bias>
class vartype_info {
 public:
    /// Default upper bound
    static Bias default_max(Vartype vartype) {
        if (vartype == Vartype::BINARY) {
            return vartype_limits<Bias, Vartype::BINARY>::default_max();
        } else if (vartype == Vartype::SPIN) {
            return vartype_limits<Bias, Vartype::SPIN>::default_max();
        } else if (vartype == Vartype::INTEGER) {
            return vartype_limits<Bias, Vartype::INTEGER>::default_max();
        } else if (vartype == Vartype::REAL) {
            return vartype_limits<Bias, Vartype::REAL>::default_max();
        } else {
            throw std::logic_error("unknown vartype");
        }
    }
    /// Default lower bound
    static Bias default_min(Vartype vartype) {
        if (vartype == Vartype::BINARY) {
            return vartype_limits<Bias, Vartype::BINARY>::default_min();
        } else if (vartype == Vartype::SPIN) {
            return vartype_limits<Bias, Vartype::SPIN>::default_min();
        } else if (vartype == Vartype::INTEGER) {
            return vartype_limits<Bias, Vartype::INTEGER>::default_min();
        } else if (vartype == Vartype::REAL) {
            return vartype_limits<Bias, Vartype::REAL>::default_min();
        } else {
            throw std::logic_error("unknown vartype");
        }
    }
    /// Maximum supported value
    static Bias max(Vartype vartype) {
        if (vartype == Vartype::BINARY) {
            return vartype_limits<Bias, Vartype::BINARY>::max();
        } else if (vartype == Vartype::SPIN) {
            return vartype_limits<Bias, Vartype::SPIN>::max();
        } else if (vartype == Vartype::INTEGER) {
            return vartype_limits<Bias, Vartype::INTEGER>::max();
        } else if (vartype == Vartype::REAL) {
            return vartype_limits<Bias, Vartype::REAL>::max();
        } else {
            throw std::logic_error("unknown vartype");
        }
    }
    /// Minimum supported value
    static Bias min(Vartype vartype) {
        if (vartype == Vartype::BINARY) {
            return vartype_limits<Bias, Vartype::BINARY>::min();
        } else if (vartype == Vartype::SPIN) {
            return vartype_limits<Bias, Vartype::SPIN>::min();
        } else if (vartype == Vartype::INTEGER) {
            return vartype_limits<Bias, Vartype::INTEGER>::min();
        } else if (vartype == Vartype::REAL) {
            return vartype_limits<Bias, Vartype::REAL>::min();
        } else {
            throw std::logic_error("unknown vartype");
        }
    }
};

}  // namespace dimod
