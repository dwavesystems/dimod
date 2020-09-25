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

namespace dimod {
namespace utils {

template<class V, class B>
bool comp_v(std::pair<V, B> ub, V v) {
    return ub.first < v;
}

template <class BQM>
class QuadraticProxy {
  public:
    using bias_type = typename BQM::bias_type;
    using variable_type = typename BQM::variable_type;

    QuadraticProxy(BQM& bqm, variable_type u, variable_type v) : bqm_(bqm), u_(u), v_(v) {}

    QuadraticProxy& operator=(const QuadraticProxy& other) { return *this = bias_type(other); }
    QuadraticProxy& operator=(const bias_type& b) {
        bqm_.set_quadratic(u_, v_, b);
        return *this;
    }
    operator bias_type() const { return bqm_.get_quadratic(u_, v_).first; }

    // The following compound assignments could later be improved to do fewer lookups into bqm.
    template <class T>
    QuadraticProxy& operator+=(const T& x) {
        return *this = *this + x;
    }

    template <class T>
    QuadraticProxy& operator-=(const T& x) {
        return *this = *this - x;
    }

    template <class T>
    QuadraticProxy& operator*=(const T& x) {
        return *this = *this * x;
    }

    template <class T>
    QuadraticProxy& operator/=(const T& x) {
        return *this = *this / x;
    }

  private:
    BQM& bqm_;
    variable_type u_;
    variable_type v_;
};

}  // namespace utils
}  // namespace dimod

#endif  // DIMOD_UTILS_H_
