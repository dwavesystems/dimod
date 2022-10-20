// Copyright 2022 D-Wave Systems Inc.
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

#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dimod/abc.h"
#include "dimod/expression.h"
#include "dimod/vartypes.h"

namespace dimod {

enum Sense { LE, GE, EQ };

enum Penalty { LINEAR, QUADRATIC, CONSTANT };

template <class Bias, class Index>
class Constraint : public Expression<Bias, Index> {
 public:
    /// The type of the base class.
    using base_type = Expression<Bias, Index>;

    /// The first template parameter (Bias).
    using bias_type = Bias;

    /// The second template parameter (Index).
    using index_type = Index;

    using size_type = typename base_type::size_type;

    using parent_type = ConstrainedQuadraticModel<bias_type, index_type>;

    Constraint();
    explicit Constraint(const parent_type* parent);

    bool is_onehot() const;
    bool is_soft() const;
    void mark_discrete(bool mark = true);
    bool marked_discrete() const;
    Penalty penalty() const;
    bias_type rhs() const;

    // note: flips sign when negative
    void scale(bias_type scalar);

    Sense sense() const;
    void set_penalty(Penalty penalty);
    void set_rhs(bias_type rhs);
    void set_sense(Sense sense);
    void set_weight(bias_type weight);
    bias_type weight() const;

 private:
    Sense sense_;
    bias_type rhs_;
    bias_type weight_;
    Penalty penalty_;

    // marker(s) - these ar not enforced by code
    bool marked_discrete_ = false;
};

template <class bias_type, class index_type>
Constraint<bias_type, index_type>::Constraint() : Constraint(nullptr) {}

template <class bias_type, class index_type>
Constraint<bias_type, index_type>::Constraint(const parent_type* parent)
        : base_type(parent),
          sense_(Sense::EQ),
          rhs_(0),
          weight_(std::numeric_limits<bias_type>::infinity()),
          penalty_(Penalty::LINEAR) {}

template <class bias_type, class index_type>
bool Constraint<bias_type, index_type>::is_onehot() const {
    // must be linear and must have at least two variables
    if (!base_type::is_linear() || base_type::num_variables() < 2) return false;

    // must be equality
    if (sense_ != Sense::EQ) return false;

    // we could check the rhs vs offset, but let's treat it canonically as
    // needing no offset
    if (base_type::offset()) return false;

    // all of out variables must be binary
    for (const auto& v : base_type::variables()) {
        if (base_type::vartype(v) != Vartype::BINARY) return false;
    }

    // we get a bit cute here and check the linear biases using the
    // underlying model, thereby bypassing the dict lookup. This is not
    // super future-proof but faster.
    for (size_type i = 0; i < base_type::num_variables(); ++i) {
        if (base_type::base_type::linear(i) != rhs_) return false;
    }

    return true;
}

template <class bias_type, class index_type>
bool Constraint<bias_type, index_type>::is_soft() const {
    // some sort of tolerance value?
    return weight_ != std::numeric_limits<bias_type>::infinity();
}

template <class bias_type, class index_type>
void Constraint<bias_type, index_type>::mark_discrete(bool marker) {
    marked_discrete_ = marker;
}

template <class bias_type, class index_type>
bool Constraint<bias_type, index_type>::marked_discrete() const {
    return marked_discrete_;
}

template <class bias_type, class index_type>
Penalty Constraint<bias_type, index_type>::penalty() const {
    return penalty_;
}

template <class bias_type, class index_type>
bias_type Constraint<bias_type, index_type>::rhs() const {
    return rhs_;
}

template <class bias_type, class index_type>
void Constraint<bias_type, index_type>::scale(bias_type scalar) {
    base_type::scale(scalar);
    rhs_ *= scalar;
    if (scalar < 0) {
        if (sense_ == Sense::LE) {
            sense_ = Sense::GE;
        } else if (sense_ == Sense::GE) {
            sense_ = Sense::LE;
        }
    }
}

template <class bias_type, class index_type>
void Constraint<bias_type, index_type>::set_penalty(Penalty penalty) {
    penalty_ = penalty;
}

template <class bias_type, class index_type>
void Constraint<bias_type, index_type>::set_rhs(bias_type rhs) {
    rhs_ = rhs;
}

template <class bias_type, class index_type>
void Constraint<bias_type, index_type>::set_sense(Sense sense) {
    sense_ = sense;
}

template <class bias_type, class index_type>
void Constraint<bias_type, index_type>::set_weight(bias_type weight) {
    weight_ = weight;
}

template <class bias_type, class index_type>
Sense Constraint<bias_type, index_type>::sense() const {
    return sense_;
}

template <class bias_type, class index_type>
bias_type Constraint<bias_type, index_type>::weight() const {
    return weight_;
}

}  // namespace dimod
