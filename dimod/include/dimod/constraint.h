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

    using parent_type = ConstrainedQuadraticModel<bias_type, index_type>;

    Sense sense_;
    bias_type rhs_;
    bias_type weight_;
    Penalty penalty_;

    explicit Constraint(parent_type* parent);

    bool is_soft() const;
    Penalty penalty() const;
    bias_type rhs() const;
    Sense sense() const;
    void set_penalty(Penalty penalty);
    void set_rhs(bias_type rhs);
    void set_sense(Sense sense);
    void set_weight(bias_type weight);
    bias_type weight() const;
};

template <class bias_type, class index_type>
Constraint<bias_type, index_type>::Constraint(parent_type* parent)
        : base_type(parent),
          sense_(Sense::EQ),
          rhs_(0),
          weight_(std::numeric_limits<bias_type>::infinity()),
          penalty_(Penalty::LINEAR) {}

template <class bias_type, class index_type>
bool Constraint<bias_type, index_type>::is_soft() const {
    // some sort of tolerance value?
    return weight_ != std::numeric_limits<bias_type>::infinity();
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
