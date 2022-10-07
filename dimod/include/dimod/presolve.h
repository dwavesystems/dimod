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

#include <algorithm>
#include <utility>
#include <vector>

#include "dimod/constrained_quadratic_model.h"

namespace dimod {
namespace presolve {

template <class Bias, class Index, class Assignment>
class PostSolver {
 public:
    using bias_type = Bias;
    using index_type = Index;
    using size_type = std::size_t;
    using assignment_type = Assignment;

    template <class T>
    std::vector<T> apply(std::vector<T> reduced) const;

    void fix_variable(index_type v, assignment_type value);

    void substitute_variable(index_type v, bias_type multiplier, bias_type offset);

 private:
    // we want to track what changes were made
    enum TransformKind { FIX, SUBSTITUTE };

    // todo: we could get fancy with pointers and templates to save a bit of
    // space here
    struct Transform {
        TransformKind kind;
        index_type v;           // what variable it was applied to
        assignment_type value;  // if it was fixed what it was fixed to
        bias_type multiplier;
        bias_type offset;

        explicit Transform(TransformKind kind)
                : kind(kind), v(-1), value(NAN), multiplier(NAN), offset(NAN) {}
    };

    std::vector<Transform> transforms_;
};

template <class bias_type, class index_type, class assignment_type>
template <class T>
std::vector<T> PostSolver<bias_type, index_type, assignment_type>::apply(
        std::vector<T> sample) const {
    // all that we have to do is undo the transforms back to front.
    for (auto it = transforms_.crbegin(); it != transforms_.crend(); ++it) {
        switch (it->kind) {
            case TransformKind::FIX:
                sample.insert(sample.begin() + it->v, it->value);
                break;
            case TransformKind::SUBSTITUTE:
                sample[it->v] *= it->multiplier;
                sample[it->v] += it->offset;
                break;
        }
    }
    return sample;
}

template <class bias_type, class index_type, class assignment_type>
void PostSolver<bias_type, index_type, assignment_type>::fix_variable(index_type v,
                                                                      assignment_type value) {
    transforms_.emplace_back(TransformKind::FIX);
    transforms_.back().v = v;
    transforms_.back().value = value;
}

template <class bias_type, class index_type, class assignment_type>
void PostSolver<bias_type, index_type, assignment_type>::substitute_variable(index_type v,
                                                                      bias_type multiplier,
                                                                      bias_type offset) {
    assert(multiplier);  // cannot undo when it's 0
    transforms_.emplace_back(TransformKind::SUBSTITUTE);
    transforms_.back().v = v;
    transforms_.back().multiplier = multiplier;
    transforms_.back().offset = offset;
}

template <class Bias, class Index = int, class Assignment = double>
class PreSolver {
 public:
    using model_type = ConstrainedQuadraticModel<Bias, Index>;

    using bias_type = Bias;
    using index_type = Index;
    using size_type = typename model_type::size_type;

    using assignment_type = Assignment;

    PreSolver();
    explicit PreSolver(model_type model);

    void apply();

    void load_default_presolvers();

    const model_type& model() const;
    const PostSolver<bias_type, index_type, assignment_type>& postsolver() const;

 private:
    model_type model_;
    PostSolver<bias_type, index_type, assignment_type> postsolver_;
};


template <class bias_type, class index_type, class assignment_type>
PreSolver<bias_type, index_type, assignment_type>::PreSolver(): model_(), postsolver_() {}

template <class bias_type, class index_type, class assignment_type>
PreSolver<bias_type, index_type, assignment_type>::PreSolver(model_type model)
        : model_(std::move(model)), postsolver_() {}

template <class bias_type, class index_type, class assignment_type>
void PreSolver<bias_type, index_type, assignment_type>::apply() {
    // todo: actually read from a vector of techniques or similar

    // One time techniques ----------------------------------------------------

    // *-- spin-to-binary
    for (size_type v = 0; v < model_.num_variables(); ++v) {
        if (model_.vartype(v) == Vartype::SPIN) {
            postsolver_.substitute_variable(v, 2, -1);
            model_.change_vartype(Vartype::BINARY, v);
        }
    }

    // *-- remove offsets
    for (size_type c = 0; c < model_.num_constraints(); ++c) {
        auto& constraint = model_.constraint_ref(c);
        if (constraint.offset()) {
            constraint.set_rhs(constraint.rhs() - constraint.offset());
            constraint.set_offset(0);
        }
    }

    // *-- flip >= constraints
    for (size_type c = 0; c < model_.num_constraints(); ++c) {
        auto& constraint = model_.constraint_ref(c);
        if (constraint.sense() == Sense::GE) {
            constraint.scale(-1);
        }
    }

    // Trivial techniques -----------------------------------------------------

    bool changes = true;
    const index_type max_num_rounds = 100;  // todo: make configurable
    for (index_type num_rounds = 0; num_rounds < max_num_rounds; ++num_rounds) {
        if (!changes) break;
        changes = false;

        // *-- todo: clear out 0 variables/interactions in the constraints

        // *-- remove single variable constraints
        size_type c = 0;
        while (c < model_.num_constraints()) {
            auto& constraint = model_.constraint_ref(c);

            if (constraint.num_variables() == 0) {
                // remove after checking feasibity
                throw std::logic_error("not implemented - infeasible");
            } else if (constraint.num_variables() == 1) {
                index_type v = constraint.variables()[0];

                // ax ◯ c
                bias_type a = constraint.linear(v);
                assert(a);  // should have already been removed if 0

                // offset should have already been removed but may as well be safe
                bias_type rhs = (constraint.rhs() - constraint.offset()) / a;

                // todo: test if negative

                if (constraint.sense() == Sense::EQ) {
                        model_.set_lower_bound(v, std::max(rhs, model_.lower_bound(v)));
                        model_.set_upper_bound(v, std::min(rhs, model_.upper_bound(v)));
                } else if ((constraint.sense() == Sense::LE) != (a < 0)) {
                    model_.set_upper_bound(v, std::min(rhs, model_.upper_bound(v)));
                } else {
                    assert((constraint.sense() == Sense::GE) == (a >= 0));
                    model_.set_lower_bound(v, std::max(rhs, model_.lower_bound(v)));
                }

                model_.remove_constraint(c);
                changes = true;
                continue;
            }

            ++c;
        }

        // *-- tighten bounds based on vartype
        bias_type lb;
        bias_type ub;
        for (size_type v = 0; v < model_.num_variables(); ++v) {
            switch (model_.vartype(v)) {
                case Vartype::SPIN:
                case Vartype::BINARY:
                case Vartype::INTEGER:
                    ub = model_.upper_bound(v);
                    if (ub != std::floor(ub)) {
                        model_.set_upper_bound(v, std::floor(ub));
                        changes = true;
                    }
                    lb = model_.lower_bound(v);
                    if (lb != std::ceil(lb)) {
                        model_.set_lower_bound(v, std::ceil(lb));
                        changes = true;
                    }
                    break;
                case Vartype::REAL:
                    break;
            }
        }

        // *-- remove variables that are fixed by bounds
        size_type v = 0;
        while (v < model_.num_variables()) {
            if (model_.lower_bound(v) == model_.upper_bound(v)) {
                postsolver_.fix_variable(v, model_.lower_bound(v));
                model_.fix_variable(v, model_.lower_bound(v));
            }
            ++v;
        }
    }
}

template <class bias_type, class index_type, class assignment_type>
void PreSolver<bias_type, index_type, assignment_type>::load_default_presolvers() {
    // placeholder, does nothing at the moment
}

template <class bias_type, class index_type, class assignment_type>
const ConstrainedQuadraticModel<bias_type, index_type>& PreSolver<bias_type, index_type, assignment_type>::model() const {
    return model_;
}

template <class bias_type, class index_type, class assignment_type>
const PostSolver<bias_type, index_type, assignment_type>&
PreSolver<bias_type, index_type, assignment_type>::postsolver() const {
    return postsolver_;
}

}  // namespace presolve
}  // namespace dimod
