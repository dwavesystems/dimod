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
#include <cmath>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dimod/constrained.h"

namespace dimod {
namespace presolve {

enum Status { FEASIBLE, INFEASIBLE, UNBOUNDED };

template <class Bias, class Index>
class PreSolveBase;

// template <class Bias, class Index>
// class PreSolver;

template <class Bias, class Index, class Assignment>
class PostSolver {
 public:
    /// The first template parameter (Bias).
    using bias_type = Bias;

    /// The second template parameter (Index).
    using index_type = Index;

    /// Unsigned integral type that can represent non-negative values.
    using size_type = std::size_t;

    using assignment_type = Assignment;

    // friend class PreSolver<bias_type, index_type>;

    explicit PostSolver(index_type n) : variables_(), assignments_() {
        assert(n >= 0);
        for (index_type i = 0; i < n; ++i) {
            variables_.push_back(i);
        }
    }

    template <class T>
    std::vector<T> apply(const std::vector<T>& reduced) const {
        assert(reduced.size() == variables_.size());

        std::vector<T> original(num_variables_fixed() + num_variables_unfixed());

        // add the fixed from assignments_
        for (const auto& p : assignments_) {
            original[p.first] = p.second;
        }

        // add the unfixed from variables_/reduced
        for (size_type i = 0; i < variables_.size(); ++i) {
            original[variables_[i]] = reduced[i];
        }

        return original;
    }

    void fix_variable(index_type v, assignment_type assignment) {
        assert(v >= 0 && static_cast<size_type>(v) < variables_.size());
        assignments_[variables_[v]] = assignment;
        variables_.erase(variables_.begin() + v);
    }

    size_type num_variables_fixed() const { return this->assignments_.size(); }

    size_type num_variables_unfixed() const { return this->variables_.size(); }

 private:
    std::vector<index_type> variables_;
    std::unordered_map<index_type, assignment_type> assignments_;
};

template <class Bias, class Index = int, class Assignment = double>
class PreSolver {
 public:
    /// The first template parameter (Bias).
    using bias_type = Bias;

    /// The second template parameter (Index).
    using index_type = Index;

    /// Unsigned integral type that can represent non-negative values.
    using size_type = std::size_t;

    using assignment_type = Assignment;

    using model_type = ConstrainedQuadraticModel<bias_type, index_type>;

    using postsolver_type = PostSolver<bias_type, index_type, assignment_type>;

 private:
    model_type model_;

    std::vector<std::unique_ptr<PreSolveBase<bias_type, index_type>>> presolvers_;

    postsolver_type postsolver_;

 public:
    template <class B, class I>
    explicit PreSolver(const ConstrainedQuadraticModel<B, I>& cqm)
            : model_(cqm), postsolver_(cqm.num_variables()) {}

    explicit PreSolver(model_type&& cqm)
            : model_(std::move(cqm)), postsolver_(model_.num_variables()) {
        assert(this->model_.num_variables() == this->postsolver_.num_variables_unfixed());
    }

    template <class T>
    void add_presolver() {
        this->presolvers_.emplace_back(
                std::unique_ptr<PreSolveBase<bias_type, index_type>>(new T()));
    }

    void apply() {
        for (auto&& presolver : this->presolvers_) {
            presolver->apply(this);
        }
    }

    void fix_variable(index_type v, assignment_type assignment) {
        assert(v >= 0 && static_cast<size_type>(v) < this->model_.num_variables());
        this->model_.fix_variable(v, assignment);
        this->postsolver_.fix_variable(v, assignment);
        assert(this->model_.num_variables() == this->postsolver_.num_variables_unfixed());
    }

    const model_type& model() const { return this->model_; }

    const postsolver_type& postsolver() const { return this->postsolver_; }

    /// Remove any variables where the upper and lower bounds are the same
    size_type remove_fixed_variables() {
        size_type start_size = this->model_.num_variables();
        for (size_type v = 0; v < this->model_.num_variables();) {
            if (this->model_.lower_bound(v) == this->model_.upper_bound(v)) {
                this->fix_variable(v, this->model_.lower_bound(v));
            } else {
                ++v;
            }
        }
        return start_size - this->model_.num_variables();
    }

    // Remove any constraints with 0 or 1 variables.
    size_type remove_trivial_constraints() {
        size_type original_size = this->model_.num_constraints();

        size_type i = 0;
        while (i < this->model_.num_constraints()) {
            auto& c = this->model_.constraints[i];

            if (c.num_variables() == 0) {
                // remove after checking feasibity
                throw std::logic_error("hello 78");
            } else if (c.num_variables() == 1) {
                index_type v = c.variable(0);

                // a * x + b ◯ c
                bias_type a = c.linear(v);
                if (!a) {
                    throw std::logic_error("hello 98");
                }
                bias_type rhs = (c.rhs() - c.offset()) / a;

                switch (c.sense()) {
                    case Sense::EQ:
                        this->model_.set_lower_bound(v, std::max(rhs, this->model_.lower_bound(v)));
                        this->model_.set_upper_bound(v, std::min(rhs, this->model_.upper_bound(v)));
                        break;
                    case Sense::LE:
                        this->model_.set_upper_bound(v, std::min(rhs, this->model_.upper_bound(v)));
                        break;
                    case Sense::GE:
                        this->model_.set_lower_bound(v, std::max(rhs, this->model_.lower_bound(v)));
                        break;
                }

                this->model_.remove_constraint(i);
            } else {
                // advance
                ++i;
            }
        }

        return original_size - this->model_.num_constraints();
    }

    size_type tighten_bounds() {
        size_type count = 0;

        // should we also check against the vartype maxes and mins?

        for (size_type v = 0; v < model_.num_variables(); ++v) {
            switch (model_.vartype(v)) {
                case Vartype::SPIN:
                case Vartype::BINARY:
                case Vartype::INTEGER:
                    // round bounds
                    model_.set_upper_bound(v, std::floor(model_.upper_bound(v)));
                    model_.set_lower_bound(v, std::ceil(model_.lower_bound(v)));
                    break;
                case Vartype::REAL:
                    break;
            }
        }

        return count;
    }
};

template <class Bias, class Index>
class PreSolveBase {
 public:
    /// The first template parameter (Bias).
    using bias_type = Bias;

    /// The second template parameter (Index).
    using index_type = Index;

    /// Unsigned integral type that can represent non-negative values.
    using size_type = std::size_t;

    virtual void apply(PreSolver<bias_type, index_type>* presolver) = 0;
};

namespace techniques {

/*
 * Remove constraints of the form a*x ◯ b
 * Remove fixed variables
 * Round bounds on BINARY/SPIN/INTEGER variables
 * Convert SPIN variables to BINARY
 */
template <class Bias, class Index = int>
class TrivialPresolver : public PreSolveBase<Bias, Index> {
 public:
    /// The first template parameter (Bias).
    using bias_type = Bias;

    /// The second template parameter (Index).
    using index_type = Index;

    /// Unsigned integral type that can represent non-negative values.
    using size_type = std::size_t;

    void apply(PreSolver<bias_type, index_type>* presolver) {
        // convert all SPIN variables into BINARY
        for (size_type v = 0; v < presolver->model().num_variables(); ++v) {
            if (presolver->model().vartype(v) == Vartype::SPIN) {
                throw std::logic_error("todo 233");
            }
        }

        // do rounds of removing constraints, tightening bounds, removing fixed
        // variables until no changes
        bool changes = true;
        while (changes) {
            changes = presolver->remove_trivial_constraints();
            changes = changes || presolver->tighten_bounds();
            changes = changes || presolver->remove_fixed_variables();
        }
    }
};
}  // namespace techniques
}  // namespace presolve
}  // namespace dimod
