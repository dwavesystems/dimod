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

#include <unordered_map>
#include <utility>
#include <vector>

#include "dimod/abc.h"
#include "dimod/quadratic_model.h"
#include "dimod/vartypes.h"

namespace dimod {

enum Sense { LE, GE, EQ };

template <class Bias, class Index>
class ConstrainedQuadraticModel;

template <class Bias, class Index>
class Constraint : public abc::QuadraticModelBase<Bias, Index> {
 public:
    /// The first template parameter (Bias).
    using bias_type = Bias;

    /// The second template parameter (Index).
    using index_type = Index;

    /// Unsigned integral type that can represent non-negative values.
    using size_type = std::size_t;

    using base_type = abc::QuadraticModelBase<bias_type, index_type>;

    using parent_type = ConstrainedQuadraticModel<bias_type, index_type>;

 private:
    const parent_type* parent_;

    std::vector<index_type> variables_;
    Sense sense_;
    bias_type rhs_;

    std::unordered_map<index_type, index_type> indices_;

 public:
    Constraint(const parent_type* parent, Sense sense)
            : base_type(), parent_(parent), variables_(), sense_(sense), rhs_(0) {}

    Constraint(const parent_type* parent, std::vector<index_type>&& variables,
               std::vector<bias_type>&& biases, Sense sense, bias_type rhs)
            : base_type(std::move(biases)),
              parent_(parent),
              variables_(std::move(variables)),
              sense_(sense),
              rhs_(rhs) {
        index_type i = 0;
        for (const index_type& v : this->variables_) {
            this->indices_[v] = i++;
        }
    }

    Constraint(const parent_type* parent, const Constraint& constraint)
            : base_type(constraint),
              parent_(parent),
              variables_(constraint.variables_),
              sense_(constraint.sense_),
              rhs_(constraint.rhs_),
              indices_(constraint.indices_) {}

    Constraint(Constraint&& other) noexcept { *this = std::move(other); }

    Constraint& operator=(Constraint&& other) noexcept {
        if (this != &other) {
            base_type::operator=(std::move(other));
            this->parent_ = other.parent_;
            this->variables_ = std::move(other.variables_);
            this->sense_ = other.sense_;
            this->rhs_ = other.rhs_;
            this->indices_ = std::move(other.indices_);
        }
        return *this;
    }

    void add_linear(index_type v, bias_type bias) {
        base_type::add_linear(this->add_variable(v), bias);
    }

    void add_quadratic(index_type u, index_type v, bias_type bias) {
        base_type::add_quadratic(this->add_variable(u), this->add_variable(v), bias);
    }

    [[noreturn]] void add_quadratic_back(index_type u, index_type v, bias_type bias) {
        throw std::logic_error("not implemented");
    }

    template <class T>
    void fix_variable(index_type v, T assignment) {
        throw std::logic_error("todo 110");
    }

    bias_type linear(index_type v) const {
        assert(v >= 0 && static_cast<size_type>(v) < this->parent_->num_variables());
        auto it = this->indices_.find(v);
        if (it == this->indices_.end()) {
            return 0;
        }
        return base_type::linear(it->second);
    }

    bias_type lower_bound(index_type v) const { return this->parent_->lower_bound(v); }

    bias_type rhs() const { return this->rhs_; }

    Sense sense() const { return this->sense_; }

    void set_linear(index_type v, bias_type bias) {
        base_type::set_linear(this->add_variable(v), bias);
    }

    void set_quadratic(index_type u, index_type v, bias_type bias) {
        base_type::set_quadratic(this->add_variable(u), this->add_variable(v), bias);
    }

    bias_type upper_bound(index_type v) const { return this->parent_->upper_bound(v); }

    index_type variable(size_type i) const { return this->variables_[i]; }

    Vartype vartype(index_type v) const { return this->parent_->vartype(v); }

 private:
    /// Returns the index of v
    index_type add_variable(index_type v) {
        assert(v >= 0 && static_cast<size_type>(v) < this->parent_->num_variables());
        auto it = this->indices_.find(v);
        if (it == this->indices_.end()) {
            // we need to add a new variable to the constraint
            this->variables_.push_back(v);
            this->indices_.emplace(v, this->variables_.size() - 1);
            return base_type::add_variable();
        } else {
            return it->second;
        }
    }
};

template <class Bias, class Index = int>
class ConstrainedQuadraticModel {
 public:
    /// The first template parameter (Bias).
    using bias_type = Bias;

    /// The second template parameter (Index).
    using index_type = Index;

    /// Unsigned integral type that can represent non-negative values.
    using size_type = std::size_t;

    // using linear_type = LinearTerm<bias_type, index_type>;

    using constraint_type = Constraint<bias_type, index_type>;

 private:
    // The objective holds the information about the variables, including
    // the vartype, bounds, number etc.
    QuadraticModel<bias_type, index_type> objective_;

    std::vector<constraint_type> constraints_;

    // should this be public?
    class ConstraintsView {
        ConstrainedQuadraticModel<bias_type, index_type>* parent_;

     public:
        explicit ConstraintsView(ConstrainedQuadraticModel<bias_type, index_type>* parent)
                : parent_(parent) {}
        const constraint_type& operator[](index_type c) const { return parent_->constraints_[c]; }
        constraint_type& operator[](index_type c) { return parent_->constraints_[c]; }

        typename std::vector<constraint_type>::iterator begin() {
            return parent_->constraints_.begin();
        }
        typename std::vector<constraint_type>::iterator end() {
            return parent_->constraints_.end();
        }
        size_type size() const { return parent_->constraints_.size(); }
    };

 public:
    ConstraintsView constraints;

    ConstrainedQuadraticModel() : constraints(this) {}

    /// Copy constructor.
    ConstrainedQuadraticModel(const ConstrainedQuadraticModel& cqm)
            : objective_(cqm.objective_), constraints(this) {
        // for the constraints, we copy them one-by-one so we can make sure
        // their pointers are pointing to the right place
        for (const auto& c : cqm.constraints_) {
            this->constraints_.emplace_back(this, c);
        }
    }

    // ~ConstrainedQuadraticModel() = default;

    void add_constraints(index_type n, Sense sense) {
        for (index_type i = 0; i < n; ++i) {
            this->constraints_.emplace_back(this, sense);
        }
    }

    index_type add_linear_constraint(std::vector<index_type>&& variables,
                                     std::vector<bias_type>&& biases, Sense sense, bias_type rhs) {
        this->constraints_.emplace_back(this, std::move(variables), std::move(biases), sense, rhs);
        return this->constraints_.size() - 1;
    }

    template <class B, class I>
    index_type add_linear_constraint(const std::vector<I>& variables, const std::vector<B>& biases,
                                     Sense sense, bias_type rhs) {
        // make some vectors we can move
        std::vector<index_type> v(variables.begin(), variables.end());
        std::vector<bias_type> b(biases.begin(), biases.end());
        return this->add_linear_constraint(std::move(v), std::move(b), sense, rhs);
    }

    index_type add_variable(Vartype vartype) { return this->objective_.add_variable(vartype); }

    index_type add_variable(Vartype vartype, bias_type lb, bias_type ub) {
        return this->objective_.add_variable(vartype, lb, ub);
    }

    // todo: swap order, return type
    void add_variables(index_type n, Vartype vartype) {
        for (index_type i = 0; i < n; ++i) {
            this->objective_.add_variable(vartype);
        }
    }

    void add_variables(index_type n, Vartype vartype, bias_type lb, bias_type ub) {
        for (index_type i = 0; i < n; ++i) {
            this->objective_.add_variable(vartype, lb, ub);
        }
    }

    // constraint_type& constraint(index_type c) { return this->constraints_[c]; }

    // const constraint_type& constraint(index_type c) const { return this->constraints_[c]; }

    template <class T>
    void fix_variable(index_type v, T assignment) {
        for (auto& c : this->constraints_) {
            c.fix_variable(v, assignment);
        }
        this->objective_.fix_variable(v, assignment);
    }

    /// Return the lower bound for variable `v`.
    bias_type lower_bound(index_type v) const { return this->objective_.lower_bound(v); }

    size_type num_constraints() const { return this->constraints_.size(); }

    /// Return the number of variables in the model.
    size_type num_variables() const { return this->objective_.num_variables(); }

    const QuadraticModel<bias_type, index_type>& objective() const { return this->objective_; }

    /// Return the upper bound for variable `v`.
    bias_type upper_bound(index_type v) const { return this->objective_.upper_bound(v); }

    void remove_constraint(index_type i) {
        this->constraints_.erase(this->constraints_.begin() + i);
    }

    void set_lower_bound(index_type v, bias_type bound) {
        this->objective_.set_lower_bound(v, bound);
    }
    void set_upper_bound(index_type v, bias_type bound) {
        this->objective_.set_upper_bound(v, bound);
    }

    template <class B, class I>
    void set_objective(const abc::QuadraticModelBase<B, I>& objective) {
        if (!this->num_variables()) {
            // the objective is empty, so we can just add, easy peasy
            for (size_type i = 0; i < objective.num_variables(); ++i) {
                this->objective_.add_variable(objective.vartype(i), objective.lower_bound(i),
                                              objective.upper_bound(i));
                this->objective_.set_linear(i, objective.linear(i));
            }

            for (auto qit = objective.cbegin_quadratic(); qit != objective.cend_quadratic();
                 ++qit) {
                this->objective_.add_quadratic_back(qit->u, qit->v, qit->bias);
            }

            this->objective_.set_offset(objective.offset());

            return;
        }

        throw std::logic_error("not implemented");
    }

    /// Return the vartype of `v`.
    Vartype vartype(index_type v) const { return this->objective_.vartype(v); }
};

}  // namespace dimod
