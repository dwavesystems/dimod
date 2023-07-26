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

#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dimod/abc.h"
#include "dimod/constraint.h"
#include "dimod/iterators.h"
#include "dimod/expression.h"
#include "dimod/vartypes.h"

namespace dimod {

/// A constrained quadratic model (CQM) can include an objective and constraints as polynomials with one or two variables per term.
template <class Bias, class Index = int>
class ConstrainedQuadraticModel;

template <class CQM>
class ConstraintsView {
 public:
    using bias_type = typename CQM::bias_type;
    using index_type = typename CQM::index_type;

    using constraint_type = typename std::conditional<std::is_const<CQM>::value,
                                                      const Constraint<bias_type, index_type>,
                                                      Constraint<bias_type, index_type>>::type;

    using iterator = iterators::ConstraintIterator<
            typename std::vector<std::shared_ptr<Constraint<bias_type, index_type>>>::iterator>;
    using const_iterator = iterators::ConstraintIterator<
            typename std::vector<
                    std::shared_ptr<Constraint<bias_type, index_type>>>::const_iterator,
            const Constraint<bias_type, index_type>>;

    explicit ConstraintsView(CQM* parent) : parent_(parent) {}

    constraint_type& operator[](index_type c) { return *(parent_->constraints_[c]); }
    const constraint_type& operator[](index_type c) const { return *(parent_->constraints_[c]); }

    constraint_type& at(index_type c) { return *(parent_->constraints_.at(c)); }
    const constraint_type& at(index_type c) const { return *(parent_->constraints_.at(c)); }

    iterator begin() { return iterators::make_constraint_iterator(parent_->constraints_.begin()); }
    const_iterator begin() const {
        return iterators::make_const_constraint_iterator(parent_->constraints_.cbegin());
    }
    iterator end() { return iterators::make_constraint_iterator(parent_->constraints_.end()); }
    const_iterator end() const {
        return iterators::make_const_constraint_iterator(parent_->constraints_.cend());
    }

    const_iterator cbegin() const {
        return iterators::make_const_constraint_iterator(parent_->constraints_.cbegin());
    }
    const_iterator cend() const {
        return iterators::make_const_constraint_iterator(parent_->constraints_.cend());
    }

    std::size_t size() const { return parent_->constraints_.size(); }

 private:
    CQM* const parent_;
};

template <class Bias, class Index>
class ConstrainedQuadraticModel {
 public:
    /// First template parameter (`Bias`).
    using bias_type = Bias;

    /// Second template parameter (`Index`).
    using index_type = Index;

    /// Unsigned integer type that can represent non-negative values.
    using size_type = std::size_t;

    friend class ConstraintsView<ConstrainedQuadraticModel<bias_type, index_type>>;
    friend class ConstraintsView<const ConstrainedQuadraticModel<bias_type, index_type>>;

    ConstrainedQuadraticModel();

    ConstrainedQuadraticModel(const ConstrainedQuadraticModel& other);
    ConstrainedQuadraticModel(ConstrainedQuadraticModel&& other) noexcept;
    ~ConstrainedQuadraticModel() = default;
    ConstrainedQuadraticModel& operator=(ConstrainedQuadraticModel other);

    index_type add_constraint();

    template <class B, class I, class T>
    index_type add_constraint(const abc::QuadraticModelBase<B, I>& lhs, Sense sense, bias_type rhs,
                              const std::vector<T>& mapping);

    index_type add_constraint(abc::QuadraticModelBase<bias_type, index_type>&& lhs, Sense sense, bias_type rhs,
                              std::vector<index_type> mapping);

    /// Add a constraint.
    /// @param constraint As created by ConstrainedQuadraticModel::new_constraint()
    /// @exception Throws std::logic_error If the constraint's parent is not this model.
    /// If an exception is thrown, there are no changes to the model.
    index_type add_constraint(Constraint<bias_type, index_type> constraint);

    /// Add `n` constraints.
    index_type add_constraints(index_type n);

    /// Add a constraint with only linear coefficients.
    index_type add_linear_constraint(std::initializer_list<index_type> variables,
                                     std::initializer_list<bias_type> biases, Sense sense,
                                     bias_type rhs);

    /// Add variable of type `vartype` with lower bound `lb` and upper bound `ub`.
    index_type add_variable(Vartype vartype, bias_type lb, bias_type ub);

    /// Add variable of type `vartype`.
    index_type add_variable(Vartype vartype);

    /// Add `n` variables of type `vartype`.
    index_type add_variables(Vartype vartype, index_type n);

    /// Add `n` variables of type `vartype` with lower bound `lb` and upper bound `ub`.
    index_type add_variables(Vartype vartype, index_type n, bias_type lb, bias_type ub);

    /// Change the variable type of variable `v` to `vartype`, updating the biases appropriately.
    void change_vartype(Vartype vartype, index_type v);

    void clear();

    /// Return a view over the constraints. The view can be iterated over.
    /// @code
    /// for (auto& constraint : cqm.constraints()) {}
    /// @endcode
    ConstraintsView<ConstrainedQuadraticModel<bias_type, index_type>> constraints();

    /// @copydoc ConstrainedQuadraticModel::constraints()
    const ConstraintsView<const ConstrainedQuadraticModel<bias_type, index_type>> constraints() const;

    Constraint<bias_type, index_type>& constraint_ref(index_type c);
    const Constraint<bias_type, index_type>& constraint_ref(index_type c) const;

    std::weak_ptr<Constraint<bias_type, index_type>> constraint_weak_ptr(index_type c);
    std::weak_ptr<const Constraint<bias_type, index_type>> constraint_weak_ptr(index_type c) const;

    /// Fix variable `v` in the model to value `assignment`.
    template <class T>
    void fix_variable(index_type v, T assignment);

    /// Create a new model by fixing many variables.
    template <class VarIter, class AssignmentIter>
    ConstrainedQuadraticModel fix_variables(VarIter first, VarIter last,
                                            AssignmentIter assignment) const;

    /// Create a new model by fixing many variables.
    template <class T>
    ConstrainedQuadraticModel fix_variables(std::initializer_list<index_type> variables,
                                            std::initializer_list<T> assignments) const;

    /// Return the lower bound on variable ``v``.
    bias_type lower_bound(index_type v) const;

    /// Return a new constraint without adding it to the model.
    Constraint<bias_type, index_type> new_constraint() const;

    /// Return the number of constraints in the model.
    size_type num_constraints() const;

    /// Return the number of variables in the model.
    size_type num_variables() const;

    /// Remove a constraint from the model.
    void remove_constraint(index_type c);

    /// Remove all constraints satisfying a certain criteria as defined by
    /// a unary predicate p which accepts a constraint reference and
    /// returns true if the constraint should be removed.
    template<class UnaryPredicate>
    void remove_constraints_if(UnaryPredicate p);

    /// Remove variable `v` from the model.
    void remove_variable(index_type v);

    /// Set a lower bound of `lb` on variable `v`.
    void set_lower_bound(index_type v, bias_type lb);

    /// Set an objective for the model.
    template <class B, class I>
    void set_objective(const abc::QuadraticModelBase<B, I>& objective);

    /// Set an objective for the model using a mapping.
    template <class B, class I, class T>
    void set_objective(const abc::QuadraticModelBase<B, I>& objective,
                       const std::vector<T>& mapping);

    /// Set an upper bound of `ub` on variable `v`.
    void set_upper_bound(index_type v, bias_type ub);
    void set_vartype(index_type v, Vartype vartype);

    void substitute_variable(index_type v, bias_type multiplier, bias_type offset);

    /// Return the upper bound on variable ``v``.
    bias_type upper_bound(index_type v) const;

    /// Return the variable type of variable ``v``.
    Vartype vartype(index_type v) const;

    friend void swap(ConstrainedQuadraticModel& first, ConstrainedQuadraticModel& second) {
        first.myswap(second);
    }

    Expression<bias_type, index_type> objective;

 private:
    // I can't figure out how to get Expression to grant friendship to friend swap...
    // Will look into it some other time.
    void myswap(ConstrainedQuadraticModel& other) {
        using std::swap;

        // for objective and constraints, we need to make sure that their
        // parent_ pointers are pointing at the correct object.

        swap(this->objective, other.objective);
        this->objective.parent_ = this;
        other.objective.parent_ = &other;

        swap(this->constraints_, other.constraints_);
        for (auto& c_ptr : this->constraints_) {
            c_ptr->parent_ = this;
        }
        for (auto& c_ptr : other.constraints_) {
            c_ptr->parent_ = &other;
        }

        swap(this->varinfo_, other.varinfo_);
    }

    static void fix_variables_expr(const Expression<bias_type, index_type>& src,
                                   Expression<bias_type, index_type>& dst,
                                   const std::vector<index_type>& old_to_new,
                                   const std::vector<bias_type>& assignments);

    std::vector<std::shared_ptr<Constraint<bias_type, index_type>>> constraints_;

    struct varinfo_type {
        Vartype vartype;
        bias_type lb;
        bias_type ub;

        varinfo_type(Vartype vartype, bias_type lb, bias_type ub)
                : vartype(vartype), lb(lb), ub(ub) {
            assert(lb <= ub);

            assert(lb >= vartype_info<bias_type>::min(vartype));
            assert(ub <= vartype_info<bias_type>::max(vartype));

            assert(vartype != Vartype::BINARY || lb == 0);
            assert(vartype != Vartype::BINARY || ub == 1);

            assert(vartype != Vartype::SPIN || lb == -1);
            assert(vartype != Vartype::SPIN || ub == +1);
        }

        explicit varinfo_type(Vartype vartype)
                : vartype(vartype),
                  lb(vartype_info<bias_type>::default_min(vartype)),
                  ub(vartype_info<bias_type>::default_max(vartype)) {}
    };

    std::vector<varinfo_type> varinfo_;
};

template <class bias_type, class index_type>
ConstrainedQuadraticModel<bias_type, index_type>::ConstrainedQuadraticModel()
        : objective(this), constraints_(), varinfo_() {}

template <class bias_type, class index_type>
ConstrainedQuadraticModel<bias_type, index_type>::ConstrainedQuadraticModel(
        const ConstrainedQuadraticModel& other)
        : objective(other.objective), constraints_(), varinfo_(other.varinfo_) {
    objective.parent_ = this;

    for (auto& c_ptr : other.constraints_) {
        constraints_.push_back(std::make_shared<Constraint<bias_type, index_type>>(*c_ptr));
        constraints_.back()->parent_ = this;
    }
}

template <class bias_type, class index_type>
ConstrainedQuadraticModel<bias_type, index_type>::ConstrainedQuadraticModel(
        ConstrainedQuadraticModel&& other) noexcept
        : ConstrainedQuadraticModel() {
    swap(*this, other);
}

template <class bias_type, class index_type>
ConstrainedQuadraticModel<bias_type, index_type>&
ConstrainedQuadraticModel<bias_type, index_type>::operator=(ConstrainedQuadraticModel other) {
    swap(*this, other);
    return *this;
}

template <class bias_type, class index_type>
index_type ConstrainedQuadraticModel<bias_type, index_type>::add_constraint() {
    constraints_.push_back(std::make_shared<Constraint<bias_type, index_type>>(this));
    return constraints_.size() - 1;
}

template <class bias_type, class index_type>
index_type ConstrainedQuadraticModel<bias_type, index_type>::add_constraint(
        Constraint<bias_type, index_type> constraint) {
    if (constraint.parent_ != this) {
        throw std::logic_error("given constraint has a different parent");
    }
    constraints_.push_back(
            std::make_shared<Constraint<bias_type, index_type>>(std::move(constraint)));
    return constraints_.size() - 1;
}

template <class bias_type, class index_type>
template <class B, class I, class T>
index_type ConstrainedQuadraticModel<bias_type, index_type>::add_constraint(
        const abc::QuadraticModelBase<B, I>& lhs, Sense sense, bias_type rhs,
        const std::vector<T>& mapping) {
    assert(mapping.size() == lhs.num_variables());

    auto constraint = new_constraint();

    for (size_type i = 0; i < lhs.num_variables(); ++i) {
        assert(mapping[i] >= 0 && static_cast<size_type>(mapping[i]) < num_variables());
        assert(vartype(mapping[i]) == lhs.vartype(i));
        assert(lower_bound(mapping[i]) == lhs.lower_bound(i));
        assert(upper_bound(mapping[i]) == lhs.upper_bound(i));

        constraint.add_linear(mapping[i], lhs.linear(i));
    }

    // quadratic biases
    for (auto it = lhs.cbegin_quadratic(); it != lhs.cend_quadratic(); ++it) {
        constraint.add_quadratic(mapping[it->u], mapping[it->v], it->bias);
    }

    // offset
    constraint.add_offset(lhs.offset());

    // sense and rhs
    constraint.set_sense(sense);
    constraint.set_rhs(rhs);

    return add_constraint(std::move(constraint));
}

template <class bias_type, class index_type>
index_type ConstrainedQuadraticModel<bias_type, index_type>::add_constraint(
        abc::QuadraticModelBase<bias_type, index_type>&& lhs, Sense sense, bias_type rhs,
        std::vector<index_type> mapping) {
    assert(mapping.size() == lhs.num_variables());

    Constraint<bias_type, index_type> constraint = new_constraint();

    // move the underlying biases/offset
    static_cast<abc::QuadraticModelBase<bias_type, index_type>&>(constraint) = std::move(lhs);

    // and set the labels based on the mapping
    constraint.relabel_variables(std::move(mapping));

    // sense and rhs
    constraint.set_sense(sense);
    constraint.set_rhs(rhs);

    return add_constraint(std::move(constraint));
}

template <class bias_type, class index_type>
index_type ConstrainedQuadraticModel<bias_type, index_type>::add_constraints(index_type n) {
    assert(n >= 0);
    index_type size = constraints_.size();
    for (index_type i = 0; i < n; ++i) {
        constraints_.push_back(std::make_shared<Constraint<bias_type, index_type>>(this));
    }
    return size;
}

template <class bias_type, class index_type>
index_type ConstrainedQuadraticModel<bias_type, index_type>::add_linear_constraint(
        std::initializer_list<index_type> variables, std::initializer_list<bias_type> biases,
        Sense sense, bias_type rhs) {
    Constraint<bias_type, index_type> constraint(this);

    auto vit = variables.begin();
    auto bit = biases.begin();
    for (; vit != variables.end() && bit != biases.end(); ++vit, ++bit) {
        constraint.add_linear(*vit, *bit);
    }

    constraint.set_rhs(rhs);
    constraint.set_sense(sense);

    constraints_.push_back(
            std::make_shared<Constraint<bias_type, index_type>>(std::move(constraint)));
    return constraints_.size() - 1;
}

template <class bias_type, class index_type>
index_type ConstrainedQuadraticModel<bias_type, index_type>::add_variable(Vartype vartype) {
    index_type v = num_variables();
    varinfo_.emplace_back(vartype);
    return v;
}

template <class bias_type, class index_type>
index_type ConstrainedQuadraticModel<bias_type, index_type>::add_variable(Vartype vartype,
                                                                          bias_type lb,
                                                                          bias_type ub) {
    index_type v = num_variables();
    varinfo_.emplace_back(vartype, lb, ub);  // also handles asserts
    return v;
}

template <class bias_type, class index_type>
index_type ConstrainedQuadraticModel<bias_type, index_type>::add_variables(Vartype vartype,
                                                                           index_type n) {
    index_type start = num_variables();
    varinfo_.insert(varinfo_.end(), n, varinfo_type(vartype));
    return start;
}

template <class bias_type, class index_type>
index_type ConstrainedQuadraticModel<bias_type, index_type>::add_variables(Vartype vartype,
                                                                           index_type n,
                                                                           bias_type lb,
                                                                           bias_type ub) {
    index_type start = num_variables();
    varinfo_.insert(varinfo_.end(), n, varinfo_type(vartype, lb, ub));
    return start;
}

template <class bias_type, class index_type>
void ConstrainedQuadraticModel<bias_type, index_type>::change_vartype(Vartype vartype,
                                                                      index_type v) {
    const Vartype& source = this->vartype(v);
    const Vartype& target = vartype;

    // todo: just call this->substitute_variable

    if (source == target) {
        return;
    } else if (source == Vartype::SPIN && target == Vartype::BINARY) {
        objective.substitute_variable(v, 2, -1);
        for (auto& c_ptr : constraints_) {
            c_ptr->substitute_variable(v, 2, -1);
        }
        varinfo_[v].lb = 0;
        varinfo_[v].ub = 1;
        varinfo_[v].vartype = Vartype::BINARY;
    } else if (source == Vartype::BINARY && target == Vartype::SPIN) {
        objective.substitute_variable(v, .5, .5);
        for (auto& c_ptr : constraints_) {
            c_ptr->substitute_variable(v, .5, .5);
        }
        varinfo_[v].lb = -1;
        varinfo_[v].ub = +1;
        varinfo_[v].vartype = Vartype::SPIN;
    } else if (source == Vartype::SPIN && target == Vartype::INTEGER) {
        // first go to BINARY, then INTEGER
        change_vartype(Vartype::BINARY, v);
        change_vartype(Vartype::INTEGER, v);
    } else if (source == Vartype::BINARY && target == Vartype::INTEGER) {
        // nothing need to change except the vartype itself
        varinfo_[v].vartype = Vartype::INTEGER;
    } else {
        // todo: there are more we could support
        throw std::logic_error("unsupported vartype change");
    }
}

template <class bias_type, class index_type>
void ConstrainedQuadraticModel<bias_type, index_type>::clear() {
    objective.clear();
    constraints_.clear();
    varinfo_.clear();
}

template <class bias_type, class index_type>
ConstraintsView<ConstrainedQuadraticModel<bias_type, index_type>>
ConstrainedQuadraticModel<bias_type, index_type>::constraints() {
    return ConstraintsView<ConstrainedQuadraticModel<bias_type, index_type>>(this);
}

template <class bias_type, class index_type>
const ConstraintsView<const ConstrainedQuadraticModel<bias_type, index_type>>
ConstrainedQuadraticModel<bias_type, index_type>::constraints() const {
    return ConstraintsView<const ConstrainedQuadraticModel<bias_type, index_type>>(this);
}

template <class bias_type, class index_type>
Constraint<bias_type, index_type>& ConstrainedQuadraticModel<bias_type, index_type>::constraint_ref(
        index_type c) {
    assert(c >= 0 && static_cast<size_type>(c) < num_constraints());
    return *constraints_[c];
}

template <class bias_type, class index_type>
const Constraint<bias_type, index_type>&
ConstrainedQuadraticModel<bias_type, index_type>::constraint_ref(index_type c) const {
    assert(c >= 0 && static_cast<size_type>(c) < num_constraints());
    return *constraints_[c];
}

template <class bias_type, class index_type>
std::weak_ptr<Constraint<bias_type, index_type>>
ConstrainedQuadraticModel<bias_type, index_type>::constraint_weak_ptr(index_type c) {
    assert(c >= 0 && static_cast<size_type>(c) < num_constraints());
    return constraints_[c];
}

template <class bias_type, class index_type>
std::weak_ptr<const Constraint<bias_type, index_type>>
ConstrainedQuadraticModel<bias_type, index_type>::constraint_weak_ptr(index_type c) const {
    assert(c >= 0 && static_cast<size_type>(c) < num_constraints());
    return constraints_[c];
}

template <class bias_type, class index_type>
template <class T>
void ConstrainedQuadraticModel<bias_type, index_type>::fix_variable(index_type v, T assignment) {
    assert(v >= 0 && static_cast<size_type>(v) < num_variables());
    substitute_variable(v, 0, assignment);
    remove_variable(v);
}

template <class bias_type, class index_type>
void ConstrainedQuadraticModel<bias_type, index_type>::fix_variables_expr(
        const Expression<bias_type, index_type>& src, Expression<bias_type, index_type>& dst,
        const std::vector<index_type>& old_to_new, const std::vector<bias_type>& assignments) {
    // We'll want to access the source expression by index for speed
    const abc::QuadraticModelBase<bias_type, index_type>& isrc = src;

    // offset
    dst.add_offset(src.offset());

    // linear biases and variables
    for (size_type i = 0; i < src.num_variables(); ++i) {
        auto v = src.variables()[i];

        if (old_to_new[v] < 0) {
            // fixed
            dst.add_offset(isrc.linear(i) * assignments[v]);
        } else {
            // not fixed
            dst.add_linear(old_to_new[v], isrc.linear(i));
        }
    }

    // quadratic
    for (auto it = isrc.cbegin_quadratic(), end = isrc.cend_quadratic(); it != end; ++it) {
        const index_type u = src.variables()[it->u];  // variable u in the source
        const index_type v = src.variables()[it->v];  // variable v in the source
        const bias_type bias = it->bias;  // bias in the source

        const index_type new_u = old_to_new[u];  // variable u in the destination
        const index_type new_v = old_to_new[v];  // variable v in the destination

        if (new_u < 0 && new_v < 0) {
            // both fixed, becomes offset
            dst.add_offset(assignments[u] * assignments[v] * bias);
        } else if (new_u < 0) {
            // u fixed, v unfixed
            dst.add_linear(new_v, assignments[u] * bias);
        } else if (new_v < 0) {
            // u unfixed, v fixed
            dst.add_linear(new_u, assignments[v] * bias);
        } else {
            // neither fixed
            dst.add_quadratic_back(new_u, new_v, bias);
        }
    }
}

template <class bias_type, class index_type>
template <class VarIter, class AssignmentIter>
ConstrainedQuadraticModel<bias_type, index_type>
ConstrainedQuadraticModel<bias_type, index_type>::fix_variables(VarIter first, VarIter last,
                                                                AssignmentIter assignment) const {
    // We're going to make a new CQM
    auto cqm = ConstrainedQuadraticModel<bias_type, index_type>();

    // Map from the old indices to the new ones. We'll use -1 to indicate a fixed
    // variable. We could use an unordered map or similar, but this ends up being
    // faster
    std::vector<index_type> old_to_new(this->num_variables());

    // The fixed variable assignments, by old indices
    // We don't actually need this to be full of 0s, but this is simple.
    // We could inherit the type from AssignmentIter, but again this seems simpler.
    std::vector<bias_type> assignments(this->num_variables());

    // Fill in our various data vectors and add the variables to the new model
    for (auto it = first; it != last; ++it, ++assignment) {
        old_to_new[*it] = -1;
        assignments[*it] = *assignment;
    }
    for (size_type i = 0; i < old_to_new.size(); ++i) {
        if (old_to_new[i] < 0) continue;  // fixed

        old_to_new[i] =
                cqm.add_variable(this->vartype(i), this->lower_bound(i), this->upper_bound(i));
    }

    // Objective
    fix_variables_expr(this->objective, cqm.objective, old_to_new, assignments);

    // Constraints
    for (auto& old_constraint_ptr : constraints_) {
        auto new_constraint = cqm.new_constraint();

        fix_variables_expr(*old_constraint_ptr, new_constraint, old_to_new, assignments);

        // dev note: this is kind of a maintenance mess. If we find ourselves doing this again
        // we should make a method for copying attributes etc
        new_constraint.set_rhs(old_constraint_ptr->rhs());
        new_constraint.set_sense(old_constraint_ptr->sense());
        new_constraint.set_weight(old_constraint_ptr->weight());
        new_constraint.set_penalty(old_constraint_ptr->penalty());
        new_constraint.mark_discrete(old_constraint_ptr->marked_discrete() &&
                                     new_constraint.is_onehot());

        cqm.add_constraint(std::move(new_constraint));
    }

    return cqm;
}

template <class bias_type, class index_type>
template <class T>
ConstrainedQuadraticModel<bias_type, index_type>
ConstrainedQuadraticModel<bias_type, index_type>::fix_variables(
        std::initializer_list<index_type> variables, std::initializer_list<T> assignments) const {
    return fix_variables(variables.begin(), variables.end(), assignments.begin());
}

template <class bias_type, class index_type>
bias_type ConstrainedQuadraticModel<bias_type, index_type>::lower_bound(index_type v) const {
    return varinfo_[v].lb;
}

template <class bias_type, class index_type>
Constraint<bias_type, index_type> ConstrainedQuadraticModel<bias_type, index_type>::new_constraint()
        const {
    return Constraint<bias_type, index_type>(this);
}

template <class bias_type, class index_type>
std::size_t ConstrainedQuadraticModel<bias_type, index_type>::num_constraints() const {
    return constraints_.size();
}

template <class bias_type, class index_type>
std::size_t ConstrainedQuadraticModel<bias_type, index_type>::num_variables() const {
    return varinfo_.size();
}

template <class bias_type, class index_type>
void ConstrainedQuadraticModel<bias_type, index_type>::remove_constraint(index_type c) {
    constraints_.erase(constraints_.begin() + c, constraints_.begin() + c + 1);
}

template <class bias_type, class index_type>
template <class UnaryPredicate>
void ConstrainedQuadraticModel<bias_type, index_type>::remove_constraints_if(UnaryPredicate p) {
    // create a new predicate that acts on a shared_ptr rather than the constraint
    auto pred = [&](std::shared_ptr<Constraint<bias_type, index_type>>& constraint_ptr) {
        return p(*constraint_ptr);
    };

    constraints_.erase(std::remove_if(constraints_.begin(), constraints_.end(), pred),
                       constraints_.end());
}

template <class bias_type, class index_type>
void ConstrainedQuadraticModel<bias_type, index_type>::remove_variable(index_type v) {
    assert(v >= 0 && static_cast<size_type>(v) < num_variables());
    for (auto& c_ptr : constraints_) c_ptr->reindex_variables(v);
    objective.reindex_variables(v);
    varinfo_.erase(varinfo_.begin() + v);
}

template <class bias_type, class index_type>
void ConstrainedQuadraticModel<bias_type, index_type>::set_lower_bound(index_type v, bias_type lb) {
    varinfo_[v].lb = lb;
}

template <class bias_type, class index_type>
template <class B, class I>
void ConstrainedQuadraticModel<bias_type, index_type>::set_objective(
        const abc::QuadraticModelBase<B, I>& objective) {
    // todo: move version
    // todo: support sparse mapping

    // add missing variables
    for (size_type v = num_variables(); v < objective.num_variables(); ++v) {
        add_variable(objective.vartype(v), objective.lower_bound(v), objective.upper_bound(v));
    }

    this->objective.clear();

    // linear biases
    for (size_type v = 0; v < objective.num_variables(); ++v) {
        // this checks the overlapping variables. todo: do this in a NDEBUG ifdef?
        assert(vartype(v) == objective.vartype(v));
        assert(lower_bound(v) == objective.lower_bound(v));
        assert(upper_bound(v) == objective.upper_bound(v));

        this->objective.add_linear(v, objective.linear(v));
    }

    // quadratic biases
    for (auto it = objective.cbegin_quadratic(); it != objective.cend_quadratic(); ++it) {
        this->objective.add_quadratic(it->u, it->v, it->bias);
    }

    // offset
    this->objective.add_offset(objective.offset());
}

// variables must all be present in this case!
template <class bias_type, class index_type>
template <class B, class I, class T>
void ConstrainedQuadraticModel<bias_type, index_type>::set_objective(
        const abc::QuadraticModelBase<B, I>& objective, const std::vector<T>& mapping) {
    // there are a tonne of potential optimizations here, but let's keep it simple for now
    assert(mapping.size() == objective.num_variables());

    this->objective.clear();

    for (size_type i = 0; i < objective.num_variables(); ++i) {
        assert(vartype(mapping[i]) == objective.vartype(i));
        assert(lower_bound(mapping[i]) == objective.lower_bound(i));
        assert(upper_bound(mapping[i]) == objective.upper_bound(i));

        this->objective.add_linear(mapping[i], objective.linear(i));
    }

    // quadratic biases
    for (auto it = objective.cbegin_quadratic(); it != objective.cend_quadratic(); ++it) {
        this->objective.add_quadratic(mapping[it->u], mapping[it->v], it->bias);
    }

    // offset
    this->objective.add_offset(objective.offset());
}

template <class bias_type, class index_type>
void ConstrainedQuadraticModel<bias_type, index_type>::set_upper_bound(index_type v, bias_type ub) {
    varinfo_[v].ub = ub;
}

template <class bias_type, class index_type>
void ConstrainedQuadraticModel<bias_type, index_type>::set_vartype(index_type v, Vartype vartype) {
    varinfo_[v].vartype = vartype;
}

template <class bias_type, class index_type>
void ConstrainedQuadraticModel<bias_type, index_type>::substitute_variable(index_type v,
                                                                           bias_type multiplier,
                                                                           bias_type offset) {
    objective.substitute_variable(v, multiplier, offset);
    for (auto& c_ptr : constraints_) {
        c_ptr->substitute_variable(v, multiplier, offset);
    }
}

template <class bias_type, class index_type>
bias_type ConstrainedQuadraticModel<bias_type, index_type>::upper_bound(index_type v) const {
    return varinfo_[v].ub;
}

template <class bias_type, class index_type>
Vartype ConstrainedQuadraticModel<bias_type, index_type>::vartype(index_type v) const {
    return varinfo_[v].vartype;
}

}  // namespace dimod
