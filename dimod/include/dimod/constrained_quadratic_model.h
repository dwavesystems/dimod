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
#include "dimod/expression.h"
#include "dimod/vartypes.h"

namespace dimod {

template <class Bias, class Index = int>
class ConstrainedQuadraticModel {
 public:
    /// The first template parameter (Bias).
    using bias_type = Bias;

    /// The second template parameter (Index).
    using index_type = Index;

    /// Unsigned integral type that can represent non-negative values.
    using size_type = std::size_t;

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

    index_type add_constraints(index_type n);

    index_type add_linear_constraint(std::initializer_list<index_type> variables,
                                     std::initializer_list<bias_type> biases, Sense sense,
                                     bias_type rhs);

    index_type add_variable(Vartype vartype, bias_type lb, bias_type ub);

    index_type add_variable(Vartype vartype);

    index_type add_variables(Vartype vartype, index_type n);

    index_type add_variables(Vartype vartype, index_type n, bias_type lb, bias_type ub);

    void change_vartype(Vartype vartype, index_type v);

    void clear();

    Constraint<bias_type, index_type>& constraint_ref(index_type c);
    const Constraint<bias_type, index_type>& constraint_ref(index_type c) const;

    std::weak_ptr<Constraint<bias_type, index_type>> constraint_weak_ptr(index_type c);
    std::weak_ptr<const Constraint<bias_type, index_type>> constraint_weak_ptr(index_type c) const;

    template <class T>
    void fix_variable(index_type v, T assignment);

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

    void remove_variable(index_type v);

    void set_lower_bound(index_type v, bias_type lb);

    template <class B, class I>
    void set_objective(const abc::QuadraticModelBase<B, I>& objective);

    template <class B, class I, class T>
    void set_objective(const abc::QuadraticModelBase<B, I>& objective,
                       const std::vector<T>& mapping);

    void set_upper_bound(index_type v, bias_type ub);
    void set_vartype(index_type v, Vartype vartype);

    void substitute_variable(index_type v, bias_type multiplier, bias_type offset);

    /// Return the upper bound on variable ``v``.
    bias_type upper_bound(index_type v) const;

    /// Return the variable type of variable ``v``.
    Vartype vartype(index_type v) const;

    Expression<bias_type, index_type> objective;

    friend void swap(ConstrainedQuadraticModel& first, ConstrainedQuadraticModel& second) {
        first.myswap(second);
    }

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
