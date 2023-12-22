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

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dimod/abc.h"
#include "dimod/binary_quadratic_model.h"
#include "dimod/vartypes.h"

namespace dimod {

/// A quadratic model (QM) is a polynomial with one or two variables per term.
template <class Bias, class Index = int>
class QuadraticModel : public abc::QuadraticModelBase<Bias, Index> {
 public:
    /// The type of the base class.
    using base_type = abc::QuadraticModelBase<Bias, Index>;

    /// The first template parameter (`Bias`).
    using bias_type = Bias;

    /// The second template parameter (`Index`).
    using index_type = Index;

    /// Unsigned integer that can represent non-negative values.
    using size_type = typename base_type::size_type;

    QuadraticModel();

    explicit QuadraticModel(const BinaryQuadraticModel<bias_type, index_type>& bqm);

    template <class B, class I>
    explicit QuadraticModel(const BinaryQuadraticModel<B, I>& bqm);

    /// Add variable of type `vartype`.
    index_type add_variable(Vartype vartype);

    /// Add `n` variables of type `vartype` with lower bound `lb` and upper bound `ub`.
    index_type add_variable(Vartype vartype, bias_type lb, bias_type ub);

    /// Add `n` variables of type `vartype`.
    index_type add_variables(Vartype vartype, index_type n);

    /// Add `n` variables of type `vartype` with lower bound `lb` and upper bound `ub`.
    index_type add_variables(Vartype vartype, index_type n, bias_type lb, bias_type ub);

    void clear();

    /// Change the vartype of `v`, updating the biases appropriately.
    void change_vartype(Vartype vartype, index_type v);

    /**
     * Remove variable `v` from the model by fixing its value.
     *
     * Note that this causes a reindexing, where all variables above `v` have
     * their index reduced by one.
     */
    template <class T>
    void fix_variable(index_type v, T assignment);

    /// Return the lower bound on variable ``v``.
    bias_type lower_bound(index_type v) const;

    /**
     * Total bytes consumed by the biases, `vartype` info, bounds, and indices.
     *
     * If `capacity` is true, use the capacity of the underlying vectors rather
     * than the size.
     */
    size_type nbytes(bool capacity = false) const;

    /// Remove variable `v`.
    void remove_variable(index_type v);

    /// Remove variables.
    void remove_variables(const std::vector<index_type>& variables);

    // Resize the model to contain `n` variables.
    void resize(index_type n);

    /**
     * Resize the model to contain `n` variables.
     *
     * Any added variables are of type `vartype` (value must be `Vartype::BINARY` or `Vartype::SPIN`)
     */
    void resize(index_type n, Vartype vartype);

    /**
     * Resize the model to contain `n` variables.
     *
     * Any added variables are of type `vartype` (value must be `Vartype::BINARY` or `Vartype::SPIN`)
     * and have lower bound `lb` and upper bound `ub`.
     */
    void resize(index_type n, Vartype vartype, bias_type lb, bias_type ub);

    /// Set a lower bound of `lb` on variable `v`.
    void set_lower_bound(index_type v, bias_type lb);

    /// Set an upper bound of `ub` on variable `v`.
    void set_upper_bound(index_type v, bias_type ub);

    /// Set the variable type of variable `v`.
    void set_vartype(index_type v, Vartype vartype);

    // todo: substitute_variable with vartype/bounds support

    /// Return the upper bound on variable ``v``.
    bias_type upper_bound(index_type v) const;

    /// Return the variable type of variable ``v``.
    Vartype vartype(index_type v) const;

 private:
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
QuadraticModel<bias_type, index_type>::QuadraticModel() : base_type(), varinfo_() {}

template <class bias_type, class index_type>
QuadraticModel<bias_type, index_type>::QuadraticModel(
        const BinaryQuadraticModel<bias_type, index_type>& bqm)
        : base_type(bqm), varinfo_(bqm.num_variables(), varinfo_type(bqm.vartype())) {}

template <class bias_type, class index_type>
template <class B, class I>
QuadraticModel<bias_type, index_type>::QuadraticModel(const BinaryQuadraticModel<B, I>& bqm) {
    // in the future we could speed this up
    this->resize(bqm.num_variables(), bqm.vartype(), bqm.lower_bound(), bqm.upper_bound());

    for (size_type v = 0; v < bqm.num_variables(); ++v) {
        this->set_linear(v, bqm.linear(v));
    }

    for (auto qit = bqm.cbegin_quadratic(); qit != bqm.cend_quadratic(); ++qit) {
        this->add_quadratic_back(qit->u, qit->v, qit->bias);
    }

    this->set_offset(bqm.offset());
}

template <class bias_type, class index_type>
index_type QuadraticModel<bias_type, index_type>::add_variable(Vartype vartype) {
    varinfo_.emplace_back(vartype);
    return base_type::add_variable();
}

template <class bias_type, class index_type>
index_type QuadraticModel<bias_type, index_type>::add_variable(Vartype vartype, bias_type lb,
                                                               bias_type ub) {
    varinfo_.emplace_back(vartype, lb, ub);  // also handles asserts
    return base_type::add_variable();
}

template <class bias_type, class index_type>
index_type QuadraticModel<bias_type, index_type>::add_variables(Vartype vartype, index_type n) {
    varinfo_.insert(varinfo_.end(), n, varinfo_type(vartype));
    return base_type::add_variables(n);
}

template <class bias_type, class index_type>
index_type QuadraticModel<bias_type, index_type>::add_variables(Vartype vartype, index_type n,
                                                                bias_type lb, bias_type ub) {
    varinfo_.insert(varinfo_.end(), n, varinfo_type(vartype, lb, ub));
    return base_type::add_variables(n);
}

template <class bias_type, class index_type>
void QuadraticModel<bias_type, index_type>::clear() {
    varinfo_.clear();
    base_type::clear();
}

template <class bias_type, class index_type>
void QuadraticModel<bias_type, index_type>::change_vartype(Vartype vartype, index_type v) {
    const Vartype& source = this->vartype(v);
    const Vartype& target = vartype;

    if (source == target) {
        return;
    } else if (source == Vartype::SPIN && target == Vartype::BINARY) {
        base_type::substitute_variable(v, 2, -1);
        this->varinfo_[v].lb = 0;
        this->varinfo_[v].ub = 1;
        this->varinfo_[v].vartype = Vartype::BINARY;
    } else if (source == Vartype::BINARY && target == Vartype::SPIN) {
        base_type::substitute_variable(v, .5, .5);
        this->varinfo_[v].lb = -1;
        this->varinfo_[v].ub = +1;
        this->varinfo_[v].vartype = Vartype::SPIN;
    } else if (source == Vartype::SPIN && target == Vartype::INTEGER) {
        // first go to BINARY, then INTEGER
        this->change_vartype(Vartype::BINARY, v);
        this->change_vartype(Vartype::INTEGER, v);
    } else if (source == Vartype::BINARY && target == Vartype::INTEGER) {
        // nothing need to change except the vartype itself
        this->varinfo_[v].vartype = Vartype::INTEGER;
    } else {
        // todo: there are more we could support
        throw std::logic_error("unsupported vartype change");
    }
}

template <class bias_type, class index_type>
template <class T>
void QuadraticModel<bias_type, index_type>::fix_variable(index_type v, T assignment) {
    base_type::fix_variable(v, assignment);
    varinfo_.erase(varinfo_.begin() + v);
}

template <class bias_type, class index_type>
bias_type QuadraticModel<bias_type, index_type>::lower_bound(index_type v) const {
    // even though v is unused, we need this to conform the the QuadraticModelBase API
    return varinfo_[v].lb;
}

template <class bias_type, class index_type>
typename QuadraticModel<bias_type, index_type>::size_type
QuadraticModel<bias_type, index_type>::nbytes(bool capacity) const {
    size_type count = base_type::nbytes(capacity);
    if (capacity) {
        count += varinfo_.capacity() * sizeof(varinfo_type);
    } else {
        count += varinfo_.size() * sizeof(varinfo_type);
    }
    return count;
}

template <class bias_type, class index_type>
void QuadraticModel<bias_type, index_type>::remove_variable(index_type v) {
    base_type::remove_variable(v);
    varinfo_.erase(varinfo_.begin() + v);
}

template <class bias_type, class index_type>
void QuadraticModel<bias_type, index_type>::remove_variables(const std::vector<index_type>& variables) {
    if (!std::is_sorted(variables.begin(), variables.end())) {
        // create a copy and sort it
        std::vector<index_type> sorted_indices = variables;
        std::sort(sorted_indices.begin(), sorted_indices.end());
        QuadraticModel<bias_type, index_type>::remove_variables(sorted_indices);
        return;
    }
    base_type::remove_variables(variables);
    varinfo_.erase(utils::remove_by_index(varinfo_.begin(), varinfo_.end(), variables.begin(), variables.end()), varinfo_.end());
}

template <class bias_type, class index_type>
void QuadraticModel<bias_type, index_type>::resize(index_type n) {
    // we could do this as an assert, but let's be careful since
    // we're often calling this from python
    if (n > static_cast<index_type>(this->num_variables())) {
        throw std::logic_error(
                "n must be smaller than the number of variables when no "
                "`vartype` is specified");
    }
    // doesn't matter what vartype we specify since we're shrinking
    base_type::resize(n);
    varinfo_.erase(varinfo_.begin() + n, varinfo_.end());
}

template <class bias_type, class index_type>
void QuadraticModel<bias_type, index_type>::resize(index_type n, Vartype vartype) {
    base_type::resize(n);
    varinfo_.resize(n, varinfo_type(vartype));
}

template <class bias_type, class index_type>
void QuadraticModel<bias_type, index_type>::resize(index_type n, Vartype vartype, bias_type lb,
                                                   bias_type ub) {
    assert(n > 0);
    varinfo_.resize(n, varinfo_type(vartype, lb, ub));
    base_type::resize(n);
}

template <class bias_type, class index_type>
void QuadraticModel<bias_type, index_type>::set_lower_bound(index_type v, bias_type lb) {
    varinfo_[v].lb = lb;
}

template <class bias_type, class index_type>
void QuadraticModel<bias_type, index_type>::set_upper_bound(index_type v, bias_type ub) {
    varinfo_[v].ub = ub;
}

template <class bias_type, class index_type>
void QuadraticModel<bias_type, index_type>::set_vartype(index_type v, Vartype vartype) {
    varinfo_[v].vartype = vartype;
}

template <class bias_type, class index_type>
bias_type QuadraticModel<bias_type, index_type>::upper_bound(index_type v) const {
    return varinfo_[v].ub;
}

template <class bias_type, class index_type>
Vartype QuadraticModel<bias_type, index_type>::vartype(index_type v) const {
    return varinfo_[v].vartype;
}

}  // namespace dimod
