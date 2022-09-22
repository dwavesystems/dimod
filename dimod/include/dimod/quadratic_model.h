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

#include <stdexcept>
#include <utility>
#include <vector>

#include "dimod/abc.h"
#include "dimod/binary_quadratic_model.h"
#include "dimod/vartypes.h"

namespace dimod {

template <class Bias, class Index = int>
class QuadraticModel : public abc::QuadraticModelBase<Bias, Index> {
 public:
    /// The type of the base class.
    using base_type = abc::QuadraticModelBase<Bias, Index>;

    /// The first template parameter (Bias).
    using bias_type = Bias;

    /// The second template parameter (Index).
    using index_type = Index;

    /// Unsigned integral that can represent non-negative values.
    using size_type = typename base_type::size_type;

    QuadraticModel();

    QuadraticModel(const QuadraticModel& qm);

    QuadraticModel(const QuadraticModel&& qm);

    explicit QuadraticModel(const BinaryQuadraticModel<bias_type, index_type>& bqm);

    template <class B, class I>
    explicit QuadraticModel(const BinaryQuadraticModel<B, I>& bqm);

    QuadraticModel& operator=(const QuadraticModel& other);

    QuadraticModel& operator=(QuadraticModel&& other) noexcept;

    index_type add_variable(Vartype vartype);

    index_type add_variable(Vartype vartype, bias_type lb, bias_type ub);

    index_type add_variables(Vartype vartype, index_type n);

    index_type add_variables(Vartype vartype, index_type n, bias_type lb, bias_type ub);

    void clear();

    /// Change the vartype of `v`, updating the biases appropriately.
    void change_vartype(Vartype vartype, index_type v);

    template <class T>
    void fix_variable(index_type v, T assignment);

    bias_type lower_bound(index_type v) const;

    /**
     * Total bytes consumed by the biases, vartype info, bounds, and indices.
     *
     * If `capacity` is true, use the capacity of the underlying vectors rather
     * than the size.
     */
    size_type nbytes(bool capacity = false) const;

    void remove_variable(index_type v);

    // Resize the model to contain `n` variables.
    void resize(index_type n);

    /**
     * Resize the model to contain `n` variables.
     *
     * The `vartype` is used to any new variables added.
     *
     * The `vartype` must be `Vartype::BINARY` or `Vartype::SPIN`.
     */
    void resize(index_type n, Vartype vartype);

    /**
     * Resize the model to contain `n` variables.
     *
     * The `vartype` is used to any new variables added.
     */
    void resize(index_type n, Vartype vartype, bias_type lb, bias_type ub);

    void set_lower_bound(index_type v, bias_type lb);

    void set_upper_bound(index_type v, bias_type ub);

    void set_vartype(index_type v, Vartype vartype);

    bias_type upper_bound(index_type v) const;

    Vartype vartype(index_type v) const;

 private:
    struct varinfo_type {
        Vartype vartype;
        bias_type lb;
        bias_type ub;

        varinfo_type(Vartype vartype, bias_type lb, bias_type ub)
                : vartype(vartype), lb(lb), ub(ub) {}

        explicit varinfo_type(Vartype vartype) : vartype(vartype) {
            this->lb = vartype_info<bias_type>::default_min(vartype);
            this->ub = vartype_info<bias_type>::default_max(vartype);
        }
    };

    std::vector<varinfo_type> varinfo_;
};  // namespace dimod

template <class bias_type, class index_type>
QuadraticModel<bias_type, index_type>::QuadraticModel() : base_type(), varinfo_() {}

template <class bias_type, class index_type>
QuadraticModel<bias_type, index_type>::QuadraticModel(const QuadraticModel& qm)
        : base_type(qm), varinfo_(qm.varinfo_) {}

template <class bias_type, class index_type>
QuadraticModel<bias_type, index_type>::QuadraticModel(const QuadraticModel&& qm)
        : base_type(std::move(qm)), varinfo_(std::move(qm.varinfo_)) {}

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
QuadraticModel<bias_type, index_type>& QuadraticModel<bias_type, index_type>::operator=(
        const QuadraticModel& other) {
    base_type::operator=(other);
    this->varinfo_ = other.varinfo_;
    return *this;
}

template <class bias_type, class index_type>
QuadraticModel<bias_type, index_type>& QuadraticModel<bias_type, index_type>::operator=(
        QuadraticModel&& other) noexcept {
    using std::swap;
    base_type::operator=(std::move(other));
    this->varinfo_ = std::move(other.varinfo_);
    return *this;
}

template <class bias_type, class index_type>
index_type QuadraticModel<bias_type, index_type>::add_variable(Vartype vartype) {
    return this->add_variable(vartype, vartype_info<bias_type>::default_min(vartype),
                              vartype_info<bias_type>::default_max(vartype));
}
template <class bias_type, class index_type>
index_type QuadraticModel<bias_type, index_type>::add_variable(Vartype vartype, bias_type lb,
                                                               bias_type ub) {
    assert(lb <= ub);

    assert(lb >= vartype_info<bias_type>::min(vartype));
    assert(ub <= vartype_info<bias_type>::max(vartype));

    assert(vartype != Vartype::BINARY || lb == 0);
    assert(vartype != Vartype::BINARY || ub == 1);

    assert(vartype != Vartype::SPIN || lb == -1);
    assert(vartype != Vartype::SPIN || ub == +1);

    index_type v = this->num_variables();

    this->varinfo_.emplace_back(vartype, lb, ub);
    base_type::add_variable();

    return v;
}
template <class bias_type, class index_type>
index_type QuadraticModel<bias_type, index_type>::add_variables(Vartype vartype, index_type n) {
    index_type start = this->num_variables();
    for (index_type i = 0; i < n; ++i) {
        this->add_variable(vartype);
    }
    return start;
}
template <class bias_type, class index_type>
index_type QuadraticModel<bias_type, index_type>::add_variables(Vartype vartype, index_type n,
                                                                bias_type lb, bias_type ub) {
    index_type start = this->num_variables();
    for (index_type i = 0; i < n; ++i) {
        this->add_variable(vartype, lb, ub);
    }
    return start;
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
        base_type::change_vartype(source, target, v);
        this->varinfo_[v].lb = 0;
        this->varinfo_[v].ub = 1;
        this->varinfo_[v].vartype = Vartype::BINARY;
    } else if (source == Vartype::BINARY && target == Vartype::SPIN) {
        base_type::change_vartype(source, target, v);
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
    // should we add an assert for bounds?
    base_type::fix_variable(v, assignment);
    this->varinfo_.erase(this->varinfo_.begin() + v);
}

template <class bias_type, class index_type>
bias_type QuadraticModel<bias_type, index_type>::lower_bound(index_type v) const {
    // even though v is unused, we need this to conform the the QuadraticModelBase API
    return this->varinfo_[v].lb;
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
    this->varinfo_.erase(this->varinfo_.begin() + v);
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
    this->varinfo_.erase(this->varinfo_.begin() + n, this->varinfo_.end());
}

template <class bias_type, class index_type>
void QuadraticModel<bias_type, index_type>::resize(index_type n, Vartype vartype) {
    if (vartype == Vartype::BINARY) {
        this->resize(n, vartype, 0, 1);
    } else if (vartype == Vartype::SPIN) {
        this->resize(n, vartype, -1, +1);
    } else {
        throw std::logic_error("must provide bounds for integer vartypes when resizing");
    }
}

template <class bias_type, class index_type>
void QuadraticModel<bias_type, index_type>::resize(index_type n, Vartype vartype, bias_type lb,
                                                   bias_type ub) {
    assert(n > 0);

    assert(lb <= ub);

    assert(lb >= vartype_info<bias_type>::min(vartype));
    assert(ub <= vartype_info<bias_type>::max(vartype));

    assert(vartype != Vartype::BINARY || lb == 0);
    assert(vartype != Vartype::BINARY || ub == 1);

    assert(vartype != Vartype::SPIN || lb == -1);
    assert(vartype != Vartype::SPIN || ub == +1);

    this->varinfo_.resize(n, varinfo_type(vartype, lb, ub));
    base_type::resize(n);
}

template <class bias_type, class index_type>
void QuadraticModel<bias_type, index_type>::set_lower_bound(index_type v, bias_type lb) {
    this->varinfo_[v].lb = lb;
}

template <class bias_type, class index_type>
void QuadraticModel<bias_type, index_type>::set_upper_bound(index_type v, bias_type ub) {
    this->varinfo_[v].ub = ub;
}

template <class bias_type, class index_type>
void QuadraticModel<bias_type, index_type>::set_vartype(index_type v, Vartype vartype) {
    this->varinfo_[v].vartype = vartype;
}

template <class bias_type, class index_type>
bias_type QuadraticModel<bias_type, index_type>::upper_bound(index_type v) const {
    // even though v is unused, we need this to conform the the QuadraticModelBase API
    return this->varinfo_[v].ub;
}

template <class bias_type, class index_type>
Vartype QuadraticModel<bias_type, index_type>::vartype(index_type v) const {
    // even though v is unused, we need this to conform the the QuadraticModelBase API
    return this->varinfo_[v].vartype;
}

}  // namespace dimod
