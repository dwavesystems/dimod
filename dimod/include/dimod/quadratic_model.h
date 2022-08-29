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
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dimod/abc.h"
#include "dimod/vartypes.h"

namespace dimod {

/// A Binary Quadratic Model is a quadratic polynomial over binary variables.
template <class Bias, class Index = int>
class BinaryQuadraticModel : public abc::QuadraticModelBase<Bias, Index> {
 public:
    /// The type of the base class.
    using base_type = abc::QuadraticModelBase<Bias, Index>;

    /// The first template parameter (Bias).
    using bias_type = typename base_type::bias_type;

    /// The second template parameter (Index).
    using index_type = typename base_type::index_type;

    /// Unsigned integral that can represent non-negative values.
    using size_type = typename base_type::size_type;

    /// Empty constructor. The vartype defaults to `Vartype::BINARY`.
    BinaryQuadraticModel() : BinaryQuadraticModel(Vartype::BINARY) {}

    /// Create a BQM of the given `vartype`.
    explicit BinaryQuadraticModel(Vartype vartype) : base_type(), vartype_(vartype) {}

    /// Create a BQM with `n` variables of the given `vartype`.
    BinaryQuadraticModel(index_type n, Vartype vartype) : base_type(n), vartype_(vartype) {}

    /**
     * Create a BQM from a dense matrix.
     *
     * `dense` must be an array of length `num_variables^2`.
     *
     * Values on the diagonal are treated differently depending on the variable
     * type.
     * If the BQM is SPIN-valued, then the values on the diagonal are
     * added to the offset.
     * If the BQM is BINARY-valued, then the values on the diagonal are added
     * as linear biases.
     *
     */
    template <class T>
    BinaryQuadraticModel(const T dense[], index_type num_variables, Vartype vartype)
            : BinaryQuadraticModel(num_variables, vartype) {
        this->add_quadratic_from_dense(dense, num_variables);
    }

    using base_type::add_quadratic;

    /*
     * Construct a BQM from COO-formated iterators.
     *
     * A sparse BQM encoded in [COOrdinate] format is specified by three
     * arrays of (row, column, value).
     *
     * [COOrdinate]: https://w.wiki/n$L
     *
     * `row_iterator` must be a random access iterator  pointing to the
     * beginning of the row data. `col_iterator` must be a random access
     * iterator pointing to the beginning of the column data. `bias_iterator`
     * must be a random access iterator pointing to the beginning of the bias
     * data. `length` must be the number of (row, column, bias) entries.
     */
    template <class ItRow, class ItCol, class ItBias>
    void add_quadratic(ItRow row_iterator, ItCol col_iterator, ItBias bias_iterator,
                       index_type length) {
        // we can resize ourself because we know the vartype
        if (length > 0) {
            index_type max_label = std::max(*std::max_element(row_iterator, row_iterator + length),
                                            *std::max_element(col_iterator, col_iterator + length));
            if (max_label >= 0 && static_cast<size_type>(max_label) >= this->num_variables()) {
                this->resize(max_label + 1);
            }
        }

        base_type::add_quadratic(row_iterator, col_iterator, bias_iterator, length);
    }

    /// Add one (disconnected) variable to the BQM and return its index.
    index_type add_variable() { return base_type::add_variable(); }

    /// Change the vartype of the binary quadratic model
    void change_vartype(Vartype vartype) {
        if (vartype == this->vartype_) return;  // nothing to do
        base_type::change_vartypes(this->vartype_, vartype);
        this->vartype_ = vartype;
    }

    bias_type lower_bound() const { return vartype_info<bias_type>::min(this->vartype_); }

    bias_type lower_bound(index_type v) const {
        return vartype_info<bias_type>::min(this->vartype_);
    }

    // Resize the model to contain `n` variables.
    void resize(index_type n) { base_type::resize(n); }

    bias_type upper_bound() const { return vartype_info<bias_type>::max(this->vartype_); }

    bias_type upper_bound(index_type v) const {
        return vartype_info<bias_type>::max(this->vartype_);
    }

    Vartype vartype() const { return this->vartype_; }

    Vartype vartype(index_type v) const { return this->vartype_; }

 private:
    // The vartype of the BQM
    Vartype vartype_;
};

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

    QuadraticModel() : base_type(), varinfo_() {}

    QuadraticModel(const QuadraticModel& qm) : base_type(qm), varinfo_(qm.varinfo_) {}

    QuadraticModel(const QuadraticModel&& qm)
            : base_type(std::move(qm)), varinfo_(std::move(qm.varinfo_)) {}

    explicit QuadraticModel(const BinaryQuadraticModel<bias_type, index_type>& bqm)
            : base_type(bqm), varinfo_(bqm.num_variables(), varinfo_type(bqm.vartype())) {}

    template <class B, class I>
    explicit QuadraticModel(const BinaryQuadraticModel<B, I>& bqm) {
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

    QuadraticModel& operator=(const QuadraticModel& other) {
        base_type::operator=(other);
        this->varinfo_ = other.varinfo_;
        return *this;
    }

    QuadraticModel& operator=(QuadraticModel&& other) noexcept {
        using std::swap;
        base_type::operator=(std::move(other));
        this->varinfo_ = std::move(other.varinfo_);
        return *this;
    }

    index_type add_variable(Vartype vartype) {
        return this->add_variable(vartype, vartype_info<bias_type>::default_min(vartype),
                                  vartype_info<bias_type>::default_max(vartype));
    }

    index_type add_variable(Vartype vartype, bias_type lb, bias_type ub) {
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

    index_type add_variables(Vartype vartype, index_type n) {
        index_type start = this->num_variables();
        for (index_type i = 0; i < n; ++i) {
            this->add_variable(vartype);
        }
        return start;
    }

    index_type add_variables(Vartype vartype, index_type n, bias_type lb, bias_type ub) {
        index_type start = this->num_variables();
        for (index_type i = 0; i < n; ++i) {
            this->add_variable(vartype, lb, ub);
        }
        return start;
    }

    void clear() {
        varinfo_.clear();
        base_type::clear();
    }

    /// Change the vartype of `v`, updating the biases appropriately.
    void change_vartype(Vartype vartype, index_type v) {
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

    template <class T>
    void fix_variable(index_type v, T assignment) {
        // should we add an assert for bounds?
        base_type::fix_variable(v, assignment);
        this->varinfo_.erase(this->varinfo_.begin() + v);
    }

    bias_type lower_bound(index_type v) const { return this->varinfo_[v].lb; }

    /**
     * Total bytes consumed by the biases, vartype info, bounds, and indices.
     *
     * If `capacity` is true, use the capacity of the underlying vectors rather
     * than the size.
     */
    size_type nbytes(bool capacity = false) const noexcept {
        size_type count = base_type::nbytes(capacity);
        if (capacity) {
            count += varinfo_.capacity() * sizeof(varinfo_type);
        } else {
            count += varinfo_.size() * sizeof(varinfo_type);
        }
        return count;
    }

    void remove_variable(index_type v) {
        base_type::remove_variable(v);
        this->varinfo_.erase(this->varinfo_.begin() + v);
    }

    // Resize the model to contain `n` variables.
    void resize(index_type n) {
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

    /**
     * Resize the model to contain `n` variables.
     *
     * The `vartype` is used to any new variables added.
     *
     * The `vartype` must be `Vartype::BINARY` or `Vartype::SPIN`.
     */
    void resize(index_type n, Vartype vartype) {
        if (vartype == Vartype::BINARY) {
            this->resize(n, vartype, 0, 1);
        } else if (vartype == Vartype::SPIN) {
            this->resize(n, vartype, -1, +1);
        } else {
            throw std::logic_error("must provide bounds for integer vartypes when resizing");
        }
    }

    /**
     * Resize the model to contain `n` variables.
     *
     * The `vartype` is used to any new variables added.
     */
    void resize(index_type n, Vartype vartype, bias_type lb, bias_type ub) {
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

    void set_lower_bound(index_type v, bias_type lb) { this->varinfo_[v].lb = lb; }

    void set_upper_bound(index_type v, bias_type ub) { this->varinfo_[v].ub = ub; }

    void set_vartype(index_type v, Vartype vartype) { this->varinfo_[v].vartype = vartype; }

    bias_type upper_bound(index_type v) const { return this->varinfo_[v].ub; }

    Vartype vartype(index_type v) const { return this->varinfo_[v].vartype; }

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
};
}  // namespace dimod
