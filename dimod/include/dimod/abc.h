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
#include <cassert>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "dimod/utils.h"
#include "dimod/vartypes.h"

namespace dimod {
namespace abc {

template <class bias_type, class index_type>
struct OneVarTerm {
    index_type v;
    bias_type bias;

    OneVarTerm(index_type v, bias_type bias) : v(v), bias(bias) {}

    friend bool operator<(const OneVarTerm& a, const OneVarTerm& b) { return a.v < b.v; }
    friend bool operator<(const OneVarTerm& a, index_type v) { return a.v < v; }
};

template <class bias_type, class index_type>
struct TwoVarTerm {
    index_type u;
    index_type v;
    bias_type bias;

    explicit TwoVarTerm(index_type u) : u(u), v(-1), bias(std::numeric_limits<double>::signaling_NaN()) {}

    TwoVarTerm(index_type u, index_type v, bias_type bias): u(u), v(v), bias(bias) {}

    friend bool operator==(const TwoVarTerm& a, const TwoVarTerm& b) {
        return (a.u == b.u && a.v == b.v && a.bias == b.bias);
    }
    friend bool operator!=(const TwoVarTerm& a, const TwoVarTerm& b) { return !(a == b); }
};

template <class bias_type, class index_type>
class ConstQuadraticIterator {
 public:
    using value_type = TwoVarTerm<bias_type, index_type>;
    using pointer = const value_type*;
    using reference = const value_type&;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    ConstQuadraticIterator() : adj_ptr_(nullptr), term_(0) {}

    ConstQuadraticIterator(
            const std::vector<std::vector<OneVarTerm<bias_type, index_type>>>* adj_ptr,
            index_type u)
            : adj_ptr_(adj_ptr), term_(u), vi_(0) {
        if (adj_ptr_ != nullptr) {
            // advance through the neighborhoods until we find one on the
            // lower triangle
            for (index_type& u = term_.u; static_cast<std::size_t>(u) < adj_ptr_->size(); ++u) {
                auto& neighborhood = (*adj_ptr_)[u];

                if (neighborhood.size() && neighborhood.cbegin()->v <= u) {
                    // we found one
                    term_.v = neighborhood.cbegin()->v;
                    term_.bias = neighborhood.cbegin()->bias;
                    return;
                }
            }
        }
    }

    reference operator*() const { return term_; }

    const pointer operator->() const { return &term_; }

    ConstQuadraticIterator& operator++() {
        index_type& vi = vi_;

        ++vi;  // advance to the next

        for (index_type& u = term_.u; static_cast<std::size_t>(u) < adj_ptr_->size(); ++u) {
            auto& neighborhood = (*adj_ptr_)[u];

            auto it = neighborhood.cbegin() + vi;

            if (it != neighborhood.cend() && it->v <= u) {
                // we found one
                term_.v = it->v;
                term_.bias = it->bias;
                return *this;
            }

            vi = 0;
        }

        return *this;
    }

    ConstQuadraticIterator operator++(int) {
        ConstQuadraticIterator tmp(*this);
        ++(*this);
        return tmp;
    }

    friend bool operator==(const ConstQuadraticIterator& a, const ConstQuadraticIterator& b) {
        return (a.adj_ptr_ == nullptr && b.adj_ptr_ == nullptr) ||
               (a.adj_ptr_ == b.adj_ptr_ && a.term_.u == b.term_.u && a.vi_ == b.vi_);
    }

    friend bool operator!=(const ConstQuadraticIterator& a, const ConstQuadraticIterator& b) {
        return !(a == b);
    }

 private:
    // note that unlike QuandraticModelBase, we use a regular pointer
    // because the QuadraticModelBase owns the adjacency structure
    const std::vector<std::vector<OneVarTerm<bias_type, index_type>>>* adj_ptr_;

    value_type term_;  // the current term

    index_type vi_;  // term_.v's location in the neighborhood of term_.u
};

template <class Bias, class Index>
class QuadraticModelBase {
 public:
    /// The first template parameter (`Bias`).
    using bias_type = Bias;

    /// The second template parameter (`Index`).
    using index_type = Index;

    /// Unsigned integer type that can represent non-negative values.
    using size_type = std::size_t;

    /// @private  <- don't doc
    using const_neighborhood_iterator =
            typename std::vector<OneVarTerm<bias_type, index_type>>::const_iterator;

    /// @private  <- don't doc
    using const_quadratic_iterator = ConstQuadraticIterator<bias_type, index_type>;

    friend class ConstQuadraticIterator<bias_type, index_type>;

    virtual ~QuadraticModelBase() = default;

    QuadraticModelBase();

    QuadraticModelBase(const QuadraticModelBase&);

    QuadraticModelBase(QuadraticModelBase&&) = default;

    QuadraticModelBase& operator=(const QuadraticModelBase&);

    QuadraticModelBase& operator=(QuadraticModelBase&&) = default;

    /// Add linear bias to variable ``v``.
    void add_linear(index_type v, bias_type bias);

    /// Add offset.
    void add_offset(bias_type bias);

    /// Add interaction between variables `v` and `u`.
    void add_quadratic(index_type u, index_type v, bias_type bias);

    /// Add interactions between row `row` and column `col`.
    void add_quadratic(std::initializer_list<index_type> row, std::initializer_list<index_type> col,
                       std::initializer_list<bias_type> biases);

    template <class ItRow, class ItCol, class ItBias>
    void add_quadratic(ItRow row_iterator, ItCol col_iterator, ItBias bias_iterator,
                       index_type length);

    /**
     * Add quadratic bias for the given variables at the end of each other's neighborhoods.
     *
     * # Parameters
     * - `u` - a variable.
     * - `v` - a variable.
     * - `bias` - the quadratic bias associated with `u` and `v`.
     *
     * # Exceptions
     * When `u` is less than the largest neighbor in `v`'s neighborhood,
     * `v` is less than the largest neighbor in `u`'s neighborhood, or either
     * `u` or `v` is greater than ``num_variables()`` then the behavior of
     * this method is undefined.
     */
    void add_quadratic_back(index_type u, index_type v, bias_type bias);

    /*
     * Add quadratic biases from a dense matrix.
     *
     * `dense` must be an array of length `num_variables^2`.
     *
     * Values on the diagonal are treated differently depending on the variable
     * type.
     *
     * # Exceptions
     * The behavior of this method is undefined when the model has fewer than
     * `num_variables` variables.
     */
    template <class T>
    void add_quadratic_from_dense(const T dense[], index_type num_variables);

    /// Return an iterator to the beginning of the neighborhood of `v`.
    const_neighborhood_iterator cbegin_neighborhood(index_type v) const;

    /// Return an iterator to the end of the neighborhood of `v`.
    const_neighborhood_iterator cend_neighborhood(index_type v) const;

    /// Return an iterator to the beginning of the quadratic interactions.
    const_quadratic_iterator cbegin_quadratic() const;

    /// Return an iterator to the end of the quadratic interactions.
    const_quadratic_iterator cend_quadratic() const;

    /// Remove the offset and all variables and interactions from the model.
    void clear();

    /**
     * Return the energy of the given sample.
     *
     * The `sample_start` must be a random access iterator pointing to the
     * beginning of the sample.
     *
     * The behavior of this function is undefined when the sample is not
     * `num_variables()` long.
     */
    template <class Iter>  // todo: allow different return types
    bias_type energy(Iter sample_start) const;

    /**
     * Remove variable `v` from the model by fixing its value.
     *
     * Note that this causes a reindexing, where all variables above `v` have
     * their index reduced by one.
     */
    template <class T>
    void fix_variable(index_type v, T assignment);

    /// Check whether `u` and `v` have an interaction.
    bool has_interaction(index_type u, index_type v) const;

    /// Test whether two quadratic models are equal.
    template <class B, class I>
    bool is_equal(const QuadraticModelBase<B, I>& other) const;

    /// Test whether the model has no quadratic biases.
    bool is_linear() const;

    /// The linear bias of variable `v`.
    bias_type linear(index_type v) const;

    /// Return the lower bound on variable ``v``.
    virtual bias_type lower_bound(index_type v) const = 0;

    [[deprecated]] std::pair<const_neighborhood_iterator, const_neighborhood_iterator> neighborhood(
            index_type v) const {
        return std::make_pair(cbegin_neighborhood(v), cend_neighborhood(v));
    }

    [[deprecated]] std::pair<const_neighborhood_iterator, const_neighborhood_iterator> neighborhood(
            index_type u, index_type start) const {
        if (has_adj()) {
            const auto& n = (*adj_ptr_)[u];
            return std::make_pair(std::lower_bound(n.cbegin(), n.cend(), start), n.cend());
        } else {
            return neighborhood(u);
        }
    }

    /**
     * Total bytes consumed by the biases and indices.
     *
     * If `capacity` is true, use the capacity of the underlying vectors rather
     * than the size.
     */
    virtual size_type nbytes(bool capacity = false) const;

    /// Return the number of interactions in the quadratic model.
    size_type num_interactions() const;

    /// Return the number of other variables that `v` interacts with.
    size_type num_interactions(index_type v) const;

    /// Return the number of variables in the quadratic model.
    size_type num_variables() const { return linear_biases_.size(); }

    /// Return the offset
    bias_type offset() const;

    /**
     * Return the quadratic bias associated with `u` and `v`.
     *
     * If `u` and `v` do not have a quadratic bias, returns 0.
     *
     * Note that this function does not return a reference because
     * each quadratic bias is stored twice.
     *
     */
    bias_type quadratic(index_type u, index_type v) const;

    /**
     * Return the quadratic bias associated with `u` and `v`.
     *
     * Note that this function does not return a reference because
     * each quadratic bias is stored twice.
     *
     * Raises an `out_of_range` error if either `u` or `v` are not variables;
     * if they do not have an interaction, the function throws an exception.
     */
    bias_type quadratic_at(index_type u, index_type v) const;

    /// Remove the interaction between variables `u` and `v`.
    bool remove_interaction(index_type u, index_type v);

    /// Remove all interactions for which `filter` returns `true`.
    /// Returns the number of interactions removed.
    /// `filter` must be symmetric. That is `filter(u, v, bias)` must equal `filter(v, u, bias)`.
    template<class Filter>
    size_type remove_interactions(Filter filter);

    /**
     * Remove variable `v` from the model.
     *
     * Note that this causes a reindexing, where all variables above `v` have
     * their index reduced by one.
     */
    virtual void remove_variable(index_type v);

    /// Remove multiple variables from the model and reindex accordingly.
    virtual void remove_variables(const std::vector<index_type>& variables);

    /// Multiply all biases by the value of `scalar`.
    void scale(bias_type scalar);

    /// Set the linear bias of variable `v`.
    void set_linear(index_type v, bias_type bias);

    /// Set the linear biases of of the variables beginning with `v`.
    void set_linear(index_type v, std::initializer_list<bias_type> biases);

    /// Set the offset.
    void set_offset(bias_type offset);

    /// Set the quadratic bias between variables `u` and `v`.
    void set_quadratic(index_type u, index_type v, bias_type bias);

    void substitute_variable(index_type v, bias_type multiplier, bias_type offset);

    void substitute_variables(bias_type multiplier, bias_type offset);

    /// Return the upper bound on variable ``v``.
    virtual bias_type upper_bound(index_type v) const = 0;

    /// Return the variable type of variable ``v``.
    virtual Vartype vartype(index_type v) const = 0;

 protected:
    explicit QuadraticModelBase(std::vector<bias_type>&& linear_biases)
            : linear_biases_(linear_biases), adj_ptr_(), offset_(0) {}

    explicit QuadraticModelBase(index_type n) : linear_biases_(n), adj_ptr_(), offset_(0) {}

    /// Increase the size of the model by one. Returns the index of the new variable.
    index_type add_variable();

    /// Increase the size of the model by `n`. Returns the index of the first variable added.
    index_type add_variables(index_type n);

    /// Return an empty neighborhood; useful when a variable does not have an adjacency.
    static const std::vector<OneVarTerm<bias_type, index_type>>& empty_neighborhood() {
        static std::vector<OneVarTerm<bias_type, index_type>> empty;
        return empty;
    }

    /// Resize model to contain n variables.
    void resize(index_type n);

    /// Protected version of vartype() that allows subclasses to distinguish
    /// between the `vartype_` called by mixin functions and the public API one.
    /// By default they are the same.
    virtual Vartype vartype_(index_type v) const { return vartype(v); }

 private:
    std::vector<bias_type> linear_biases_;

    std::unique_ptr<std::vector<std::vector<OneVarTerm<bias_type, index_type>>>> adj_ptr_;

    bias_type offset_;

    // Assumes adj exists!
    // Creates the bias if it doesn't already exist
    bias_type& asymmetric_quadratic_ref(index_type u, index_type v) {
        assert(0 <= u && static_cast<size_type>(u) < num_variables());
        assert(0 <= v && static_cast<size_type>(v) < num_variables());
        assert(has_adj());

        auto& neighborhood = (*adj_ptr_)[u];
        auto it = std::lower_bound(neighborhood.begin(), neighborhood.end(), v);
        if (it == neighborhood.end() || it->v != v) {
            // we could make a bunch individual functions to avoid needing to
            // default to 0, but this is a lot simpler.
            it = neighborhood.emplace(it, v, 0);
        }
        return it->bias;
    }

    /// Create the adjacency structure if it doesn't already exist.
    void enforce_adj() {
        if (!adj_ptr_) {
            adj_ptr_ = std::unique_ptr<std::vector<std::vector<OneVarTerm<bias_type, index_type>>>>(
                    new std::vector<std::vector<OneVarTerm<bias_type, index_type>>>(
                            num_variables()));
        }
    }

    /// Return true if the model's adjacency structure exists
    bool has_adj() const { return static_cast<bool>(adj_ptr_); }
};

template <class bias_type, class index_type>
QuadraticModelBase<bias_type, index_type>::QuadraticModelBase()
        : linear_biases_(), adj_ptr_(), offset_(0) {}

template <class bias_type, class index_type>
QuadraticModelBase<bias_type, index_type>::QuadraticModelBase(const QuadraticModelBase& other)
        : linear_biases_(other.linear_biases_), adj_ptr_(), offset_(other.offset_) {
    // need to handle the adj if present
    if (!other.is_linear()) {
        adj_ptr_ = std::unique_ptr<std::vector<std::vector<OneVarTerm<bias_type, index_type>>>>(
                new std::vector<std::vector<OneVarTerm<bias_type, index_type>>>(*other.adj_ptr_));
    }
}

template <class bias_type, class index_type>
QuadraticModelBase<bias_type, index_type>& QuadraticModelBase<bias_type, index_type>::operator=(
        const QuadraticModelBase& other) {
    if (this != &other) {
        linear_biases_ = other.linear_biases_;
        if (!other.is_linear()) {
            adj_ptr_ = std::unique_ptr<std::vector<std::vector<OneVarTerm<bias_type, index_type>>>>(
                    new std::vector<std::vector<OneVarTerm<bias_type, index_type>>>(
                            *other.adj_ptr_));
        } else {
            adj_ptr_.reset(nullptr);
        }
        offset_ = other.offset_;
    }
    return *this;
}

template <class bias_type, class index_type>
void QuadraticModelBase<bias_type, index_type>::add_linear(index_type v, bias_type bias) {
    assert(v >= 0 && static_cast<size_type>(v) < num_variables());
    linear_biases_[v] += bias;
}

template <class bias_type, class index_type>
void QuadraticModelBase<bias_type, index_type>::add_offset(bias_type bias) {
    offset_ += bias;
}

template <class bias_type, class index_type>
void QuadraticModelBase<bias_type, index_type>::add_quadratic(index_type u, index_type v,
                                                              bias_type bias) {
    assert(0 <= u && static_cast<size_type>(u) < num_variables());
    assert(0 <= v && static_cast<size_type>(v) < num_variables());

    enforce_adj();

    if (u == v) {
        switch (this->vartype_(u)) {
            case Vartype::BINARY: {
                // 1*1 == 1 and 0*0 == 0 so this is linear
                linear_biases_[u] += bias;
                break;
            }
            case Vartype::SPIN: {
                // -1*-1 == +1*+1 == 1 so this is a constant offset
                offset_ += bias;
                break;
            }
            default: {
                // self-loop
                asymmetric_quadratic_ref(u, u) += bias;
                break;
            }
        }
    } else {
        asymmetric_quadratic_ref(u, v) += bias;
        asymmetric_quadratic_ref(v, u) += bias;
    }
}

template <class bias_type, class index_type>
void QuadraticModelBase<bias_type, index_type>::add_quadratic(
        std::initializer_list<index_type> row, std::initializer_list<index_type> col,
        std::initializer_list<bias_type> biases) {
    auto rit = row.begin();
    auto cit = col.begin();
    auto bit = biases.begin();
    for (; rit < row.end() && cit < col.end() && bit < biases.end(); ++rit, ++cit, ++bit) {
        add_quadratic(*rit, *cit, *bit);
    }
}

template <class bias_type, class index_type>
template <class ItRow, class ItCol, class ItBias>
void QuadraticModelBase<bias_type, index_type>::add_quadratic(ItRow row_iterator,
                                                              ItCol col_iterator,
                                                              ItBias bias_iterator,
                                                              index_type length) {
    // todo: performance testing of adding and then sorting later
    for (index_type i = 0; i < length; ++i, ++row_iterator, ++col_iterator, ++bias_iterator) {
        add_quadratic(*row_iterator, *col_iterator, *bias_iterator);
    }
}

template <class bias_type, class index_type>
void QuadraticModelBase<bias_type, index_type>::add_quadratic_back(index_type u, index_type v,
                                                                   bias_type bias) {
    assert(0 <= u && static_cast<size_t>(u) <= num_variables());
    assert(0 <= v && static_cast<size_t>(v) <= num_variables());

    enforce_adj();

    // check the condition for adding at the back
    assert((*adj_ptr_)[v].empty() || (*adj_ptr_)[v].back().v <= u);
    assert((*adj_ptr_)[u].empty() || (*adj_ptr_)[u].back().v <= v);

    if (u == v) {
        switch (this->vartype_(u)) {
            case Vartype::BINARY: {
                // 1*1 == 1 and 0*0 == 0 so this is linear
                add_linear(u, bias);
                break;
            }
            case Vartype::SPIN: {
                // -1*-1 == +1*+1 == 1 so this is a constant offset
                offset_ += bias;
                break;
            }
            default: {
                // self-loop
                (*adj_ptr_)[u].emplace_back(v, bias);
                break;
            }
        }
    } else {
        (*adj_ptr_)[u].emplace_back(v, bias);
        (*adj_ptr_)[v].emplace_back(u, bias);
    }
}

template <class bias_type, class index_type>
template <class T>
void QuadraticModelBase<bias_type, index_type>::add_quadratic_from_dense(const T dense[],
                                                                         index_type num_variables) {
    static_assert(std::is_arithmetic<T>::value, "T must be numeric");
    assert(0 <= num_variables);
    assert(static_cast<size_type>(num_variables) <= this->num_variables());

    enforce_adj();

    if (is_linear()) {
        for (index_type u = 0; u < num_variables; ++u) {
            // diagonal
            add_quadratic_back(u, u, dense[u * (num_variables + 1)]);

            // off-diagonal
            for (index_type v = u + 1; v < num_variables; ++v) {
                bias_type qbias = dense[u * num_variables + v] + dense[v * num_variables + u];

                if (qbias) {
                    add_quadratic_back(u, v, qbias);
                }
            }
        }
    } else {
        // we cannot rely on the ordering
        for (index_type u = 0; u < num_variables; ++u) {
            // diagonal
            add_quadratic(u, u, dense[u * (num_variables + 1)]);

            // off-diagonal
            for (index_type v = u + 1; v < num_variables; ++v) {
                bias_type qbias = dense[u * num_variables + v] + dense[v * num_variables + u];

                if (qbias) {
                    add_quadratic(u, v, qbias);
                }
            }
        }
    }
}

template <class bias_type, class index_type>
index_type QuadraticModelBase<bias_type, index_type>::add_variable() {
    return add_variables(1);
}

template <class bias_type, class index_type>
index_type QuadraticModelBase<bias_type, index_type>::add_variables(index_type n) {
    assert(n >= 0);
    index_type size = num_variables();

    linear_biases_.resize(size + n);
    if (has_adj()) {
        adj_ptr_->resize(size + n);
    }

    return size;
}

template <class bias_type, class index_type>
typename QuadraticModelBase<bias_type, index_type>::const_neighborhood_iterator
QuadraticModelBase<bias_type, index_type>::cbegin_neighborhood(index_type v) const {
    assert(0 <= v && static_cast<size_t>(v) <= num_variables());
    if (has_adj()) {
        return (*adj_ptr_)[v].begin();
    } else {
        return empty_neighborhood().begin();
    }
}

template <class bias_type, class index_type>
typename QuadraticModelBase<bias_type, index_type>::const_neighborhood_iterator
QuadraticModelBase<bias_type, index_type>::cend_neighborhood(index_type v) const {
    assert(0 <= v && static_cast<size_t>(v) <= num_variables());
    if (has_adj()) {
        return (*adj_ptr_)[v].end();
    } else {
        return empty_neighborhood().end();
    }
}

template <class bias_type, class index_type>
ConstQuadraticIterator<bias_type, index_type>
QuadraticModelBase<bias_type, index_type>::cbegin_quadratic() const {
    return const_quadratic_iterator(adj_ptr_.get(), 0);
}

template <class bias_type, class index_type>
ConstQuadraticIterator<bias_type, index_type>
QuadraticModelBase<bias_type, index_type>::cend_quadratic() const {
    return const_quadratic_iterator(adj_ptr_.get(), num_variables());
}

template <class bias_type, class index_type>
void QuadraticModelBase<bias_type, index_type>::clear() {
    adj_ptr_.reset(nullptr);
    linear_biases_.clear();
    offset_ = 0;
}

template <class bias_type, class index_type>
template <class Iter>
bias_type QuadraticModelBase<bias_type, index_type>::energy(Iter sample_start) const {
    static_assert(std::is_same<std::random_access_iterator_tag,
                               typename std::iterator_traits<Iter>::iterator_category>::value,
                  "iterators must be random access");

    bias_type en = offset();

    if (has_adj()) {
        for (index_type u = 0; static_cast<size_type>(u) < num_variables(); ++u) {
            auto u_val = *(sample_start + u);

            en += u_val * linear(u);

            for (auto& term : (*adj_ptr_)[u]) {
                if (term.v > u) break;
                en += term.bias * u_val * *(sample_start + term.v);
            }
        }
    } else {
        for (auto it = linear_biases_.begin(); it != linear_biases_.end(); ++it, ++sample_start) {
            en += *sample_start * *it;
        }
    }

    return en;
}

template <class bias_type, class index_type>
template <class T>
void QuadraticModelBase<bias_type, index_type>::fix_variable(index_type v, T assignment) {
    static_assert(std::is_arithmetic<T>::value, "T must be numeric");
    assert(v >= 0 && static_cast<size_type>(v) < num_variables());
    // associated quadratic biases become linear
    if (has_adj()) {
        for (auto it = cbegin_neighborhood(v); it != cend_neighborhood(v); ++it) {
            add_linear(it->v, it->bias * assignment);
        }
    }

    // linear gets added to the offset
    // by doing this after the quadratic we also handle self-loops
    add_offset(assignment * linear(v));

    // finally remove v
    QuadraticModelBase<bias_type, index_type>::remove_variable(v);
}

template <class bias_type, class index_type>
bool QuadraticModelBase<bias_type, index_type>::has_interaction(index_type u, index_type v) const {
    assert(0 <= u && static_cast<size_type>(u) < num_variables());
    assert(0 <= v && static_cast<size_type>(v) < num_variables());

    if (!adj_ptr_) {
        return false;
    }

    const auto& n = (*adj_ptr_)[u];
    auto it = std::lower_bound(n.cbegin(), n.cend(), v);
    if (it == n.cend() || it->v != v) {
        return false;
    }

    return true;
}

template <class bias_type, class index_type>
template <class B, class I>
bool QuadraticModelBase<bias_type, index_type>::is_equal(
        const QuadraticModelBase<B, I>& other) const {
    // easy checks first
    if (this->num_variables() != other.num_variables() ||
        this->num_interactions() != other.num_interactions() || this->offset_ != other.offset_ ||
        !std::equal(this->linear_biases_.begin(), this->linear_biases_.end(),
                    other.linear_biases_.begin())) {
        return false;
    }

    // check the vartype
    for (size_type v = 0; v < this->num_variables(); ++v) {
        if (this->vartype_(v) != other.vartype_(v)) {
            return false;
        }
    }

    // check quadratic. We already checked the number of interactions so
    // if one is present we can assume it's empty
    if (this->has_adj() && other.has_adj()) {
        // check that the neighborhoods are equal
        auto it1 = this->cbegin_quadratic();
        auto it2 = other.cbegin_quadratic();
        for (; it1 != this->cend_quadratic(); ++it1, ++it2) {
            if (*it1 != *it2) {
                return false;
            }
        }
    }

    return true;
}

template <class bias_type, class index_type>
bool QuadraticModelBase<bias_type, index_type>::is_linear() const {
    if (has_adj()) {
        for (const auto& n : *adj_ptr_) {
            if (n.size()) return false;
        }
    }
    return true;
}

template <class bias_type, class index_type>
bias_type QuadraticModelBase<bias_type, index_type>::linear(index_type v) const {
    assert(0 <= v && static_cast<size_t>(v) <= num_variables());
    return linear_biases_[v];
}

template <class bias_type, class index_type>
std::size_t QuadraticModelBase<bias_type, index_type>::nbytes(bool capacity) const {
    size_type count = sizeof(bias_type);  // offset
    if (capacity) {
        count += linear_biases_.capacity() * sizeof(bias_type);
    } else {
        count += linear_biases_.size() * sizeof(bias_type);
    }
    if (has_adj()) {
        if (capacity) {
            for (const auto& n : (*adj_ptr_)) {
                count += n.capacity() * sizeof(OneVarTerm<bias_type, index_type>);
            }
        } else {
            for (const auto& n : (*adj_ptr_)) {
                count += n.size() * sizeof(OneVarTerm<bias_type, index_type>);
            }
        }
    }
    return count;
}

template <class bias_type, class index_type>
std::size_t QuadraticModelBase<bias_type, index_type>::num_interactions() const {
    size_type count = 0;
    if (has_adj()) {
        index_type v = 0;
        for (auto& n : *(adj_ptr_)) {
            count += n.size();

            // account for self-loops
            auto lb = std::lower_bound(n.cbegin(), n.cend(), v);
            if (lb != n.cend() && lb->v == v) {
                count += 1;
            }

            ++v;
        }
    }
    return count / 2;
}

template <class bias_type, class index_type>
std::size_t QuadraticModelBase<bias_type, index_type>::num_interactions(index_type v) const {
    if (has_adj()) {
        return (*adj_ptr_)[v].size();
    } else {
        return 0;
    }
}

template <class bias_type, class index_type>
bias_type QuadraticModelBase<bias_type, index_type>::offset() const {
    return offset_;
}

template <class bias_type, class index_type>
bias_type QuadraticModelBase<bias_type, index_type>::quadratic(index_type u, index_type v) const {
    if (!adj_ptr_) {
        return 0;
    }

    const auto& n = (*adj_ptr_)[u];
    auto it = std::lower_bound(n.cbegin(), n.cend(), v);
    if (it == n.cend() || it->v != v) {
        return 0;
    }

    return it->bias;
}

template <class bias_type, class index_type>
bias_type QuadraticModelBase<bias_type, index_type>::quadratic_at(index_type u,
                                                                  index_type v) const {
    if (!adj_ptr_) {
        throw std::out_of_range("given variables have no interaction");
    }

    const auto& n = (*adj_ptr_)[u];
    auto it = std::lower_bound(n.cbegin(), n.cend(), v);
    if (it == n.cend() || it->v != v) {
        throw std::out_of_range("given variables have no interaction");
    }

    return it->bias;
}

template <class bias_type, class index_type>
bool QuadraticModelBase<bias_type, index_type>::remove_interaction(index_type u, index_type v) {
    if (!has_adj()) return false;  // no quadratic to remove

    auto& Nu = (*adj_ptr_)[u];
    auto it = std::lower_bound(Nu.begin(), Nu.end(), v);  // find v in the neighborhood of u
    if (it != Nu.end() && it->v == v) {
        // u and v have an interaction
        Nu.erase(it);

        if (u != v) {
            auto& Nv = (*adj_ptr_)[v];
            Nv.erase(std::lower_bound(Nv.begin(), Nv.end(), u));  // guaranteed to be present
        }
        return true;
    }
    return false;
}

template <class bias_type, class index_type>
template <class Filter>
std::size_t QuadraticModelBase<bias_type, index_type>::remove_interactions(Filter filter) {
    if (!has_adj()) return 0;  // nothing to filter

    std::size_t num_removed = 0;

    index_type u = 0;
    for (auto& n : *adj_ptr_) {
        auto it = std::remove_if(n.begin(), n.end(),
                                 [&u, &filter](const OneVarTerm<bias_type, index_type>& term) {
                                     const index_type& v = term.v;
                                     const bias_type& bias = term.bias;
                                     assert(filter(u, v, bias) == filter(v, u, bias));
                                     return filter(u, v, bias);
                                 });

        num_removed += n.end() - it;

        n.erase(it, n.end());

        u += 1;
    }

    assert(num_removed % 2 == 0);

    return num_removed / 2;
}

template <class bias_type, class index_type>
void QuadraticModelBase<bias_type, index_type>::remove_variable(index_type v) {
    assert(0 <= v && static_cast<size_type>(v) < num_variables());

    linear_biases_.erase(linear_biases_.cbegin() + v);

    if (has_adj()) {
        // remove v's neighborhood
        adj_ptr_->erase(adj_ptr_->cbegin() + v);

        for (auto& n : *adj_ptr_) {
            // work backwards through each neighborhood, decrementing the indices above v
            // we'll work with a regular iterator rather than a reverse one because
            // we'll eventually call erase
            for (auto it = n.end(); it != n.begin();) {
                --it;

                if (it->v > v) {
                    // decrement the index
                    --(it->v);
                } else if (it->v == v) {
                    // remove this element and break
                    n.erase(it);
                    break;
                } else {
                    // we're done
                    break;
                }
            }
        }
    }
}

template <class bias_type, class index_type>
void QuadraticModelBase<bias_type, index_type>::remove_variables(
        const std::vector<index_type>& variables) {
    if (!variables.size()) return;  // shortcut

    if (!std::is_sorted(variables.begin(), variables.end())) {
        // create a copy and sort it
        std::vector<index_type> sorted_indices = variables;
        std::sort(sorted_indices.begin(), sorted_indices.end());
        QuadraticModelBase<bias_type, index_type>::remove_variables(sorted_indices);
        return;
    }

    linear_biases_.erase(utils::remove_by_index(linear_biases_.begin(), linear_biases_.end(),
                                                variables.begin(), variables.end()),
                         linear_biases_.end());

    if (has_adj()) {
        // clean up the remaining neighborhoods
        // in this case we need a reindexing scheme, so we do the expensive O(num_variables)
        // thing once to save time later on
        std::vector<int> reindex(adj_ptr_->size());
        for (const auto& v : variables) {
            if (v > static_cast<int>(reindex.size())) break;  // we can break because it's sorted
            reindex[v] = -1;
        }
        int label = 0;
        for (auto& v : reindex) {
            if (v == -1) continue;  // the removed variables
            v = label;
            ++label;
        }

        // remove the relevant neighborhoods
        adj_ptr_->erase(utils::remove_by_index(adj_ptr_->begin(), adj_ptr_->end(), variables.begin(),
                                               variables.end()),
                        adj_ptr_->end());

        // now go through and adjust the remaining neighborhoods
        auto pred = [&reindex](OneVarTerm<bias_type, index_type>& term) {
            if (reindex[term.v] == -1) return true;  // remove
            // otherwise apply the new label
            term.v = reindex[term.v];
            return false;
        };
        for (auto& n : *adj_ptr_) {
            // we modify the indices and remove the variables we need to remove
            n.erase(std::remove_if(n.begin(), n.end(), pred), n.end());
        }
    }
}

template <class bias_type, class index_type>
void QuadraticModelBase<bias_type, index_type>::resize(index_type n) {
    assert(n >= 0);

    if (has_adj()) {
        if (static_cast<size_type>(n) < num_variables()) {
            // Clean out any of the to-be-deleted variables from the
            // neighborhoods.
            // This approach is better in the dense case. In the sparse case
            // we could determine which neighborhoods need to be trimmed rather
            // than just doing them all.
            for (auto& neighborhood : (*adj_ptr_)) {
                neighborhood.erase(std::lower_bound(neighborhood.begin(), neighborhood.end(), n),
                                   neighborhood.end());
            }
        }
        adj_ptr_->resize(n);
    }

    linear_biases_.resize(n);

    assert(!has_adj() || linear_biases_.size() == adj_ptr_->size());
}

template <class bias_type, class index_type>
void QuadraticModelBase<bias_type, index_type>::scale(bias_type scalar) {
    offset_ *= scalar;

    // linear biases
    for (bias_type& bias : linear_biases_) {
        bias *= scalar;
    }

    if (has_adj()) {
        // quadratic biases
        for (auto& n : (*adj_ptr_)) {
            for (auto& term : n) {
                term.bias *= scalar;
            }
        }
    }
}

template <class bias_type, class index_type>
void QuadraticModelBase<bias_type, index_type>::set_linear(index_type v, bias_type bias) {
    assert(v >= 0 && static_cast<size_type>(v) < num_variables());
    linear_biases_[v] = bias;
}

template <class bias_type, class index_type>
void QuadraticModelBase<bias_type, index_type>::set_linear(
        index_type v, std::initializer_list<bias_type> biases) {
    assert(v >= 0 && static_cast<size_type>(v) < num_variables());
    assert(biases.size() + v <= num_variables());
    for (const bias_type& b : biases) {
        linear_biases_[v] = b;
        ++v;
    }
}

template <class bias_type, class index_type>
void QuadraticModelBase<bias_type, index_type>::set_offset(bias_type offset) {
    offset_ = offset;
}

template <class bias_type, class index_type>
void QuadraticModelBase<bias_type, index_type>::set_quadratic(index_type u, index_type v,
                                                              bias_type bias) {
    assert(0 <= u && static_cast<size_t>(u) < num_variables());
    assert(0 <= v && static_cast<size_t>(v) < num_variables());

    enforce_adj();

    if (u == v) {
        switch (this->vartype_(u)) {
            // unlike add_quadratic, setting is not really defined.
            case Vartype::BINARY: {
                throw std::domain_error(
                        "Cannot set the quadratic bias of a binary variable with itself");
            }
            case Vartype::SPIN: {
                throw std::domain_error(
                        "Cannot set the quadratic bias of a spin variable with itself");
            }
            default: {
                asymmetric_quadratic_ref(u, v) = bias;
                break;
            }
        }
    } else {
        asymmetric_quadratic_ref(u, v) = bias;
        asymmetric_quadratic_ref(v, u) = bias;
    }
}

// todo: version the accepts rational numbers
template <class bias_type, class index_type>
void QuadraticModelBase<bias_type, index_type>::substitute_variable(index_type v,
                                                                    bias_type multiplier,
                                                                    bias_type offset) {
    offset_ += linear_biases_[v] * offset;
    linear_biases_[v] *= multiplier;

    if (has_adj()) {
        for (auto& term : (*adj_ptr_)[v]) {
            linear_biases_[term.v] += term.bias * offset;

            // the quadratic interactions
            asymmetric_quadratic_ref(term.v, v) *= multiplier;
            term.bias *= multiplier;
        }
    }
}

// todo: version that accepts rational numbers
template <class bias_type, class index_type>
void QuadraticModelBase<bias_type, index_type>::substitute_variables(bias_type multiplier,
                                                                     bias_type offset) {
    bias_type quad_mp = multiplier * multiplier;
    bias_type lin_quad_mp = multiplier * offset;
    bias_type quad_offset_mp = offset * offset / 2;  // we do this twice so divide by two

    for (size_type v = 0; v < num_variables(); ++v) {
        offset_ += linear_biases_[v] * offset;
        linear_biases_[v] *= multiplier;
    }

    if (has_adj()) {
        for (size_type v = 0; v < num_variables(); ++v) {
            for (auto& term : (*adj_ptr_)[v]) {
                offset_ += quad_offset_mp * term.bias;
                linear_biases_[v] += lin_quad_mp * term.bias;
                term.bias *= quad_mp;
            }
        }
    }
}

}  // namespace abc
}  // namespace dimod
