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
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "dimod/vartypes.h"

namespace dimod {
namespace abc {

template <class bias_type, class index_type>
struct LinearTerm {
    index_type v;
    bias_type bias;

    LinearTerm(index_type v, bias_type bias) : v(v), bias(bias) {}

    friend bool operator<(const LinearTerm& a, const LinearTerm& b) { return a.v < b.v; }
    friend bool operator<(const LinearTerm& a, index_type v) { return a.v < v; }
};

template <class bias_type, class index_type>
struct QuadraticTerm {
    index_type u;
    index_type v;
    bias_type bias;

    explicit QuadraticTerm(index_type u) : u(u), v(-1), bias(NAN) {}

    friend bool operator==(const QuadraticTerm& a, const QuadraticTerm& b) {
        return (a.u == b.u && a.v == b.v && a.bias == b.bias);
    }
    friend bool operator!=(const QuadraticTerm& a, const QuadraticTerm& b) { return !(a == b); }
};

template <class bias_type, class index_type>
class ConstQuadraticIterator {
 public:
    using value_type = QuadraticTerm<bias_type, index_type>;
    using pointer = const value_type*;
    using reference = const value_type&;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    ConstQuadraticIterator() : adj_ptr_(nullptr), term_(0) {}

    ConstQuadraticIterator(
            const std::vector<std::vector<LinearTerm<bias_type, index_type>>>* adj_ptr,
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

    const pointer operator->() const { return &(term_); }

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
    const std::vector<std::vector<LinearTerm<bias_type, index_type>>>* adj_ptr_;

    value_type term_;  // the current term

    index_type vi_;  // term_.v's location in the neighborhood of term_.u
};

template <class Bias, class Index>
class QuadraticModelBase {
 public:
    /// The first template parameter (Bias).
    using bias_type = Bias;

    /// The second template parameter (Index).
    using index_type = Index;

    /// Unsigned integral type that can represent non-negative values.
    using size_type = std::size_t;

    /// @private  <- don't doc
    using const_neighborhood_iterator =
            typename std::vector<LinearTerm<bias_type, index_type>>::const_iterator;

    /// @private  <- don't doc
    using const_quadratic_iterator = ConstQuadraticIterator<bias_type, index_type>;

    friend class ConstQuadraticIterator<bias_type, index_type>;

    virtual ~QuadraticModelBase() {}

    /// Add linear bias to variable ``v``.
    void add_linear(index_type v, bias_type bias);

    /// Add offset.
    void add_offset(bias_type bias);

    void add_quadratic(index_type u, index_type v, bias_type bias);

    void add_quadratic(std::initializer_list<index_type> row, std::initializer_list<index_type> col,
                       std::initializer_list<bias_type> biases);

    template <class ItRow, class ItCol, class ItBias>
    void add_quadratic(ItRow row_iterator, ItCol col_iterator, ItBias bias_iterator,
                       index_type length);

    /**
     * Add quadratic bias for the given variables at the end of eachother's neighborhoods.
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

    const_neighborhood_iterator cbegin_neighborhood(index_type v) const {
        if (has_adj()) {
            return (*adj_ptr_)[v].begin();
        } else {
            // I am a bit suspicious that this works, but it seems to
            return const_neighborhood_iterator();
        }
    }

    const_neighborhood_iterator cend_neighborhood(index_type v) const {
        if (has_adj()) {
            return (*adj_ptr_)[v].end();
        } else {
            // I am a bit suspicious that this works, but it seems to
            return const_neighborhood_iterator();
        }
    }

    /// todo
    const_quadratic_iterator cbegin_quadratic() const;

    /// todo
    const_quadratic_iterator cend_quadratic() const;

    /// Remove the offset and all variables and interactions from the model.
    void clear();

    /**
     * Return the energy of the given sample.
     *
     * The `sample_start` must be random access iterator pointing to the
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

    /// Test whether two quadratic models are equal.
    template <class B, class I>
    bool is_equal(const QuadraticModelBase<B, I>& other) const;

    /// Return True if the model has no quadratic biases.
    bool is_linear() const;

    /// The linear bias of variable `v`.
    bias_type linear(index_type v) const;

    /// The lower bound of variable `v`.
    virtual bias_type lower_bound(index_type) const = 0;

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
    size_type nbytes(bool capacity = false) const;

    /// Return the number of interactions in the quadratic model.
    size_type num_interactions() const;

    /// The number of other variables `v` interacts with.
    size_type num_interactions(index_type v) const;

    /// Return the number of variables in the quadratic model.
    size_type num_variables() const { return linear_biases_.size(); }

    /// Return the offset
    bias_type offset() const;

    /**
     * Return the quadratic bias associated with `u`, `v`.
     *
     * If `u` and `v` do not have a quadratic bias, returns 0.
     *
     * Note that this function does not return a reference, this is because
     * each quadratic bias is stored twice.
     *
     */
    bias_type quadratic(index_type u, index_type v) const;

    /**
     * Return the quadratic bias associated with `u`, `v`.
     *
     * Note that this function does not return a reference, this is because
     * each quadratic bias is stored twice.
     *
     * Raises an `out_of_range` error if either `u` or `v` are not variables or
     * if they do not have an interaction then the function throws an exception.
     */
    bias_type quadratic_at(index_type u, index_type v) const;

    /// Remove the interaction between variables `u` and `v`.
    bool remove_interaction(index_type u, index_type v);

    /**
     * Remove variable `v` from the model.
     *
     * Note that this causes a reindexing, where all variables above `v` have
     * their index reduced by one.
     */
    void remove_variable(index_type v);

    /// Multiply all biases 'scalar'
    void scale(bias_type scalar);

    /// Set the linear bias of variable `v`.
    void set_linear(index_type v, bias_type bias);

    /// Set the linear biases of of the variables beginning with `v`.
    void set_linear(index_type v, std::initializer_list<bias_type> biases);

    /// Set the offset.
    void set_offset(bias_type offset);

    /// Set the quadratic bias for the given variables.
    void set_quadratic(index_type u, index_type v, bias_type bias);

    /// Return the upper bound of variable of `v`.
    virtual bias_type upper_bound(index_type) const = 0;

    /// Return the vartype of `v`.
    virtual Vartype vartype(index_type) const = 0;

 protected:
    QuadraticModelBase();

    /// Copy constructor
    QuadraticModelBase(const QuadraticModelBase& other);

    /// Move constructor
    QuadraticModelBase(QuadraticModelBase&& other) noexcept;

    /// Copy assignment operator
    QuadraticModelBase& operator=(const QuadraticModelBase& other) {
        if (this != &other) {
            this->linear_biases_ = other.linear_biases_;  // copy
            if (other.has_adj()) {
                this->adj_ptr_ = std::unique_ptr<
                        std::vector<std::vector<LinearTerm<bias_type, index_type>>>>(
                        new std::vector<std::vector<LinearTerm<bias_type, index_type>>>(
                                *other.adj_ptr_));
            } else {
                this->adj_ptr_.reset(nullptr);
            }

            this->offset_ = other.offset_;
        }
        return *this;
    }

    /// Move assignment operator
    QuadraticModelBase& operator=(QuadraticModelBase&& other) noexcept {
        if (this != &other) {
            this->linear_biases_ = std::move(other.linear_biases_);
            this->adj_ptr_ = std::move(other.adj_ptr_);
            this->offset_ = other.offset_;
        }
        return *this;
    }

    explicit QuadraticModelBase(std::vector<bias_type>&& linear_biases)
            : linear_biases_(linear_biases), adj_ptr_(), offset_(0) {}

    explicit QuadraticModelBase(index_type n) : linear_biases_(n), adj_ptr_(), offset_(0) {}

    /// Increase the size of the model by one. Returns the index of the new variable.
    index_type add_variable();

    /// Increase the size of the model by `n`. Returns the index of the first variable added.
    index_type add_variables(index_type n);

    /**
     * Adjust the biases of a single variable to reinterpret it as being a
     * different vartype.
     *
     * This method is meant to be used by subclasses, since it does not check
     * or update the actual vartypes of the variables.
     */
    void change_vartype(Vartype source, Vartype target, index_type v);

    /**
     * Adjust all of the biases in the model to reinterpret them as being
     * from the source vartype to the target.
     *
     * This method is meant to be used by subclasses, since it does not check
     * or update the actual vartypes of the variables.
     */
    void change_vartypes(Vartype source, Vartype target);

    /// Resize model to contain n variables.
    void resize(index_type n);

 private:
    std::vector<bias_type> linear_biases_;

    std::unique_ptr<std::vector<std::vector<LinearTerm<bias_type, index_type>>>> adj_ptr_;

    bias_type offset_;

    // Assumes adj exists!
    // Creates the bias if it doesn't already exist
    bias_type& asym_quadratic(index_type u, index_type v) {
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
            adj_ptr_ = std::unique_ptr<std::vector<std::vector<LinearTerm<bias_type, index_type>>>>(
                    new std::vector<std::vector<LinearTerm<bias_type, index_type>>>(
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
        adj_ptr_ = std::unique_ptr<std::vector<std::vector<LinearTerm<bias_type, index_type>>>>(
                new std::vector<std::vector<LinearTerm<bias_type, index_type>>>(*other.adj_ptr_));
    }
}

template <class bias_type, class index_type>
QuadraticModelBase<bias_type, index_type>::QuadraticModelBase(QuadraticModelBase&& other) noexcept {
    *this = std::move(other);
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
        switch (this->vartype(u)) {
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
                asym_quadratic(u, u) += bias;
                break;
            }
        }
    } else {
        asym_quadratic(u, v) += bias;
        asym_quadratic(v, u) += bias;
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
        switch (this->vartype(u)) {
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
void QuadraticModelBase<bias_type, index_type>::change_vartype(Vartype source, Vartype target,
                                                               index_type v) {
    // todo: template version with constexpr if
    if (source == target) {
        return;
    } else if (source == Vartype::SPIN && target == Vartype::BINARY) {
        if (has_adj()) {
            for (auto it = (*adj_ptr_)[v].begin(); it != (*adj_ptr_)[v].end(); ++it) {
                linear_biases_[it->v] -= it->bias;
                asym_quadratic(it->v, v) *= 2;  // log(n)
                it->bias *= 2;
            }
        }

        offset_ -= linear_biases_[v];
        linear_biases_[v] *= 2;
    } else if (source == Vartype::BINARY && target == Vartype::SPIN) {
        if (has_adj()) {
            for (auto it = (*adj_ptr_)[v].begin(); it != (*adj_ptr_)[v].end(); ++it) {
                linear_biases_[it->v] += it->bias / 2;
                asym_quadratic(it->v, v) /= 2;  // log(n)
                it->bias /= 2;
            }
        }

        offset_ += linear_biases_[v] / 2;
        linear_biases_[v] /= 2;
    } else {
        // todo: there are more we could support
        throw std::logic_error("unsupported vartype change");
    }
}

template <class bias_type, class index_type>
void QuadraticModelBase<bias_type, index_type>::change_vartypes(Vartype source, Vartype target) {
    if (source == target) return;  // nothing to do

    bias_type lin_mp, lin_offset_mp, quad_mp, quad_offset_mp, lin_quad_mp;
    if (source == Vartype::SPIN && target == Vartype::BINARY) {
        lin_mp = 2;
        lin_offset_mp = -1;
        quad_mp = 4;
        lin_quad_mp = -2;
        quad_offset_mp = .5;
    } else if (source == Vartype::BINARY && target == Vartype::SPIN) {
        lin_mp = .5;
        lin_offset_mp = .5;
        quad_mp = .25;
        lin_quad_mp = .25;
        quad_offset_mp = .125;
    } else {
        throw std::logic_error("unsupported vartype combo");
    }

    for (size_type u = 0; u < num_variables(); ++u) {
        bias_type lbias = linear_biases_[u];

        linear_biases_[u] *= lin_mp;
        offset_ += lin_offset_mp * lbias;

        if (has_adj()) {
            for (auto it = (*adj_ptr_)[u].begin(); it != (*adj_ptr_)[u].end(); ++it) {
                bias_type qbias = it->bias;

                it->bias *= quad_mp;
                linear_biases_[u] += lin_quad_mp * qbias;
                offset_ += quad_offset_mp * qbias;
            }
        }
    }
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
                               typename std::iterator_traits<Iter>::iterator_category>::value, "iterators must be random access");

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
    remove_variable(v);
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
        if (this->vartype(v) != other.vartype(v)) {
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
                count += n.capacity() * sizeof(LinearTerm<bias_type, index_type>);
            }
        } else {
            for (const auto& n : (*adj_ptr_)) {
                count += n.size() * sizeof(LinearTerm<bias_type, index_type>);
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
        switch (this->vartype(u)) {
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
                asym_quadratic(u, v) = bias;
                break;
            }
        }
    } else {
        asym_quadratic(u, v) = bias;
        asym_quadratic(v, u) = bias;
    }
}

}  // namespace abc
}  // namespace dimod
