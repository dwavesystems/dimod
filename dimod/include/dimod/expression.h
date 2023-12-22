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
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dimod/abc.h"
#include "dimod/utils.h"
#include "dimod/vartypes.h"

namespace dimod {

// forward declaration
template <class Bias, class Index>
class ConstrainedQuadraticModel;

template <class Bias, class Index = int>
class Expression : public abc::QuadraticModelBase<Bias, Index> {
 public:
    /// The type of the base class.
    using base_type = abc::QuadraticModelBase<Bias, Index>;

    /// The first template parameter (Bias).
    using bias_type = Bias;

    /// The second template parameter (Index).
    using index_type = Index;

    /// Unsigned integer type that can represent non-negative values.
    using size_type = std::size_t;

    using parent_type = ConstrainedQuadraticModel<bias_type, index_type>;

    /// @private  <- don't doc
    class ConstQuadraticIterator {
     public:
        using value_type = abc::TwoVarTerm<bias_type, index_type>;
        using pointer = const value_type*;
        using reference = const value_type&;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        ConstQuadraticIterator() : expression_(nullptr), it_(), end_(), term_(0) {}
        ConstQuadraticIterator(const Expression* expression,
                               const abc::ConstQuadraticIterator<bias_type, index_type>& it,
                               const abc::ConstQuadraticIterator<bias_type, index_type>& end)
                : expression_(expression), it_(it), end_(end), term_(0) {
            if (it_ != end_) {
                term_ = value_type(expression_->variables()[it_->u],
                                   expression_->variables()[it_->v], it_->bias);
            }
        }

        reference operator*() { return term_; }

        const pointer operator->() { return &term_; }

        ConstQuadraticIterator& operator++() {
            ++it_;
            if (it_ != end_) {
                term_ = value_type(expression_->variables()[it_->u],
                                   expression_->variables()[it_->v], it_->bias);
            }
            return *this;
        }

        ConstQuadraticIterator operator++(int) {
            ConstQuadraticIterator tmp(*this);
            ++(*this);
            return tmp;
        }

        friend bool operator==(const ConstQuadraticIterator& a, const ConstQuadraticIterator& b) {
            return a.it_ == b.it_;
        }

        friend bool operator!=(const ConstQuadraticIterator& a, const ConstQuadraticIterator& b) {
            return a.it_ != b.it_;
        }

     private:
        const Expression* expression_;
        abc::ConstQuadraticIterator<bias_type, index_type> it_;
        abc::ConstQuadraticIterator<bias_type, index_type> end_;
        value_type term_;
    };

    using const_quadratic_iterator = ConstQuadraticIterator;

    /// @private  <- don't doc
    class ConstNeighborhoodIterator {
     public:
        using value_type = abc::OneVarTerm<bias_type, index_type>;
        using pointer = const value_type*;
        using reference = const value_type&;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;  // could do random access

        using wrapped_type =
                typename abc::QuadraticModelBase<bias_type,
                                                 index_type>::const_neighborhood_iterator;

        reference operator*() { return term_; }

        const pointer operator->() { return &term_; }

        ConstNeighborhoodIterator() : expression_(nullptr), it_(), end_(), term_(-1, 0) {}
        ConstNeighborhoodIterator(const Expression* expression, const wrapped_type& it,
                                  const wrapped_type& end)
                : expression_(expression), it_(it), end_(end), term_(-1, 0) {
            if (it_ != end_) {
                term_ = value_type(expression_->variables()[it_->v], it_->bias);
            }
        }

        ConstNeighborhoodIterator& operator++() {
            ++it_;
            if (it_ != end_) {
                term_ = value_type(expression_->variables()[it_->v], it_->bias);
            }
            return *this;
        }

        ConstNeighborhoodIterator operator++(int) {
            ConstNeighborhoodIterator tmp(*this);
            ++(*this);
            return tmp;
        }

        friend bool operator==(const ConstNeighborhoodIterator& a,
                               const ConstNeighborhoodIterator& b) {
            return a.it_ == b.it_;
        }
        friend bool operator!=(const ConstNeighborhoodIterator& a,
                               const ConstNeighborhoodIterator& b) {
            return a.it_ != b.it_;
        }

     private:
        const Expression* expression_;
        wrapped_type it_;
        wrapped_type end_;
        value_type term_;
    };

    using const_neighborhood_iterator = ConstNeighborhoodIterator;

    friend class ConstrainedQuadraticModel<bias_type, index_type>;

    Expression();
    explicit Expression(const parent_type* parent);

    Expression(const parent_type* parent, base_type&& other);

    /// Add linear bias to variable ``v``.
    void add_linear(index_type v, bias_type bias);

    void add_quadratic(index_type u, index_type v, bias_type bias);

    void add_quadratic(std::initializer_list<index_type> row, std::initializer_list<index_type> col,
                       std::initializer_list<bias_type> biases);

    template <class ItRow, class ItCol, class ItBias>
    void add_quadratic(ItRow row_iterator, ItCol col_iterator, ItBias bias_iterator,
                       index_type length);

    void add_quadratic_back(index_type u, index_type v, bias_type bias);

    template <class T>
    void add_quadratic_from_dense(const T dense[], index_type num_variables);

    const_neighborhood_iterator cbegin_neighborhood(index_type v) const;

    const_neighborhood_iterator cend_neighborhood(index_type v) const;

    const_quadratic_iterator cbegin_quadratic() const;

    const_quadratic_iterator cend_quadratic() const;

    /// Remove the offset and all variables and interactions from the model. Does not affect parent
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
    template <class Iter>
    bias_type energy(Iter sample_start) const;

    template <class T>
    void fix_variable(index_type v, T assignment);

    /// Check whether u and v have an interaction
    bool has_interaction(index_type u,  index_type v) const;

    bool has_variable(index_type v) const;

    bool is_disjoint(const Expression& other) const;

    template <class B, class I>
    bool is_equal(const abc::QuadraticModelBase<B, I>& other) const;

    /// The linear bias of variable `v`.
    bias_type linear(index_type v) const;

    /// Return the lower bound on variable ``v``.
    bias_type lower_bound(index_type v) const;

    /**
     * Total bytes consumed by the biases and indices.
     *
     * If `capacity` is true, use the capacity of the underlying vectors rather
     * than the size.
     */
    size_type nbytes(bool capacity = false) const;

    using base_type::num_interactions;

    /// The number of other variables `v` interacts with.
    size_type num_interactions(index_type v) const;

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

    void relabel_variables(std::vector<index_type> labels);

    /// Remove the interaction between variables `u` and `v`.
    bool remove_interaction(index_type u, index_type v);

    virtual void remove_variable(index_type v);

    /// Remove several variables from the expression.
    template<class Iter>
    void remove_variables(Iter first, Iter last);

    void remove_variables(const std::vector<index_type>& variables) {
        return remove_variables(variables.begin(), variables.end());
    }

    /// Set the linear bias of variable `v`.
    void set_linear(index_type v, bias_type bias);

    /// Set the linear biases of of the variables beginning with `v`.
    void set_linear(index_type v, std::initializer_list<bias_type> biases);

    /// Set the quadratic bias for the given variables.
    void set_quadratic(index_type u, index_type v, bias_type bias);

    bool shares_variables(const Expression& other) const;

    void substitute_variable(index_type v, bias_type multiplier, bias_type offset);

    /// Return the upper bound on variable ``v``.
    bias_type upper_bound(index_type v) const;

    const std::vector<index_type>& variables() const;

    /// Return the variable type of variable ``v``.
    Vartype vartype(index_type v) const;

 protected:
    /// This removes a variable from the model *and* reindexes
    void reindex_variables(index_type v);

    /// This gets used by base_type to determine how to handle self-loops, etc.
    /// So we want to use the vartype relative to the base_type's indices.
    Vartype vartype_(index_type v) const { return vartype(variables_[v]); }

    const parent_type* parent_;

 private:
    /// List of the parent's label used by expression
    std::vector<index_type> variables_;

    /// Map from parent's labels to the internal ones
    std::unordered_map<index_type, index_type> indices_;  // todo: consider Tessil

    /// Make sure ``v`` exists in the model and return the index in the underlying QM
    index_type enforce_variable(index_type v) {
        auto it = indices_.find(v);
        if (it != indices_.end()) {
            // we're already tracking it
            return it->second;
        }
        // we need to create it
        assert(v >= 0 && static_cast<size_type>(v) < parent_->num_variables());

        index_type vi = variables_.size();
        indices_[v] = vi;
        variables_.emplace_back(v);
        base_type::add_variable();
        return vi;
    }
};

template <class bias_type, class index_type>
Expression<bias_type, index_type>::Expression() : Expression(nullptr) {}

template <class bias_type, class index_type>
Expression<bias_type, index_type>::Expression(const parent_type* parent)
        : base_type(), parent_(parent) {}

template <class bias_type, class index_type>
Expression<bias_type, index_type>::Expression(const parent_type* parent, base_type&& other) {
    throw std::logic_error("not implemented - construction from other");
}

template <class bias_type, class index_type>
void Expression<bias_type, index_type>::add_linear(index_type v, bias_type bias) {
    base_type::add_linear(enforce_variable(v), bias);
}

template <class bias_type, class index_type>
void Expression<bias_type, index_type>::add_quadratic(index_type u, index_type v, bias_type bias) {
    base_type::add_quadratic(enforce_variable(u), enforce_variable(v), bias);
}

template <class bias_type, class index_type>
void Expression<bias_type, index_type>::add_quadratic(std::initializer_list<index_type> row,
                                                      std::initializer_list<index_type> col,
                                                      std::initializer_list<bias_type> biases) {
    throw std::logic_error("not implemented - add_quadratic from initializer_list");
}

template <class bias_type, class index_type>
template <class ItRow, class ItCol, class ItBias>
void Expression<bias_type, index_type>::add_quadratic(ItRow row_iterator, ItCol col_iterator,
                                                      ItBias bias_iterator, index_type length) {
    throw std::logic_error("not implemented - add_quadratic from iterators");
}

template <class bias_type, class index_type>
void Expression<bias_type, index_type>::add_quadratic_back(index_type u, index_type v, bias_type bias) {
    base_type::add_quadratic_back(enforce_variable(u), enforce_variable(v), bias);
}

template <class bias_type, class index_type>
    template <class T>
void Expression<bias_type, index_type>::add_quadratic_from_dense(const T dense[], index_type num_variables) {
    throw std::logic_error("not implemented - add_quadratic_from_dense");
}

template <class bias_type, class index_type>
typename Expression<bias_type, index_type>::const_neighborhood_iterator
Expression<bias_type, index_type>::cbegin_neighborhood(index_type v) const {
    auto it = indices_.find(v);
    if (it == indices_.end()) {
        assert(v >= 0 && static_cast<size_type>(v) < parent_->num_variables());
        auto empty = base_type::empty_neighborhood();
        return const_neighborhood_iterator(this, empty.begin(), empty.end());
    }
    return const_neighborhood_iterator(this, base_type::cbegin_neighborhood(it->second),
                                       base_type::cend_neighborhood(it->second));
}

template <class bias_type, class index_type>
typename Expression<bias_type, index_type>::const_neighborhood_iterator
Expression<bias_type, index_type>::cend_neighborhood(index_type v) const {
    auto it = indices_.find(v);
    if (it == indices_.end()) {
        assert(v >= 0 && static_cast<size_type>(v) < parent_->num_variables());
        auto empty = base_type::empty_neighborhood();
        return const_neighborhood_iterator(this, empty.end(), empty.end());
    }
    return const_neighborhood_iterator(this, base_type::cend_neighborhood(it->second),
                                       base_type::cend_neighborhood(it->second));
}

template <class bias_type, class index_type>
typename Expression<bias_type, index_type>::const_quadratic_iterator
Expression<bias_type, index_type>::cbegin_quadratic() const {
    return const_quadratic_iterator(this, base_type::cbegin_quadratic(),
                                    base_type::cend_quadratic());
}

template <class bias_type, class index_type>
typename Expression<bias_type, index_type>::const_quadratic_iterator
Expression<bias_type, index_type>::cend_quadratic() const {
    return const_quadratic_iterator(this, base_type::cend_quadratic(), base_type::cend_quadratic());
}

template <class bias_type, class index_type>
void Expression<bias_type, index_type>::clear() {
    base_type::clear();
    indices_.clear();
    variables_.clear();
}

template <class bias_type, class index_type>
template <class T>
void Expression<bias_type, index_type>::fix_variable(index_type v, T assignment) {
    assert(v >= 0 && static_cast<size_type>(v) < parent_->num_variables());

    auto vit = indices_.find(v);
    if (vit == indices_.end()) return;  // nothing to remove

    // remove the biases
    base_type::fix_variable(vit->second, assignment);

    // update the indices
    auto it = variables_.erase(variables_.begin() + vit->second);
    indices_.erase(vit);
    for (; it != variables_.end(); ++it) {
        indices_[*it] -= 1;
    }
}

template <class bias_type, class index_type>
bool Expression<bias_type, index_type>::has_interaction(index_type u, index_type v) const {
    auto uit = indices_.find(u);
    auto vit = indices_.find(v);
    if (uit == indices_.end() || vit == indices_.end()) {
        assert(u >= 0 && static_cast<size_type>(u) < parent_->num_variables());
        assert(v >= 0 && static_cast<size_type>(v) < parent_->num_variables());
        return 0;
    }

    return base_type::has_interaction(uit->second, vit->second);
}

template <class bias_type, class index_type>
bool Expression<bias_type, index_type>::has_variable(index_type v) const {
    return indices_.count(v);
}

template <class bias_type, class index_type>
bool Expression<bias_type, index_type>::is_disjoint(const Expression& other) const {
    // we want to do the direction that has the fewest variables
    if (other.num_variables() < base_type::num_variables()) {
        return other.is_disjoint(*this);
    }

    for (const auto& v : variables_) {
        if (other.indices_.count(v)) {
            return false;
        }
    }

    return true;
}

template <class bias_type, class index_type>
template <class B, class I>
bool Expression<bias_type, index_type>:: is_equal(const abc::QuadraticModelBase<B, I>& other) const {
    throw std::logic_error("not implemented - is_equal");
}

template <class bias_type, class index_type>
template <class Iter>
bias_type Expression<bias_type, index_type>::energy(Iter sample_start) const {
    static_assert(std::is_same<std::random_access_iterator_tag,
                               typename std::iterator_traits<Iter>::iterator_category>::value,
                  "iterator must be random access");

    // we could try to do this "virtually" but almost certainly we'll get better
    // performance by just making a new sample in the order of the underlying base type
    std::vector<typename std::iterator_traits<Iter>::value_type> subsample;
    for (const auto& v : variables_) {
        subsample.push_back(*(sample_start + v));
    }

    // now just calculate the energy as normal on the subsample
    return base_type::energy(subsample.begin());
}

template <class bias_type, class index_type>
bias_type Expression<bias_type, index_type>::linear(index_type v) const {
    auto it = indices_.find(v);
    if (it == indices_.end()) {
        assert(v >= 0 && static_cast<size_type>(v) < parent_->num_variables());
        return 0;
    }
    return base_type::linear(it->second);
}

template <class bias_type, class index_type>
bias_type Expression<bias_type, index_type>::lower_bound(index_type v) const {
    return parent_->lower_bound(v);
}

template <class bias_type, class index_type>
bias_type Expression<bias_type, index_type>::quadratic(index_type u, index_type v) const {
    auto uit = indices_.find(u);
    auto vit = indices_.find(v);
    if (uit == indices_.end() || vit == indices_.end()) {
        assert(u >= 0 && static_cast<size_type>(u) < parent_->num_variables());
        assert(v >= 0 && static_cast<size_type>(v) < parent_->num_variables());
        return 0;
    }
    return base_type::quadratic(uit->second, vit->second);
}

template <class bias_type, class index_type>
bias_type Expression<bias_type, index_type>::quadratic_at(index_type u, index_type v) const {
    auto uit = indices_.find(u);
    auto vit = indices_.find(v);
    if (uit == indices_.end() || vit == indices_.end()) {
        throw std::out_of_range("given variables have no interaction");
    }
    return base_type::quadratic_at(uit->second, vit->second);
}

template <class bias_type, class index_type>
typename Expression<bias_type, index_type>::size_type Expression<bias_type, index_type>::nbytes(
        bool capacity) const {
    throw std::logic_error("not implemented - nbytes");
}

template <class bias_type, class index_type>
typename Expression<bias_type, index_type>::size_type
Expression<bias_type, index_type>::num_interactions(index_type v) const {
    auto it = indices_.find(v);
    if (it == indices_.end()) {
        assert(v >= 0 && static_cast<size_type>(v) < parent_->num_variables());
        return 0;
    }
    return base_type::num_interactions(it->second);
}

template <class bias_type, class index_type>
void Expression<bias_type, index_type>::reindex_variables(index_type v) {
    size_type start = variables_.size();  // the start of the indices that need to change

    // see if v is present
    auto it = indices_.find(v);
    if (it != indices_.end()) {
        start = it->second;
        base_type::remove_variable(it->second);
        variables_.erase(variables_.begin() + it->second);
        indices_.erase(it);
    }

    // remove any v/index pairs from indices_ for which v changed
    for (auto& u : variables_) {
        if (u > v) {
            indices_.erase(u);
            --u;
        }
    }

    // update the indices before start
    for (size_type i = 0; i < start; ++i) {
        if (variables_[i] >= v) {
            indices_[variables_[i]] = i;
        }
    }

    // update the indices after start
    for (size_type i = start; i < variables_.size(); ++i) {
        indices_[variables_[i]] = i;
    }

    assert(indices_.size() == variables_.size());
}

template <class bias_type, class index_type>
void Expression<bias_type, index_type>::relabel_variables(std::vector<index_type> labels) {
    assert(labels.size() == base_type::num_variables());

    variables_ = std::move(labels);

    indices_.clear();
    for (size_type ui = 0; ui < variables_.size(); ++ui) {
        indices_[variables_[ui]] = ui;
    }
}


template <class bias_type, class index_type>
bool Expression<bias_type, index_type>::remove_interaction(index_type u, index_type v) {
    auto uit = indices_.find(u);
    auto vit = indices_.find(v);
    if (uit == indices_.end() || vit == indices_.end()) {
        return false;
    }
    return base_type::remove_interaction(uit->second, vit->second);
}

template <class bias_type, class index_type>
void Expression<bias_type, index_type>::remove_variable(index_type v) {
    assert(v >= 0 && static_cast<size_type>(v) < parent_->num_variables());

    auto vit = indices_.find(v);
    if (vit == indices_.end()) return;  // nothing to remove

    // remove the biases
    base_type::remove_variable(vit->second);

    // update the indices
    auto it = variables_.erase(variables_.begin() + vit->second);
    indices_.erase(vit);
    for (; it != variables_.end(); ++it) {
        indices_[*it] -= 1;
    }
}

template <class bias_type, class index_type>
template <class Iter>
void Expression<bias_type, index_type>::remove_variables(Iter first, Iter last) {
    // get the indices of any variables that need to be removed
    std::vector<index_type> to_remove;
    for (auto it = first; it != last; ++it) {
        auto search = indices_.find(*it);
        if (search != indices_.end()) {
            to_remove.emplace_back(search->second);
        }
    }
    std::sort(to_remove.begin(), to_remove.end());

    // remove the indices from variables_ and the underlying
    variables_.erase(utils::remove_by_index(variables_.begin(), variables_.end(), to_remove.begin(),
                                            to_remove.end()),
                     variables_.end());

    // remove the indices from the underlying quadratic model
    base_type::remove_variables(to_remove);

    // finally fix the indices by rebuilding from scratch
    indices_.clear();
    for (size_type i = 0, end = variables_.size(); i < end; ++i) {
        indices_[variables_[i]] = i;
    }
}

template <class bias_type, class index_type>
void Expression<bias_type, index_type>::set_linear(index_type v, bias_type bias) {
    base_type::set_linear(enforce_variable(v), bias);
}

template <class bias_type, class index_type>
void Expression<bias_type, index_type>::set_quadratic(index_type u, index_type v, bias_type bias) {
    base_type::set_quadratic(enforce_variable(u), enforce_variable(v), bias);
}

template <class bias_type, class index_type>
bool Expression<bias_type, index_type>::shares_variables(const Expression& other) const {
    for (auto& v : variables_) {
        if (other.has_variable(v)) return true;  // overlap!
    }
    return false;
}

template <class bias_type, class index_type>
void Expression<bias_type, index_type>::substitute_variable(index_type v, bias_type multiplier,
                                                            bias_type offset) {
    auto it = indices_.find(v);
    if (it == indices_.end()) {
        assert(v >= 0 && static_cast<size_type>(v) < parent_->num_variables());
        return;
    }
    return base_type::substitute_variable(it->second, multiplier, offset);
}

template <class bias_type, class index_type>
bias_type Expression<bias_type, index_type>::upper_bound(index_type v) const {
    return parent_->upper_bound(v);
}

template <class bias_type, class index_type>
const std::vector<index_type>& Expression<bias_type, index_type>::variables() const {
    return variables_;
}

template <class bias_type, class index_type>
Vartype Expression<bias_type, index_type>::vartype(index_type v) const {
    return parent_->vartype(v);
}

}  // namespace dimod
