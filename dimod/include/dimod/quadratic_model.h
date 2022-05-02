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

#include "dimod/iterators.h"
#include "dimod/utils.h"

namespace dimod {

/// Encode the domain of a variable.
enum Vartype {
    BINARY,   ///< Variables that are either 0 or 1.
    SPIN,     ///< Variables that are either -1 or 1.
    INTEGER,  ///< Variables that are integer valued.
    REAL      ///< Variables that are real valued.
};

/// Compile-time limits by variable type.
template <class Bias, Vartype vartype>
class vartype_limits {};

template <class Bias>
class vartype_limits<Bias, Vartype::BINARY> {
 public:
    static constexpr Bias default_max() noexcept { return 1; }
    static constexpr Bias default_min() noexcept { return 0; }
    static constexpr Bias max() noexcept { return 1; }
    static constexpr Bias min() noexcept { return 0; }
};

template <class Bias>
class vartype_limits<Bias, Vartype::SPIN> {
 public:
    static constexpr Bias default_max() noexcept { return 1; }
    static constexpr Bias default_min() noexcept { return -1; }
    static constexpr Bias max() noexcept { return +1; }
    static constexpr Bias min() noexcept { return -1; }
};

template <class Bias>
class vartype_limits<Bias, Vartype::INTEGER> {
 public:
    static constexpr Bias default_max() noexcept { return max(); }
    static constexpr Bias default_min() noexcept { return 0; }
    static constexpr Bias max() noexcept {
        return ((std::int64_t)1 << (std::numeric_limits<Bias>::digits)) - 1;
    }
    static constexpr Bias min() noexcept { return -max(); }
};
template <class Bias>
class vartype_limits<Bias, Vartype::REAL> {
 public:
    static constexpr Bias default_max() noexcept { return max(); }
    static constexpr Bias default_min() noexcept { return 0; }
    static constexpr Bias max() noexcept { return 1e30; }
    static constexpr Bias min() noexcept { return -1e30; }
};

/// Runtime limits by variable type.
template <class Bias>
class vartype_info {
 public:
    static Bias default_max(Vartype vartype) {
        if (vartype == Vartype::BINARY) {
            return vartype_limits<Bias, Vartype::BINARY>::default_max();
        } else if (vartype == Vartype::SPIN) {
            return vartype_limits<Bias, Vartype::SPIN>::default_max();
        } else if (vartype == Vartype::INTEGER) {
            return vartype_limits<Bias, Vartype::INTEGER>::default_max();
        } else if (vartype == Vartype::REAL) {
            return vartype_limits<Bias, Vartype::REAL>::default_max();
        } else {
            throw std::logic_error("unknown vartype");
        }
    }
    static Bias default_min(Vartype vartype) {
        if (vartype == Vartype::BINARY) {
            return vartype_limits<Bias, Vartype::BINARY>::default_min();
        } else if (vartype == Vartype::SPIN) {
            return vartype_limits<Bias, Vartype::SPIN>::default_min();
        } else if (vartype == Vartype::INTEGER) {
            return vartype_limits<Bias, Vartype::INTEGER>::default_min();
        } else if (vartype == Vartype::REAL) {
            return vartype_limits<Bias, Vartype::REAL>::default_min();
        } else {
            throw std::logic_error("unknown vartype");
        }
    }
    static Bias max(Vartype vartype) {
        if (vartype == Vartype::BINARY) {
            return vartype_limits<Bias, Vartype::BINARY>::max();
        } else if (vartype == Vartype::SPIN) {
            return vartype_limits<Bias, Vartype::SPIN>::max();
        } else if (vartype == Vartype::INTEGER) {
            return vartype_limits<Bias, Vartype::INTEGER>::max();
        } else if (vartype == Vartype::REAL) {
            return vartype_limits<Bias, Vartype::REAL>::max();
        } else {
            throw std::logic_error("unknown vartype");
        }
    }
    static Bias min(Vartype vartype) {
        if (vartype == Vartype::BINARY) {
            return vartype_limits<Bias, Vartype::BINARY>::min();
        } else if (vartype == Vartype::SPIN) {
            return vartype_limits<Bias, Vartype::SPIN>::min();
        } else if (vartype == Vartype::INTEGER) {
            return vartype_limits<Bias, Vartype::INTEGER>::min();
        } else if (vartype == Vartype::REAL) {
            return vartype_limits<Bias, Vartype::REAL>::min();
        } else {
            throw std::logic_error("unknown vartype");
        }
    }
};

/**
 * Used internally by QuadraticModelBase to sparsely encode the neighborhood of
 * a variable.
 *
 * Internally, Neighborhoods keep two vectors, one of neighbors and the other
 * of biases. Neighborhoods are designed to make access more like a standard
 * library map.
 */
template <class Bias, class Index>
class Neighborhood {
 public:
    /// The first template parameter (Bias).
    using bias_type = Bias;

    /// The second template parameter (Index).
    using index_type = Index;

    /// Unsigned integral type that can represent non-negative values.
    using size_type = std::size_t;

    /// Exactly `pair<index_type, bias_type>`.
    using value_type = typename std::pair<index_type, bias_type>;

    /// A random access iterator to `pair<index_type, bias_type>`.
    using iterator = typename std::vector<value_type>::iterator;

    /// A random access iterator to `const pair<index_type, bias_type>.`
    using const_iterator = typename std::vector<value_type>::const_iterator;

    /**
     * Return a reference to the bias associated with `v`.
     *
     * This function automatically checks whether `v` is a variable in the
     * neighborhood and throws a `std::out_of_range` exception if it is not.
     */
    bias_type at(index_type v) const {
        auto it = this->lower_bound(v);
        if (it != this->cend() && it->first == v) {
            // it exists
            return it->second;
        } else {
            // it doesn't exist
            throw std::out_of_range("given variable has no interaction");
        }
    }

    /// Returns an iterator to the beginning.
    iterator begin() { return this->neighborhood_.begin(); }

    /// Returns an iterator to the end.
    iterator end() { return this->neighborhood_.end(); }

    /// Returns a const_iterator to the beginning.
    const_iterator cbegin() const { return this->neighborhood_.cbegin(); }

    /// Returns a const_iterator to the end.
    const_iterator cend() const { return this->neighborhood_.cend(); }

    /**
     * Insert a neighbor, bias pair at the end of the neighborhood.
     *
     * Note that this does not keep the neighborhood self-consistent and should
     * only be used when you know that the neighbor is greater than the current
     * last element.
     */
    void emplace_back(index_type v, bias_type bias) { this->neighborhood_.emplace_back(v, bias); }

    /**
     * Erase an element from the neighborhood.
     *
     * Returns the number of element removed, either 0 or 1.
     */
    size_type erase(index_type v) {
        auto it = this->lower_bound(v);
        if (it != this->end() && it->first == v) {
            // is there to erase
            this->neighborhood_.erase(it);
            return 1;
        } else {
            return 0;
        }
    }

    /// Erase elements from the neighborhood.
    void erase(iterator first, iterator last) { this->neighborhood_.erase(first, last); }

    /// Return an iterator to the first element that does not come before `v`.
    iterator lower_bound(index_type v) {
        return std::lower_bound(this->begin(), this->end(), v, this->cmp);
    }

    /// Return an iterator to the first element that does not come before `v`.
    const_iterator lower_bound(index_type v) const {
        return std::lower_bound(this->cbegin(), this->cend(), v, this->cmp);
    }

    /**
     * Total bytes consumed by the biases and indices.
     *
     * If `capacity` is true, use the capacity of the underlying vectors rather
     * than the size.
     */
    size_type nbytes(bool capacity = false) const noexcept {
        // so there is no guaruntee that the compiler will not implement
        // pair as pointers or whatever, but this seems like a reasonable
        // assumption.
        if (capacity) {
            return this->neighborhood_.capacity() * sizeof(std::pair<index_type, bias_type>);
        } else {
            return this->neighborhood_.size() * sizeof(std::pair<index_type, bias_type>);
        }
    }

    /**
     * Return the bias at neighbor `v` or the default value.
     *
     * Return the bias of `v` if `v` is in the neighborhood, otherwise return
     * the `value` provided without inserting `v`.
     */
    bias_type get(index_type v, bias_type value = 0) const {
        auto it = this->lower_bound(v);

        if (it != this->cend() && it->first == v) {
            // it exists
            return it->second;
        } else {
            // it doesn't exist
            return value;
        }
    }

    /// Request that the neighborhood capacity be at least enough to contain `n`
    /// elements.
    void reserve(index_type n) { this->neighborhood_.reserve(n); }

    /// Return the size of the neighborhood.
    size_type size() const { return this->neighborhood_.size(); }

    /// Sort the neighborhood and sum the biases of duplicate variables.
    void sort_and_sum() {
        if (!std::is_sorted(this->begin(), this->end())) {
            std::sort(this->begin(), this->end());
        }

        // now remove any duplicates, summing the biases of duplicates
        size_type i = 0;
        size_type j = 1;

        // walk quickly through the neighborhood until we find a duplicate
        while (j < this->neighborhood_.size() &&
               this->neighborhood_[i].first != this->neighborhood_[j].first) {
            ++i;
            ++j;
        }

        // if we found one, move into de-duplication
        while (j < this->neighborhood_.size()) {
            if (this->neighborhood_[i].first == this->neighborhood_[j].first) {
                this->neighborhood_[i].second += this->neighborhood_[j].second;
                ++j;
            } else {
                ++i;
                this->neighborhood_[i] = this->neighborhood_[j];
                ++j;
            }
        }

        // finally resize to contain only the unique values
        this->neighborhood_.resize(i + 1);
    }

    /**
     * Access the bias of `v`.
     *
     * If `v` is in the neighborhood, the function returns a reference to
     * its bias. If `v` is not in the neighborhood, it is inserted and a
     * reference is returned to its bias.
     */
    bias_type& operator[](index_type v) {
        auto it = this->lower_bound(v);
        if (it == this->end() || it->first != v) {
            // it doesn't exist so insert
            it = this->neighborhood_.emplace(it, v, 0);
        }
        return it->second;
    }

 protected:
    std::vector<value_type> neighborhood_;

    static inline bool cmp(value_type ub, index_type v) { return ub.first < v; }
};

template <class Bias, class Index = int>
class QuadraticModelBase {
 public:
    /// The first template parameter (Bias)
    using bias_type = Bias;

    /// The second template parameter (Index).
    using index_type = Index;

    /// Unsigned integral type that can represent non-negative values.
    using size_type = std::size_t;

    /// A random access iterator to `pair<const index_type&, const bias_type&>`
    using const_neighborhood_iterator =
            typename Neighborhood<bias_type, index_type>::const_iterator;

    /// A forward iterator pointing to the quadratic biases.
    using const_quadratic_iterator = ConstQuadraticIterator<Bias, Index>;

    template <class B, class I>
    friend class BinaryQuadraticModel;

    QuadraticModelBase() : offset_(0) {}

    /// Add quadratic bias for the given variables.
    void add_quadratic(index_type u, index_type v, bias_type bias) {
        assert(0 <= u && static_cast<size_t>(u) < this->num_variables());
        assert(0 <= v && static_cast<size_t>(v) < this->num_variables());

        this->adj_[u][v] += bias;
        if (u != v) {
            this->adj_[v][u] += bias;
        }
    }

    /// Return True if the model has no quadratic biases.
    bool is_linear() const {
        for (auto it = adj_.begin(); it != adj_.end(); ++it) {
            if ((*it).size()) {
                return false;
            }
        }
        return true;
    }

    const_quadratic_iterator cbegin_quadratic() const { return const_quadratic_iterator(this, 0); }

    const_quadratic_iterator cend_quadratic() const {
        return const_quadratic_iterator(this, this->num_variables());
    }

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
    bias_type energy(Iter sample_start) {
        bias_type en = offset();

        for (index_type u = 0; u < static_cast<index_type>(num_variables()); ++u) {
            auto u_val = *(sample_start + u);

            en += u_val * linear(u);

            auto span = neighborhood(u);
            for (auto nit = span.first; nit != span.second && (*nit).first <= u; ++nit) {
                auto v_val = *(sample_start + (*nit).first);
                auto bias = (*nit).second;
                en += bias * u_val * v_val;
            }
        }

        return en;
    }

    /**
     * Return the energy of the given sample.
     *
     * The `sample` must be a vector containing the sample.
     *
     * The behavior of this function is undefined when the sample is not
     * `num_variables()` long.
     */
    template <class T>
    bias_type energy(const std::vector<T>& sample) {
        // todo: check length?
        return energy(sample.cbegin());
    }

    /// Return a reference to the linear bias associated with `v`.
    bias_type& linear(index_type v) { return linear_biases_[v]; }

    /// Return a reference to the linear bias associated with `v`.
    const bias_type& linear(index_type v) const { return linear_biases_[v]; }

    /// Return a pair of iterators - the start and end of the neighborhood
    std::pair<const_neighborhood_iterator, const_neighborhood_iterator> neighborhood(
            index_type u) const {
        return std::make_pair(adj_[u].cbegin(), adj_[u].cend());
    }

    /**
     * The neighborhood of variable `v`.
     *
     * @param A variable `v`.
     * @param The neighborhood will start with the first out variable that
     * does not compare less than `start`.
     *
     * @returns A pair of iterators pointing to the start and end of the
     *     neighborhood.
     */
    std::pair<const_neighborhood_iterator, const_neighborhood_iterator> neighborhood(
            index_type u, index_type start) const {
        return std::make_pair(adj_[u].lower_bound(start), adj_[u].cend());
    }

    /**
     * Return the quadratic bias associated with `u`, `v`.
     *
     * If `u` and `v` do not have a quadratic bias, returns 0.
     *
     * Note that this function does not return a reference, this is because
     * each quadratic bias is stored twice.
     *
     */
    bias_type quadratic(index_type u, index_type v) const { return adj_[u].get(v); }

    /**
     * Return the quadratic bias associated with `u`, `v`.
     *
     * Note that this function does not return a reference, this is because
     * each quadratic bias is stored twice.
     *
     * Raises an `out_of_range` error if either `u` or `v` are not variables or
     * if they do not have an interaction then the function throws an exception.
     */
    bias_type quadratic_at(index_type u, index_type v) const { return adj_[u].at(v); }

    /**
     * Total bytes consumed by the biases and indices.
     *
     * If `capacity` is true, use the capacity of the underlying vectors rather
     * than the size.
     */
    size_type nbytes(bool capacity = false) const noexcept {
        size_type count = sizeof(bias_type);  // offset
        if (capacity) {
            count += this->linear_biases_.capacity() * sizeof(bias_type);
        } else {
            count += this->linear_biases_.size() * sizeof(bias_type);
        }
        for (size_type v = 0; v < this->num_variables(); ++v) {
            count += this->adj_[v].nbytes(capacity);
        }
        return count;
    }

    /// Return the number of variables in the quadratic model.
    size_type num_variables() const { return linear_biases_.size(); }

    /**
     *  Return the number of interactions in the quadratic model.
     *
     * `O(num_variables*log(num_variables))` complexity.
     */
    size_type num_interactions() const {
        size_type count = 0;
        for (index_type v = 0; v < static_cast<index_type>(this->num_variables()); ++v) {
            count += this->adj_[v].size();

            // account for self-loops
            auto lb = this->adj_[v].lower_bound(v);
            if (lb != this->adj_[v].cend() && lb->first == v) {
                count += 1;
            }
        }
        return count / 2;
    }

    /// The number of other variables `v` interacts with.
    size_type num_interactions(index_type v) const { return adj_[v].size(); }

    /// Return a reference to the offset
    bias_type& offset() { return offset_; }

    /// Return a reference to the offset
    const bias_type& offset() const { return offset_; }

    /// Remove the interaction if it exists
    bool remove_interaction(index_type u, index_type v) {
        if (adj_[u].erase(v)) {
            if (u != v) {
                adj_[v].erase(u);
            }
            return true;
        } else {
            return false;
        }
    }

    /// Resize model to contain n variables.
    void resize(index_type n) {
        if (n < (index_type)this->num_variables()) {
            // Clean out any of the to-be-deleted variables from the
            // neighborhoods.
            // This approach is better in the dense case. In the sparse case
            // we could determine which neighborhoods need to be trimmed rather
            // than just doing them all.
            for (index_type v = 0; v < n; ++v) {
                this->adj_[v].erase(this->adj_[v].lower_bound(n), this->adj_[v].end());
            }
        }

        this->linear_biases_.resize(n);
        this->adj_.resize(n);
    }

    /// Scale offset, linear biases, and interactions by a factor
    void scale(double scale_factor) {
        // adjust offset
        offset() *= scale_factor;

        // adjust linear biases and quadratic interactions
        for (size_type u = 0; u < num_variables(); u++) {
            linear_biases_[u] *= scale_factor;

            auto begin = adj_[u].begin();
            auto end = adj_[u].end();
            for (auto nit = begin; nit != end; ++nit) {
                (*nit).second *= scale_factor;
            }
        }
    }

    void set_quadratic(index_type u, index_type v, bias_type bias) {
        assert(0 <= u && static_cast<size_t>(u) < this->num_variables());
        assert(0 <= v && static_cast<size_t>(v) < this->num_variables());

        this->adj_[u][v] = bias;
        if (u != v) {
            this->adj_[v][u] = bias;
        }
    }

    /// Exchange the contents of the quadratic model with the contents of `other`.
    void swap(QuadraticModelBase<bias_type, index_type>& other) {
        std::swap(this->linear_biases_, other.linear_biases_);
        std::swap(this->adj_, other.adj_);
        std::swap(this->offset_, other.offset_);
    }

    /// Swap the linear and quadratic biases between two variables.
    void swap_variables(index_type u, index_type v) {
        assert(0 <= u && static_cast<size_t>(u) < this->num_variables());
        assert(0 <= v && static_cast<size_t>(v) < this->num_variables());

        if (u == v) return;  // nothing to do!

        // developer note, this is pretty expensive since we remove an
        // element and then add an element to each of the neighborhoods that
        // touch u or v. We could do it more efficiently by just swapping
        // their values if they are both present, or only moving the elements
        // between u and v if only one is there. But let's use the simple
        // implementation for now.

        // remove any references to u or v in any other neighborhoods (don't
        // worry, we'll put them back)
        for (auto it = this->adj_[u].begin(); it < this->adj_[u].end(); ++it) {
            if (it->first != v) {
                this->adj_[it->first].erase(u);
            }
        }
        for (auto it = this->adj_[v].begin(); it < this->adj_[v].end(); ++it) {
            if (it->first != u) {
                this->adj_[it->first].erase(v);
            }
        }

        // swap the neighborhoods of u and v
        std::swap(this->adj_[u], this->adj_[v]);
        std::swap(this->linear_biases_[u], this->linear_biases_[v]);

        // now put u and v back into their neighbor's neighborhoods (according
        // to their new indices)
        for (auto it = this->adj_[u].begin(); it < this->adj_[u].end(); ++it) {
            if (it->first != v) {
                this->adj_[it->first][u] = it->second;
            }
        }
        for (auto it = this->adj_[v].begin(); it < this->adj_[v].end(); ++it) {
            if (it->first != u) {
                this->adj_[it->first][v] = it->second;
            }
        }

        // finally fix u and v themselves
        if (this->adj_[u].erase(u)) {
            auto bias = this->adj_[v][v];
            this->adj_[u][v] = bias;
            this->adj_[v][u] = bias;
            this->adj_[v].erase(v);
        }
    }

 protected:
    std::vector<bias_type> linear_biases_;
    std::vector<Neighborhood<bias_type, index_type>> adj_;

    bias_type offset_;

    friend class ConstQuadraticIterator<Bias, Index>;
};

/**
 * A Binary Quadratic Model is a quadratic polynomial over binary variables.
 *
 * Internally, BQMs are stored in a vector-of-vectors adjacency format.
 *
 */
template <class Bias, class Index = int>
class BinaryQuadraticModel : public QuadraticModelBase<Bias, Index> {
 public:
    /// The type of the base class.
    using base_type = QuadraticModelBase<Bias, Index>;

    /// The first template parameter (Bias).
    using bias_type = typename base_type::bias_type;

    /// The second template parameter (Index).
    using index_type = Index;

    /// Unsigned integral that can represent non-negative values.
    using size_type = typename base_type::size_type;

    /// Empty constructor. The vartype defaults to `Vartype::BINARY`.
    BinaryQuadraticModel() : base_type(), vartype_(Vartype::BINARY) {}

    /// Create a BQM of the given `vartype`.
    explicit BinaryQuadraticModel(Vartype vartype) : base_type(), vartype_(vartype) {}

    /// Create a BQM with `n` variables of the given `vartype`.
    BinaryQuadraticModel(index_type n, Vartype vartype) : BinaryQuadraticModel(vartype) {
        this->resize(n);
    }

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
        add_quadratic(dense, num_variables);
    }

    /**
     * Add the variables, interactions and biases from another BQM.
     *
     * The size of the updated BQM will be adjusted appropriately.
     *
     * If the other BQM does not have the same vartype, the biases are adjusted
     * accordingly.
     */
    template <class B, class I>
    void add_bqm(const BinaryQuadraticModel<B, I>& bqm) {
        if (bqm.vartype() != this->vartype()) {
            // we could do this without the copy, but for now let's just do
            // it simply
            auto bqm_copy = BinaryQuadraticModel<B, I>(bqm);
            bqm_copy.change_vartype(vartype());
            this->add_bqm(bqm_copy);
            return;
        }

        // make sure we're big enough
        if (bqm.num_variables() > this->num_variables()) {
            this->resize(bqm.num_variables());
        }

        // offset
        this->offset() += bqm.offset();

        // linear
        for (size_type v = 0; v < bqm.num_variables(); ++v) {
            base_type::linear(v) += bqm.linear(v);
        }

        // quadratic
        for (size_type v = 0; v < bqm.num_variables(); ++v) {
            if (bqm.adj_[v].size() == 0) continue;

            this->adj_[v].reserve(this->adj_[v].size() + bqm.adj_[v].size());

            auto span = bqm.neighborhood(v);
            for (auto it = span.first; it != span.second; ++it) {
                this->adj_[v].emplace_back(it->first, it->second);
            }

            this->adj_[v].sort_and_sum();
        }
    }

    /**
     * Add the variables, interactions and biases from another BQM.
     *
     * The size of the updated BQM will be adjusted appropriately.
     *
     * If the other BQM does not have the same vartype, the biases are adjusted
     * accordingly.
     *
     * `mapping` must be a vector at least as long as the given BQM. It
     * should map the variables of `bqm` to new labels.
     */
    template <class B, class I, class T>
    void add_bqm(const BinaryQuadraticModel<B, I>& bqm, const std::vector<T>& mapping) {
        if (bqm.vartype() != this->vartype()) {
            // we could do this without the copy, but for now let's just do
            // it simply
            auto bqm_copy = BinaryQuadraticModel<B, I>(bqm);
            bqm_copy.change_vartype(vartype());
            this->add_bqm(bqm_copy, mapping);
            return;
        }

        // offset
        this->offset() += bqm.offset();

        if (bqm.num_variables() == 0)
            // nothing else to do, other BQM is empty
            return;

        // make sure we're big enough
        T max_v = *std::max_element(mapping.begin(), mapping.begin() + bqm.num_variables());
        if (max_v < 0) {
            throw std::out_of_range("contents of mapping must be non-negative");
        } else if ((size_type)max_v >= this->num_variables()) {
            this->resize(max_v + 1);
        }

        // linear
        for (size_type v = 0; v < bqm.num_variables(); ++v) {
            this->linear(mapping[v]) += bqm.linear(v);
        }

        // quadratic
        for (size_type v = 0; v < bqm.num_variables(); ++v) {
            if (bqm.adj_[v].size() == 0) continue;

            index_type this_v = mapping[v];

            this->adj_[this_v].reserve(this->adj_[this_v].size() + bqm.adj_[v].size());

            auto span = bqm.neighborhood(v);
            for (auto it = span.first; it != span.second; ++it) {
                this->adj_[this_v].emplace_back(mapping[it->first], it->second);
            }

            this->adj_[this_v].sort_and_sum();
        }
    }

    /// Add quadratic bias for the given variables.
    void add_quadratic(index_type u, index_type v, bias_type bias) {
        if (u == v) {
            assert(0 <= u && static_cast<size_t>(u) < this->num_variables());
            if (vartype_ == Vartype::BINARY) {
                base_type::linear(u) += bias;
            } else if (vartype_ == Vartype::SPIN) {
                base_type::offset_ += bias;
            } else {
                throw std::logic_error("unknown vartype");
            }
        } else {
            base_type::add_quadratic(u, v, bias);
        }
    }

    /*
     * Add quadratic biases to the BQM from a dense matrix.
     *
     * `dense` must be an array of length `num_variables^2`.
     *
     * The behavior of this method is undefined when the bqm has fewer than
     * `num_variables` variables.
     *
     * Values on the diagonal are treated differently depending on the variable
     * type.
     * If the BQM is SPIN-valued, then the values on the diagonal are
     * added to the offset.
     * If the BQM is BINARY-valued, then the values on the diagonal are added
     * as linear biases.
     */
    template <class T>
    void add_quadratic(const T dense[], index_type num_variables) {
        // todo: let users add quadratic off the diagonal with row_offset,
        // col_offset

        assert((size_t)num_variables <= base_type::num_variables());

        bool sort_needed = !base_type::is_linear();  // do we need to sort after

        bias_type qbias;
        for (index_type u = 0; u < num_variables; ++u) {
            for (index_type v = u + 1; v < num_variables; ++v) {
                qbias = dense[u * num_variables + v] + dense[v * num_variables + u];

                if (qbias != 0) {
                    base_type::adj_[u].emplace_back(v, qbias);
                    base_type::adj_[v].emplace_back(u, qbias);
                }
            }
        }

        if (sort_needed) {
            throw std::logic_error("not implemented yet");
        }

        // handle the diagonal according to vartype
        if (vartype_ == Vartype::SPIN) {
            // diagonal is added to the offset since -1*-1 == 1*1 == 1
            for (index_type v = 0; v < num_variables; ++v) {
                base_type::offset_ += dense[v * (num_variables + 1)];
            }
        } else if (vartype_ == Vartype::BINARY) {
            // diagonal is added as linear biases since 1*1 == 1, 0*0 == 0
            for (index_type v = 0; v < num_variables; ++v) {
                base_type::linear_biases_[v] += dense[v * (num_variables + 1)];
            }
        } else {
            throw std::logic_error("bad vartype");
        }
    }

    /**
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
        // determine the number of variables so we can resize ourself if needed
        if (length > 0) {
            index_type max_label = std::max(*std::max_element(row_iterator, row_iterator + length),
                                            *std::max_element(col_iterator, col_iterator + length));

            if ((size_t)max_label >= base_type::num_variables()) {
                this->resize(max_label + 1);
            }
        } else if (length < 0) {
            throw std::out_of_range("length must be positive");
        }

        // count the number of elements to be inserted into each
        std::vector<index_type> counts(base_type::num_variables(), 0);
        ItRow rit(row_iterator);
        ItCol cit(col_iterator);
        for (index_type i = 0; i < length; ++i, ++rit, ++cit) {
            if (*rit != *cit) {
                counts[*rit] += 1;
                counts[*cit] += 1;
            }
        }

        // reserve the neighborhoods
        for (size_type i = 0; i < counts.size(); ++i) {
            base_type::adj_[i].reserve(counts[i]);
        }

        // add the values to the neighborhoods, not worrying about order
        rit = row_iterator;
        cit = col_iterator;
        ItBias bit(bias_iterator);
        for (index_type i = 0; i < length; ++i, ++rit, ++cit, ++bit) {
            if (*rit == *cit) {
                // let add_quadratic handle this case based on vartype
                add_quadratic(*rit, *cit, *bit);
            } else {
                base_type::adj_[*rit].emplace_back(*cit, *bit);
                base_type::adj_[*cit].emplace_back(*rit, *bit);
            }
        }

        // finally sort and sum the neighborhoods we touched
        for (size_type i = 0; i < counts.size(); ++i) {
            if (counts[i] > 0) {
                base_type::adj_[i].sort_and_sum();
            }
        }
    }

    /// Add one (disconnected) variable to the BQM and return its index.
    index_type add_variable() {
        index_type vi = this->num_variables();
        this->resize(vi + 1);
        return vi;
    }

    /// Change the vartype of the binary quadratic model
    void change_vartype(Vartype vartype) {
        if (vartype == vartype_) return;  // nothing to do

        bias_type lin_mp, lin_offset_mp, quad_mp, quad_offset_mp, lin_quad_mp;
        if (vartype == Vartype::BINARY) {
            lin_mp = 2;
            lin_offset_mp = -1;
            quad_mp = 4;
            lin_quad_mp = -2;
            quad_offset_mp = .5;
        } else if (vartype == Vartype::SPIN) {
            lin_mp = .5;
            lin_offset_mp = .5;
            quad_mp = .25;
            lin_quad_mp = .25;
            quad_offset_mp = .125;
        } else {
            throw std::logic_error("unexpected vartype");
        }

        for (size_type ui = 0; ui < base_type::num_variables(); ++ui) {
            bias_type lbias = base_type::linear_biases_[ui];

            base_type::linear_biases_[ui] *= lin_mp;
            base_type::offset_ += lin_offset_mp * lbias;

            auto begin = base_type::adj_[ui].begin();
            auto end = base_type::adj_[ui].end();
            for (auto nit = begin; nit != end; ++nit) {
                bias_type qbias = (*nit).second;

                (*nit).second *= quad_mp;
                base_type::linear_biases_[ui] += lin_quad_mp * qbias;
                base_type::offset_ += quad_offset_mp * qbias;
            }
        }

        vartype_ = vartype;
    }

    bias_type lower_bound(index_type v) const {
        return vartype_info<bias_type>::default_min(this->vartype_);
    }

    /**
     *  Return the number of interactions in the quadratic model.
     *
     * `O(num_variables)` complexity.
     */
    size_type num_interactions() const {
        // we can do better than QuadraticModelBase by not needing to account
        // for self-loops
        size_type count = 0;
        for (auto it = this->adj_.begin(); it != this->adj_.end(); ++it) {
            count += it->size();
        }
        return count / 2;
    }

    /// The number of other variables `v` interacts with.
    size_type num_interactions(index_type v) const { return base_type::num_interactions(v); }

    /// Set the quadratic bias for the given variables.
    void set_quadratic(index_type u, index_type v, bias_type bias) {
        if (u == v) {
            // unlike add_quadratic, this is not really well defined for
            // binary quadratic models. I.e. if there is a linear bias, do we
            // overwrite? Or?
            throw std::domain_error("Cannot set the quadratic bias of a variable with itself");
        } else {
            base_type::set_quadratic(u, v, bias);
        }
    }

    /// Exchange the contents of the binary quadratic model with the contents of `other`.
    void swap(BinaryQuadraticModel<bias_type, index_type>& other) {
        base_type::swap(other);
        std::swap(this->vartype_, other.vartype_);
    }

    bias_type upper_bound(index_type v) const {
        return vartype_info<bias_type>::default_max(this->vartype_);
    }

    /// Return the vartype of the binary quadratic model.
    const Vartype& vartype() const { return vartype_; }

    /// Return the vartype of `v`.
    const Vartype& vartype(index_type v) const { return vartype_; }

 private:
    Vartype vartype_;
};

template <class T>
struct VarInfo {
    Vartype vartype;
    T lb;
    T ub;

    VarInfo(Vartype vartype, T lb, T ub) : vartype(vartype), lb(lb), ub(ub) {}
};

template <class Bias, class Index = int>
class QuadraticModel : public QuadraticModelBase<Bias, Index> {
 public:
    /// The type of the base class.
    using base_type = QuadraticModelBase<Bias, Index>;

    /// The first template parameter (Bias).
    using bias_type = Bias;

    /// The second template parameter (Index).
    using index_type = Index;

    /// Unsigned integral that can represent non-negative values.
    using size_type = typename base_type::size_type;

    QuadraticModel() : base_type(), varinfo_() {}

    template <class B, class T>
    explicit QuadraticModel(const BinaryQuadraticModel<B, T>& bqm) : base_type(bqm) {
        assert(bqm.vartype() == Vartype::BINARY || bqm.vartype() == Vartype::SPIN);

        auto info = VarInfo<bias_type>(bqm.vartype(), vartype_info<bias_type>::min(bqm.vartype()),
                                       vartype_info<bias_type>::max(bqm.vartype()));

        this->varinfo_.insert(this->varinfo_.begin(), bqm.num_variables(), info);
    }

    void add_quadratic(index_type u, index_type v, bias_type bias) {
        if (u == v) {
            assert(0 <= u && static_cast<size_t>(u) < this->num_variables());
            auto vartype = this->vartype(u);
            if (vartype == Vartype::BINARY) {
                base_type::linear(u) += bias;
            } else if (vartype == Vartype::SPIN) {
                base_type::offset_ += bias;
            } else if (vartype == Vartype::INTEGER || vartype == Vartype::REAL) {
                base_type::add_quadratic(u, u, bias);
            } else {
                throw std::logic_error("unknown vartype");
            }
        } else {
            base_type::add_quadratic(u, v, bias);
        }
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
        this->linear_biases_.resize(v + 1);
        this->adj_.resize(v + 1);

        return v;
    }

    /// Change the vartype of `v`, updating the biases appropriately.
    void change_vartype(Vartype vartype, index_type v) {
        if (vartype == this->vartype(v)) return;  // nothing to do

        if (this->vartype(v) == Vartype::BINARY && vartype == Vartype::SPIN) {
            // binary to spin
            for (auto it = this->adj_[v].begin(); it != this->adj_[v].end(); ++it) {
                this->linear(it->first) += it->second / 2;
                this->adj_[it->first][v] /= 2;  // log(n)
                it->second /= 2;
            }
            this->offset() += this->linear(v) / 2;
            this->linear(v) /= 2;

            this->vartype(v) = Vartype::SPIN;
            this->lower_bound(v) = -1;
            this->upper_bound(v) = +1;
        } else if (this->vartype(v) == Vartype::SPIN && vartype == Vartype::BINARY) {
            // spin to binary
            for (auto it = this->adj_[v].begin(); it != this->adj_[v].end(); ++it) {
                this->linear(it->first) -= it->second;
                this->adj_[it->first][v] *= 2;  // log(n)
                it->second *= 2;
            }
            this->offset() -= this->linear(v);
            this->linear(v) *= 2;

            this->vartype(v) = Vartype::BINARY;
            this->lower_bound(v) = 0;
            this->upper_bound(v) = 1;
        } else if (this->vartype(v) == Vartype::BINARY && vartype == Vartype::INTEGER) {
            // binary to integer
            this->varinfo_[v].vartype = Vartype::INTEGER;
        } else if (this->vartype(v) == Vartype::SPIN && vartype == Vartype::INTEGER) {
            // spin to integer (via spin to binary)
            this->change_vartype(Vartype::BINARY, v);
            this->change_vartype(Vartype::INTEGER, v);
        } else {
            // todo: support integer to real and vice versa, need to figure
            // out how to handle bounds in that case though
            throw std::logic_error("invalid vartype change");
        }
    }

    bias_type& lower_bound(index_type v) { return varinfo_[v].lb; }

    const bias_type& lower_bound(index_type v) const { return varinfo_[v].lb; }

    constexpr bias_type max_integer() { return vartype_limits<bias_type, Vartype::INTEGER>::max(); }

    /**
     * Total bytes consumed by the biases, vartype info, bounds, and indices.
     *
     * If `capacity` is true, use the capacity of the underlying vectors rather
     * than the size.
     */
    size_type nbytes(bool capacity = false) const noexcept {
        size_type count = base_type::nbytes(capacity);
        if (capacity) {
            count += this->varinfo_.capacity() * sizeof(VarInfo<bias_type>);
        } else {
            count += this->varinfo_.size() * sizeof(VarInfo<bias_type>);
        }
        return count;
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
        return this->resize(n, Vartype::BINARY, 0, 1);
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

        this->varinfo_.resize(n, VarInfo<bias_type>(vartype, lb, ub));
        base_type::resize(n);
    }

    void set_quadratic(index_type u, index_type v, bias_type bias) {
        if (u == v && (this->vartype(u) == Vartype::SPIN || this->vartype(u) == Vartype::BINARY)) {
            throw std::domain_error(
                    "Cannot set the quadratic bias of a spin or binary "
                    "variable with itself");
        } else {
            base_type::set_quadratic(u, v, bias);
        }
    }

    bias_type& upper_bound(index_type v) { return varinfo_[v].ub; }

    const bias_type& upper_bound(index_type v) const { return varinfo_[v].ub; }

    Vartype& vartype(index_type v) { return varinfo_[v].vartype; }

    const Vartype& vartype(index_type v) const { return varinfo_[v].vartype; }

    void swap(QuadraticModel<bias_type, index_type>& other) {
        base_type::swap(other);
        std::swap(this->varinfo_, other.varinfo_);
    }

    /// Exchange the contents of the quadratic model with the contents of `other`.
    void swap_variables(index_type u, index_type v) {
        base_type::swap_variables(u, v);  // also handles asserts
        std::swap(this->varinfo_[u], this->varinfo_[v]);
    }

 private:
    std::vector<VarInfo<bias_type>> varinfo_;
};

template <class B, class N>
std::ostream& operator<<(std::ostream& os, const BinaryQuadraticModel<B, N>& bqm) {
    os << "BinaryQuadraticModel\n";

    if (bqm.vartype() == Vartype::SPIN) {
        os << "  vartype: spin\n";
    } else if (bqm.vartype() == Vartype::BINARY) {
        os << "  vartype: binary\n";
    } else {
        os << "  vartype: unkown\n";
    }

    os << "  offset: " << bqm.offset() << "\n";

    os << "  linear (" << bqm.num_variables() << " variables):\n";
    for (size_t v = 0; v < bqm.num_variables(); ++v) {
        auto bias = bqm.linear(v);
        if (bias) {
            os << "    " << v << " " << bias << "\n";
        }
    }

    os << "  quadratic (" << bqm.num_interactions() << " interactions):\n";
    for (size_t u = 0; u < bqm.num_variables(); ++u) {
        auto span = bqm.neighborhood(u);
        for (auto nit = span.first; nit != span.second && (*nit).first < u; ++nit) {
            os << "    " << u << " " << (*nit).first << " " << (*nit).second << "\n";
        }
    }

    return os;
}

}  // namespace dimod
