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
#include <stdexcept>
#include <utility>
#include <vector>

namespace dimod {

enum Vartype { BINARY, SPIN, INTEGER };

template <class Bias, class Neighbor>
class Neighborhood {
 public:
    using bias_type = Bias;
    using neighbor_type = Neighbor;
    using size_type = std::size_t;

    struct iterator {
        // this could be random access but let's keep it simple for now
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = std::pair<const neighbor_type&, bias_type&>;
        using pointer = value_type*;
        using reference = value_type;

        using bias_iterator = typename std::vector<bias_type>::iterator;
        using neighbor_iterator =
                typename std::vector<neighbor_type>::const_iterator;

        iterator(neighbor_iterator nit, bias_iterator bit)
                : neighbor_it_(nit), bias_it_(bit) {}

        iterator& operator++() {
            bias_it_++;
            neighbor_it_++;
            return *this;
        }

        iterator operator++(int) {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const iterator& a, const iterator& b) {
            return a.neighbor_it_ == b.neighbor_it_ && a.bias_it_ == b.bias_it_;
        }

        friend bool operator!=(const iterator& a, const iterator& b) {
            return a.neighbor_it_ != b.neighbor_it_ && a.bias_it_ != b.bias_it_;
        }

        value_type operator*() { return value_type{*neighbor_it_, *bias_it_}; }

     private:
        bias_iterator bias_it_;
        neighbor_iterator neighbor_it_;

        friend class Neighborhood;
    };

    struct const_iterator {
        // this could be random access but let's keep it simple for now
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = std::pair<const neighbor_type&, const bias_type&>;
        using pointer = value_type*;
        using reference = value_type;

        using bias_iterator = typename std::vector<bias_type>::const_iterator;
        using neighbor_iterator =
                typename std::vector<neighbor_type>::const_iterator;

        const_iterator() {}

        const_iterator(neighbor_iterator nit, bias_iterator bit)
                : neighbor_it_(nit), bias_it_(bit) {}

        const_iterator& operator++() {
            bias_it_++;
            neighbor_it_++;
            return *this;
        }

        const_iterator operator++(int) {
            const_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const const_iterator& a,
                               const const_iterator& b) {
            return a.neighbor_it_ == b.neighbor_it_ && a.bias_it_ == b.bias_it_;
        }

        friend bool operator!=(const const_iterator& a,
                               const const_iterator& b) {
            return a.neighbor_it_ != b.neighbor_it_ && a.bias_it_ != b.bias_it_;
        }

        const value_type operator*() const {
            return value_type{*neighbor_it_, *bias_it_};
        }

     private:
        bias_iterator bias_it_;
        neighbor_iterator neighbor_it_;

        friend class Neighborhood;
    };

    bias_type at(neighbor_type v) const {
        auto it = std::lower_bound(neighbors.begin(), neighbors.end(), v);
        size_type idx = std::distance(neighbors.begin(), it);
        if (it != neighbors.end() && (*it) == v) {
            // it exists
            return quadratic_biases[idx];
        } else {
            // it doesn't exist
            throw std::out_of_range("given variables have no interaction");
        }
    }

    iterator begin() {
        return iterator{neighbors.cbegin(), quadratic_biases.begin()};
    }

    iterator end() {
        return iterator{neighbors.cend(), quadratic_biases.end()};
    }

    const_iterator cbegin() const {
        return const_iterator{neighbors.cbegin(), quadratic_biases.cbegin()};
    }

    const_iterator cend() const {
        return const_iterator{neighbors.cend(), quadratic_biases.cend()};
    }

    void emplace_back(neighbor_type v, bias_type bias) {
        neighbors.push_back(v);
        quadratic_biases.push_back(bias);
    }

    size_type erase(neighbor_type v) {
        auto it = std::lower_bound(neighbors.begin(), neighbors.end(), v);
        if (it != neighbors.end() && (*it) == v) {
            // is is there to erase
            size_type idx = std::distance(neighbors.begin(), it);

            neighbors.erase(it);
            quadratic_biases.erase(quadratic_biases.begin() + idx);

            return 1;
        } else {
            return 0;
        }
    }

    void erase(iterator first, iterator last) {
        quadratic_biases.erase(first.bias_it_, last.bias_it_);
        neighbors.erase(first.neighbor_it_, last.neighbor_it_);
    }

    /// return an iterator to the first element that does not come before v
    iterator lower_bound(neighbor_type v) {
        auto it = std::lower_bound(neighbors.begin(), neighbors.end(), v);
        return iterator{it, quadratic_biases.begin() +
                                    std::distance(neighbors.begin(), it)};
    }

    // returns default if missing without inserting
    bias_type get(neighbor_type v, bias_type value = 0) const {
        auto it = std::lower_bound(neighbors.begin(), neighbors.end(), v);
        size_type idx = std::distance(neighbors.begin(), it);
        if (it != neighbors.end() && (*it) == v) {
            // it exists
            return quadratic_biases[idx];
        } else {
            // it doesn't exist
            return value;
        }
    }

    size_type size() const { return neighbors.size(); }

    // inserts if missing
    bias_type& operator[](neighbor_type v) {
        auto it = std::lower_bound(neighbors.begin(), neighbors.end(), v);
        size_type idx = std::distance(neighbors.begin(), it);
        if (it == neighbors.end() || (*it) != v) {
            // it doesn't exist so insert
            neighbors.insert(it, v);
            quadratic_biases.insert(quadratic_biases.begin() + idx, 0);
        }

        return quadratic_biases[idx];
    }

 protected:
    std::vector<neighbor_type> neighbors;
    std::vector<bias_type> quadratic_biases;
};

template <class Bias, class Neighbor = std::size_t>
class QuadraticModelBase {
 public:
    using bias_type = Bias;
    using neighbor_type = Neighbor;
    using size_type = std::size_t;

    using const_neighborhood_iterator =
            typename Neighborhood<bias_type, neighbor_type>::const_iterator;

    QuadraticModelBase() : offset_(0) {}

    /// Return True if the model has no quadratic biases.
    bool is_linear() const {
        for (auto it = adj_.begin(); it != adj_.end(); ++it) {
            if ((*it).size()) {
                return false;
            }
        }
        return true;
    }

    /**
     * Return the energy of the given sample.
     *
     * @param sample_start A random access iterator pointing to the beginning
     *     of the sample.
     * @return The energy of the given sample.
     *
     * The behavior of this function is undefined when the sample is not
     * <num_variables>"()" long.
     */
    template <class Iter>  // todo: allow different return types
    bias_type energy(Iter sample_start) {
        bias_type en = offset();

        for (size_type u = 0; u < num_variables(); ++u) {
            auto u_val = *(sample_start + u);

            en += u_val * linear(u);

            auto span = neighborhood(u);
            for (auto nit = span.first; nit != span.second && (*nit).first < u;
                 ++nit) {
                auto v_val = *(sample_start + (*nit).first);
                auto bias = (*nit).second;
                en += u_val * v_val * bias;
            }
        }

        return en;
    }

    /**
     * Return the energy of the given sample.
     *
     * @param sample A vector containing the sample.
     * @return The energy of the given sample.
     *
     * The behavior of this function is undefined when the sample is not
     * <num_variables>"()" long.
     */
    template <class T>
    bias_type energy(const std::vector<T>& sample) {
        // todo: check length?
        return energy(sample.cbegin());
    }

    /// Return a reference to the linear bias associated with `v`.
    bias_type& linear(size_type v) { return linear_biases_[v]; }

    /// Return a reference to the linear bias associated with `v`.
    const bias_type& linear(size_type v) const { return linear_biases_[v]; }

    /// Return a pair of iterators - the start and end of the neighborhood
    std::pair<const_neighborhood_iterator, const_neighborhood_iterator>
    neighborhood(size_type u) const {
        return std::make_pair(adj_[u].cbegin(), adj_[u].cend());
    }

    /**
     * Return the quadratic bias associated with `u`, `v`.
     *
     * Note that this function does not return a reference, this is because
     * each quadratic bias is stored twice.
     *
     * @param u A variable in the binary quadratic model.
     * @param v A variable in the binary quadratic model.
     * @return The quadratic bias if it exists, otherwise 0.
     */
    bias_type quadratic(size_type u, neighbor_type v) const {
        return adj_[u].get(v);
    }

    /**
     * Return the quadratic bias associated with `u`, `v`.
     *
     * Note that this function does not return a reference, this is because
     * each quadratic bias is stored twice.
     *
     * @param u A variable in the binary quadratic model.
     * @param v A variable in the binary quadratic model.
     * @return The quadratic bias if it exists, otherwise 0.
     * @exception out_of_range If either `u` or `v` are not variables or if
     *     they do not have an interaction then the function throws an
     *     exception.
     */
    bias_type quadratic_at(size_type u, size_type v) const {
        return adj_[u].at(v);
    }

    /// Return the number of variables in the quadratic model.
    size_type num_variables() const { return linear_biases_.size(); }

    /// Return the number of interactions in the quadratic model.
    size_type num_interactions() const {
        size_type count = 0;

        for (auto it = adj_.begin(); it != adj_.end(); ++it) {
            count += (*it).size();
        }

        return count / 2;
    }

    /// The number of other variables `v` interacts with.
    size_type num_interactions(size_type v) const { return adj_[v].size(); }

    /// Return a reference to the offset
    bias_type& offset() { return offset_; }

    /// Return a reference to the offset
    const bias_type& offset() const { return offset_; }

    /// Remove the interaction if it exists
    bool remove_interaction(size_type u, size_type v) {
        if (adj_[u].erase(v)) {
            return adj_[v].erase(u);  // should always be true
        } else {
            return false;
        }
    }

 protected:
    std::vector<bias_type> linear_biases_;
    std::vector<Neighborhood<bias_type, neighbor_type>> adj_;

    bias_type offset_;
};

/*
 * A binary quadratic model.
 */
template <class Bias, class Neighbor = std::size_t>
class BinaryQuadraticModel : public QuadraticModelBase<Bias, Neighbor> {
 private:
    using base_type = QuadraticModelBase<Bias, Neighbor>;

 public:
    using bias_type = typename base_type::bias_type;
    using size_type = typename base_type::size_type;
    using neighbor_type = typename base_type::neighbor_type;

    BinaryQuadraticModel() : base_type(), vartype_(Vartype::BINARY) {}

    explicit BinaryQuadraticModel(Vartype vartype)
            : base_type(), vartype_(vartype) {}

    BinaryQuadraticModel(size_type n, Vartype vartype)
            : BinaryQuadraticModel(vartype) {
        resize(n);
    }

    template <class T>
    BinaryQuadraticModel(const T dense[], size_type num_variables,
                         Vartype vartype)
            : BinaryQuadraticModel(num_variables, vartype) {
        add_quadratic(dense, num_variables);
    }

    void add_quadratic(size_type u, size_type v, bias_type bias) {
        if (u == v) {
            if (vartype_ == Vartype::BINARY) {
                base_type::linear(u) += bias;
            } else if (vartype_ == Vartype::SPIN) {
                base_type::offset_ += bias;
            } else {
                throw std::logic_error("unknown vartype");
            }
        } else {
            base_type::adj_[u][v] += bias;
            base_type::adj_[v][u] += bias;
        }
    }

    /*
     * Add quadratic biases to the BQM from a dense matrix.
     *
     * Variables must already be present.
     */
    template <class T>
    void add_quadratic(const T dense[], size_type num_variables) {
        // todo: let users add quadratic off the diagonal with row_offset,
        // col_offset

        assert(num_variables <= base_type::num_variables());

        bool sort_needed = !base_type::is_linear();  // do we need to sort after

        bias_type qbias;
        for (size_type u = 0; u < num_variables; ++u) {
            for (size_type v = u + 1; v < num_variables; ++v) {
                qbias = dense[u * num_variables + v] +
                        dense[v * num_variables + u];

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
            for (size_type v = 0; v < num_variables; ++v) {
                base_type::offset_ += dense[v * (num_variables + 1)];
            }
        } else if (vartype_ == Vartype::BINARY) {
            // diagonal is added as linear biases since 1*1 == 1, 0*0 == 0
            for (size_type v = 0; v < num_variables; ++v) {
                base_type::linear_biases_[v] += dense[v * (num_variables + 1)];
            }
        } else {
            throw std::logic_error("bad vartype");
        }
    }

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

    /// Resize the binary quadratic model to contain n variables.
    void resize(size_type n) {
        if (n < base_type::num_variables()) {
            // Clean out any of the to-be-deleted variables from the
            // neighborhoods.
            // This approach is better in the dense case. In the sparse case
            // we could determine which neighborhoods need to be trimmed rather
            // than just doing them all.
            for (size_type v = 0; v < n; ++v) {
                base_type::adj_[v].erase(base_type::adj_[v].lower_bound(n),
                                         base_type::adj_[v].end());
            }
        }

        base_type::linear_biases_.resize(n);
        base_type::adj_.resize(n);
    }

    void set_quadratic(size_type u, size_type v, bias_type bias) {
        if (u == v) {
            throw std::logic_error("not implemented yet");
        } else {
            base_type::adj_[u][v] = bias;
            base_type::adj_[v][u] = bias;
        }
    }

    /// Return the vartype of the binary quadratic model.
    const Vartype& vartype() const { return vartype_; }

    /// Return the vartype of `v`.
    const Vartype& vartype(size_type v) const { return vartype_; }

 private:
    Vartype vartype_;
};

template <class B, class N>
std::ostream& operator<<(std::ostream& os,
                         const BinaryQuadraticModel<B, N>& bqm) {
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
        for (auto nit = span.first; nit != span.second && (*nit).first < u;
             ++nit) {
            os << "    " << u << " " << (*nit).first << " " << (*nit).second
               << "\n";
        }
    }

    return os;
}

}  // namespace dimod
