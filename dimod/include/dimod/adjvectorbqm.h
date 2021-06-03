// Copyright 2020 D-Wave Systems Inc.
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

#ifndef DIMOD_ADJVECTORBQM_H_
#define DIMOD_ADJVECTORBQM_H_

#include <stdio.h>
#include <algorithm>
#include <utility>
#include <vector>

#include "dimod/utils.h"

namespace dimod {

template <class V, class B>
class AdjVectorBQM {
 public:
    using bias_type = B;
    using variable_type = V;
    using size_type = std::size_t;

    using outvars_iterator = typename std::vector<std::pair<V, B>>::iterator;
    using const_outvars_iterator =
            typename std::vector<std::pair<V, B>>::const_iterator;

    // in the future we'd probably like to make this protected
    std::vector<std::pair<std::vector<std::pair<V, B>>, B>> adj;

    AdjVectorBQM() {}

    template <class BQM>
    explicit AdjVectorBQM(const BQM &bqm) {
        adj.resize(bqm.num_variables());

        for (size_type v = 0; v < bqm.num_variables(); ++v) {
            linear(v) = bqm.linear(v);

            auto span = bqm.neighborhood(v);
            adj[v].first.insert(adj[v].first.begin(), span.first, span.second);
        }
    }

    /**
     * Construct a BQM from a dense array.
     *
     * @param dense An array containing the biases. Assumed to contain
     *     `num_variables`^2 elements. The upper and lower triangle are summed.
     * @param num_variables The number of variables.
     */
    template <class B2>
    AdjVectorBQM(const B2 dense[], size_type num_variables,
                 bool ignore_diagonal = false) {
        // we know how big our linear is going to be
        adj.resize(num_variables);

        bias_type qbias;

        if (!ignore_diagonal) {
            for (size_type v = 0; v < num_variables; ++v) {
                adj[v].second = dense[v * (num_variables + 1)];
            }
        }

        for (size_type u = 0; u < num_variables; ++u) {
            for (size_type v = u + 1; v < num_variables; ++v) {
                qbias = dense[u * num_variables + v] +
                        dense[v * num_variables + u];

                if (qbias != 0) {
                    adj[u].first.emplace_back(v, qbias);
                    adj[v].first.emplace_back(u, qbias);
                }
            }
        }
    }

    /**
     * Construct a BQM from a dense array. This constructor is parallelized
     * and temporarily zeroes out the diagonal of the dense array but restores
     * it back.
     *
     * @param dense An array containing the biases. Assumed to contain
     *     `num_variables`^2 elements. The upper and lower triangle are summed.
     * @param num_variables The number of variables.
     */
    template <class B2>
    AdjVectorBQM(B2 dense[], size_type num_variables,
                 bool ignore_diagonal = false) {
        // we know how big our linear is going to be
        adj.resize(num_variables);

        // Backup copy of the diagonal of the dense matrix.
        std::vector<B2> dense_diagonal(num_variables);

        if (!ignore_diagonal) {
#pragma omp parallel for
            for (size_type v = 0; v < num_variables; ++v) {
                adj[v].second = dense[v * (num_variables + 1)];
            }
        }

#pragma omp parallel
        {
// Zero out the diagonal to avoid expensive checks inside innermost
// loop in the code for reading the matrix. The diagonal will be
// restored so a backup copy is saved.
#pragma omp for schedule(static)
            for (size_type v = 0; v < num_variables; ++v) {
                dense_diagonal[v] = dense[v * (num_variables + 1)];
                dense[v * (num_variables + 1)] = 0;
            }

            size_type counters[BLOCK_SIZE] = {0};
            size_type buffer_size = num_variables * BLOCK_SIZE *
                                    sizeof(std::pair<variable_type, bias_type>);
            std::pair<variable_type, bias_type> *temp_buffer =
                    (std::pair<variable_type, bias_type> *)malloc(buffer_size);

            if (temp_buffer == NULL) {
                printf("Memory allocation failure.\n");
                exit(0);
            }

// We process the matrix in blocks of size BLOCK_SIZE*BLOCK_SIZE to take
// advantage of cache locality. Dynamic scheduling is used as  we know some
// blocks may be more sparse than others and processing them may finish earlier.
#pragma omp for schedule(dynamic)
            for (size_type u_st = 0; u_st < num_variables; u_st += BLOCK_SIZE) {
                size_type u_end = std::min(u_st + BLOCK_SIZE, num_variables);
                for (size_type v_st = 0; v_st < num_variables;
                     v_st += BLOCK_SIZE) {
                    size_type v_end =
                            std::min(v_st + BLOCK_SIZE, num_variables);
                    for (size_type u = u_st, n = 0; u < u_end; u++, n++) {
                        size_type counter_u = counters[n];
                        size_type counter_u_old = counter_u;
                        for (size_type v = v_st; v < v_end; v++) {
                            bias_type qbias = dense[u * num_variables + v] +
                                              dense[v * num_variables + u];
                            if (qbias != 0) {
                                temp_buffer[n * num_variables + counter_u++] = {
                                        v, qbias};
                            }
                        }
                        if (counter_u != counter_u_old) {
                            counters[n] = counter_u;
                        }
                    }
                }

                for (size_type n = 0; n < BLOCK_SIZE; n++) {
                    if (counters[n]) {
                        adj[u_st + n].first.assign(
                                temp_buffer + n * num_variables,
                                temp_buffer + n * num_variables + counters[n]);
                        counters[n] = 0;
                    }
                }
            }

            free(temp_buffer);

// Restore the diagonal of the original dense matrix
#pragma omp for schedule(static)
            for (size_type v = 0; v < num_variables; ++v) {
                dense[v * (num_variables + 1)] = dense_diagonal[v];
            }
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
     * @param row_iterator Iterator pointing to the beginning of the row data.
     *     Must be a random access iterator.
     * @param col_iterator Iterator pointing to the beginning of the column
     *     data. Must be a random access iterator.
     * @param bias_iterator Iterator pointing to the beginning of the bias data.
     *     Must be a random access iterator.
     * @param length The number of (row, column, bias) entries.
     * @param ignore_diagonal If true, entries on the diagonal of the sparse
     *     matrix are ignored.
     */
    template <class ItRow, class ItCol, class ItBias>
    AdjVectorBQM(ItRow row_iterator, ItCol col_iterator, ItBias bias_iterator,
                 size_type length, bool ignore_diagonal = false) {
        // determine the number of variables so we can allocate adj
        if (length > 0) {
            size_type max_label = std::max(
                    *std::max_element(row_iterator, row_iterator + length),
                    *std::max_element(col_iterator, col_iterator + length));
            adj.resize(max_label + 1);
        }

        // Count the degrees and use that to reserve the neighborhood vectors
        std::vector<size_type> degrees(adj.size());
        ItRow rit(row_iterator);
        ItCol cit(col_iterator);
        for (size_type i = 0; i < length; ++i, ++rit, ++cit) {
            if (*rit != *cit) {
                degrees[*rit] += 1;
                degrees[*cit] += 1;
            }
        }
        for (size_type i = 0; i < degrees.size(); ++i) {
            adj[i].first.reserve(degrees[i]);
        }

        // add the values to the adjacency, not worrying about order or
        // duplicates
        for (size_type i = 0; i < length; i++) {
            if (*row_iterator == *col_iterator) {
                // linear bias
                if (!ignore_diagonal) {
                    linear(*row_iterator) += *bias_iterator;
                }
            } else {
                // quadratic bias
                adj[*row_iterator].first.emplace_back(*col_iterator,
                                                      *bias_iterator);
                adj[*col_iterator].first.emplace_back(*row_iterator,
                                                      *bias_iterator);
            }

            ++row_iterator;
            ++col_iterator;
            ++bias_iterator;
        }

        normalize_neighborhood();
    }

    /// Add one (disconnected) variable to the BQM and return its index.
    variable_type add_variable() {
        adj.resize(adj.size() + 1);
        return adj.size() - 1;
    }

    /// Get the degree of variable `v`.
    size_type degree(variable_type v) const { return adj[v].first.size(); }

    [[deprecated("Use AdjVectorBQM::linear(v)")]] bias_type get_linear(
            variable_type v) const { return linear(v); }

    std::pair<bias_type, bool> get_quadratic(variable_type u,
                                             variable_type v) const {
        assert(u >= 0 && (size_type)u < adj.size());
        assert(v >= 0 && (size_type)v < adj.size());
        assert(u != v);

        auto span = neighborhood(u);
        auto low = std::lower_bound(span.first, span.second, v,
                                    utils::comp_v<V, B>);

        if (low == span.second || low->first != v)
            return std::make_pair(0, false);
        return std::make_pair(low->second, true);
    }

    bias_type &linear(variable_type v) {
        assert(v >= 0 && (size_type)v < adj.size());
        return adj[v].second;
    }

    const bias_type &linear(variable_type v) const {
        assert(v >= 0 && (size_type)v < adj.size());
        return adj[v].second;
    }

    std::pair<outvars_iterator, outvars_iterator> neighborhood(
            variable_type u) {
        assert(u >= 0 && (size_type)u < adj.size());
        return std::make_pair(adj[u].first.begin(), adj[u].first.end());
    }

    std::pair<const_outvars_iterator, const_outvars_iterator> neighborhood(
            variable_type u) const {
        assert(u >= 0 && (size_type)u < adj.size());
        return std::make_pair(adj[u].first.cbegin(), adj[u].first.cend());
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
    std::pair<const_outvars_iterator, const_outvars_iterator> neighborhood(
            variable_type v, variable_type start) const {
        auto span = neighborhood(v);
        auto low = std::lower_bound(span.first, span.second, start,
                                    utils::comp_v<V, B>);
        return std::make_pair(low, span.second);
    }

    /// sort each neighborhood and merge duplicates
    void normalize_neighborhood() {
        for (size_type v = 0; v < adj.size(); ++v) {
            normalize_neighborhood(v);
        }
    }

    template<class Iter>
    void normalize_neighborhood(Iter begin, Iter end) {
        while (begin != end) {
            normalize_neighborhood(*begin);
            ++begin;
        }
    }

    void normalize_neighborhood(variable_type v) {
        auto span = neighborhood(v);
        if (!std::is_sorted(span.first, span.second)) {
            std::sort(span.first, span.second);
        }

        // now merge any duplicate variables, adding the biases
        auto it = adj[v].first.begin();
        while (it + 1 < adj[v].first.end()) {
            if (it->first == (it + 1)->first) {
                it->second += (it + 1)->second;
                adj[v].first.erase(it + 1);
            } else {
                ++it;
            }
        }
    }

    size_type num_variables() const { return adj.size(); }

    size_type num_interactions() const {
        size_type count = 0;
        for (auto it = adj.begin(); it != adj.end(); ++it)
            count += it->first.size();
        return count / 2;
    }

    variable_type pop_variable() {
        assert(adj.size() > 0);

        variable_type v = adj.size() - 1;

        // remove v from all of its neighbor's neighborhoods
        for (auto it = adj[v].first.cbegin(); it != adj[v].first.cend(); ++it) {
            auto span = neighborhood(it->first);
            auto low = std::lower_bound(span.first, span.second, v,
                                        utils::comp_v<V, B>);
            adj[it->first].first.erase(low);
        }

        adj.pop_back();

        return adj.size();
    }

    bool remove_interaction(variable_type u, variable_type v) {
        assert(u >= 0 && (size_type)u < adj.size());
        assert(v >= 0 && (size_type)v < adj.size());

        auto span = neighborhood(u);
        auto low = std::lower_bound(span.first, span.second, v,
                                    utils::comp_v<V, B>);

        bool exists = !(low == span.second || low->first != v);

        if (exists) {
            adj[u].first.erase(low);

            span = neighborhood(v);
            low = std::lower_bound(span.first, span.second, u,
                                   utils::comp_v<V, B>);

            assert(!(low == span.second || low->first != u) == exists);

            adj[v].first.erase(low);
        }

        return exists;
    }

    [[deprecated("Use AdjVectorBQM::linear(v)")]] void set_linear(
            variable_type v, bias_type b) {
        assert(v >= 0 && (size_type)v < adj.size());
        linear(v) = b;
    }

    bool set_quadratic(variable_type u, variable_type v, bias_type b) {
        assert(u >= 0 && (size_type)u < adj.size());
        assert(v >= 0 && (size_type)v < adj.size());
        assert(u != v);

        auto span = neighborhood(u);
        auto low = std::lower_bound(span.first, span.second, v,
                                    utils::comp_v<V, B>);

        bool exists = !(low == span.second || low->first != v);

        if (exists) {
            low->second = b;
        } else {
            adj[u].first.emplace(low, v, b);
        }

        span = neighborhood(v);
        low = std::lower_bound(span.first, span.second, u, utils::comp_v<V, B>);

        assert(!(low == span.second || low->first != u) == exists);

        if (exists) {
            low->second = b;
        } else {
            adj[v].first.emplace(low, u, b);
        }

        // to be consistent with AdjArrayBQM, we return whether the value was
        // set
        return true;
    }
};
}  // namespace dimod

#endif  // DIMOD_ADJVECTORBQM_H_
