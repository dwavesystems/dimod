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

#ifndef DIMOD_ADJARRAYBQM_H_
#define DIMOD_ADJARRAYBQM_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "dimod/utils.h"

namespace dimod {

template<class V, class B, class N = std::size_t>
class AdjArrayBQM {
 public:
    using bias_type = B;
    using neighborhood_type = N;
    using variable_type = V;
    using size_type = std::size_t;

    using outvars_iterator = typename std::vector<std::pair<V, B>>::iterator;
    using const_outvars_iterator = typename std::vector<std::pair<V, B>>::const_iterator;

    // in the future we'd probably like to make this protected
    std::vector<std::pair<N, B>> invars;
    std::vector<std::pair<V, B>> outvars;

    AdjArrayBQM() {}

    // can we specify this slightly better?
    template<class BQM>
    explicit AdjArrayBQM(const BQM &bqm) {
        invars.reserve(bqm.num_variables());
        outvars.reserve(2*bqm.num_interactions());

        for (auto v = 0; v < bqm.num_variables(); ++v) {
            invars.emplace_back(outvars.size(), bqm.linear(v));

            auto span = bqm.neighborhood(v);
            outvars.insert(outvars.end(), span.first, span.second);
        }
    }

    /**
     * Construct a BQM from a dense array.
     *
     * @param dense An array containing the biases. Assumed to contain
     *     `num_variables`^2 elements. The upper and lower triangle are summed.
     * @param num_variables The number of variables. 
     */
    template<class B2>
    AdjArrayBQM(const B2 dense[], size_type num_variables,
                bool ignore_diagonal = false) {
        // we know how big our linear is going to be. We'd also like to
        // reserve quadratic, but we ignore 0s on the off-digonal so we can't
        invars.reserve(num_variables);

        bias_type qbias;

        for (size_type u = 0; u < num_variables; ++u) {
            // handle the linear
            if (ignore_diagonal) {
                invars.emplace_back(outvars.size(), 0);
            } else {
                invars.emplace_back(outvars.size(), dense[u*(num_variables+1)]);
            }

            for (size_type v = 0; v < num_variables; ++v) {
                if (u == v) continue;  // already did linear

                qbias = dense[u*num_variables+v] + dense[v*num_variables+u];

                if (qbias != 0)
                    outvars.emplace_back(v, qbias);
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
    AdjArrayBQM(B2 dense[], size_type num_variables, bool ignore_diagonal = false) {
        // we know how big our linear is going to be.
        invars.resize(num_variables);

        // Aligned memory is to avoid false sharing between threads.
        size_type* counters_cumsum = (size_type*)utils::aligned_calloc(num_variables, sizeof(size_type));

        // Backup copy of the diagonal of the dense matrix.
        std::vector<B2> dense_diagonal(num_variables);

        #pragma omp parallel
        {
            // Zero out the diagonal to avoid expensive checks inside innermost
            // loop in the code for reading the matrix. The diagonal will be
            // restored so a backup copy is saved.
            #pragma omp for schedule(static)
            for (size_type u = 0; u < num_variables; ++u) {
                dense_diagonal[u] = dense[u * (num_variables + 1)];
                dense[u * (num_variables + 1)] = 0;
            }

            // We process the matrix in two passes, in the first pass we take note
            // of the number of total elements and elements in each row for proper
            // memory allocation. In the second pass we fill up our desired bqm.
            // We process the matrix in blocks of size BLOCK_SIZE*BLOCK_SIZE to take
            // advantage of cache locality. Dynamic scheduling is used as we know some
            // blocks may be more sparse than others and processing them may finish earlier.
            #pragma omp for schedule(dynamic)
            for (size_type u_st = 0; u_st < num_variables; u_st += BLOCK_SIZE) {
                size_type u_end = std::min(u_st + BLOCK_SIZE, num_variables);
                for (size_type v_st = 0; v_st < num_variables; v_st += BLOCK_SIZE) {
                    size_type v_end = std::min(v_st + BLOCK_SIZE, num_variables);
                    for (size_type u = u_st; u < u_end; ++u) {
                        size_type counter_u = counters_cumsum[u];
                        size_type counter_u_old = counter_u;
                        for (size_type v = v_st; v < v_end; ++v) {
                            bias_type qbias = dense[u * num_variables + v] + dense[v * num_variables + u];
                            if (qbias != 0) {
                                counter_u++;
                            }
                        }
                        if (counter_u != counter_u_old) {
                            counters_cumsum[u] = counter_u;
                        }
                    }
                }
            }
        }

        // Calculate the exclusive scan of the counters. Thus it will contain the
        // starting indices in the bqm for inserting elements for each row of the dense
        // matrix. This could be parallelized but has diminishing returns.
        size_type sum_counters = 0;
        for (size_type u = 0; u < num_variables; ++u) {
            size_type prev_counter = counters_cumsum[u];
            counters_cumsum[u] = sum_counters;
            sum_counters += prev_counter;
        }

        // TODO : This is the bottleneck for moderately dense input arrays.
        // stl vector initializes the values during resize. Update it if a
        // resize function is made available that does not do initialization.
        // Moreover we need to pass in an initialization value, as otherwise a
        // constructor for pairs will be called slowing this part by a factor of two.
        outvars.resize(sum_counters, {0, 0});

        if (ignore_diagonal) {
            #pragma omp for schedule(static)
            for (size_type u = 0; u < num_variables; ++u) {
                invars[u] = {counters_cumsum[u], 0};
            }
        } else {
            #pragma omp for schedule(static)
            for (size_type u = 0; u < num_variables; ++u) {
                invars[u] = {counters_cumsum[u], dense_diagonal[u]};
            }
        }

        #pragma omp parallel
        {
            // Now that we have allocated proper amounts of memory as calculated
            // in the previou pass (see above) we cann directly assign the values.
            // Note the array of counters now contains the starting indices for
            // insertion in the bqm for each row of the dense matrix.
            #pragma omp for schedule(dynamic)
            for (size_type u_st = 0; u_st < num_variables; u_st += BLOCK_SIZE) {
                size_type u_end = std::min(u_st + BLOCK_SIZE, num_variables);
                for (size_type v_st = 0; v_st < num_variables; v_st += BLOCK_SIZE) {
                    size_type v_end = std::min(v_st + BLOCK_SIZE, num_variables);
                    for (size_type u = u_st; u < u_end; ++u) {
                        size_type counter_u = counters_cumsum[u];
                        size_type counter_u_old = counter_u;
                        for (size_type v = v_st; v < v_end; ++v) {
                            bias_type qbias = dense[u * num_variables + v] + dense[v * num_variables + u];
                            if (qbias != 0) {
                                outvars[counter_u++] = {v, qbias};
                            }
                        }
                        if (counter_u != counter_u_old) {
                            counters_cumsum[u] = counter_u;
                        }
                    }
                }
            }

            // Restore the diagonal of the original dense matrix
            #pragma omp for schedule(static)
            for (size_type u = 0; u < num_variables; ++u) {
                dense[u * (num_variables + 1)] = dense_diagonal[u];
            }
        }

        utils::aligned_free(counters_cumsum);
    }

    size_type num_interactions() const {
        return outvars.size() / 2;
    }

    size_type num_variables() const {
        return invars.size();
    }

    [[deprecated("Use AdjArrayBQM::linear(v)")]]
    bias_type get_linear(variable_type v) const {
        return linear(v);
    }

    std::pair<bias_type, bool>
    get_quadratic(variable_type u, variable_type v) const {
        assert(u >= 0 && u < invars.size());
        assert(v >= 0 && v < invars.size());
        assert(u != v);

        auto span = neighborhood(u);
        auto low = std::lower_bound(span.first, span.second, v,
                                    utils::comp_v<V, B>);

        if (low == span.second || low->first != v)
            return std::make_pair(0, false);
        return std::make_pair(low->second, true);
    }

    size_type degree(variable_type v) const {
        assert(v >= 0 && v < invars.size());

        // need to check the case that v is the last variable
        if ((unsigned) v == invars.size() - 1)
            return outvars.size() - invars[v].first;

        return invars[v+1].first - invars[v].first;
    }

    bias_type& linear(variable_type v) {
        assert(v >= 0 && v < invars.size());
        return invars[v].second;
    }

    const bias_type& linear(variable_type v) const {
        assert(v >= 0 && v < invars.size());
        return invars[v].second;
    }

    std::pair<outvars_iterator, outvars_iterator>
    neighborhood(variable_type u) {
        assert(u >= 0 && u < invars.size());

        outvars_iterator end;
        if ((unsigned) u == invars.size() - 1) {
            end = outvars.end();
        } else {
            end = outvars.begin() + invars[u+1].first;
        }

        return std::make_pair(outvars.begin() + invars[u].first, end);
    }

    std::pair<const_outvars_iterator, const_outvars_iterator>
    neighborhood(variable_type u) const {
        assert(u >= 0 && u < invars.size());

        const_outvars_iterator end;
        if ((unsigned) u == invars.size() - 1) {
            end = outvars.cend();
        } else {
            end = outvars.cbegin() + invars[u+1].first;
        }

        return std::make_pair(outvars.cbegin() + invars[u].first, end);
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
    std::pair<const_outvars_iterator, const_outvars_iterator>
    neighborhood(variable_type v, variable_type start) const {
        auto span = neighborhood(v);
        auto low = std::lower_bound(span.first, span.second,
                                    start, utils::comp_v<V, B>);
        return std::make_pair(low, span.second);
    } 

    [[deprecated("Use AdjArrayBQM::linear(v)")]]
    void set_linear(variable_type v, bias_type b) {
        assert(v >= 0 && v < invars.size());
        linear(v) = b;
    }

    bool set_quadratic(variable_type u, variable_type v, bias_type b) {
        assert(u >= 0 && u < invars.size());
        assert(v >= 0 && v < invars.size());
        assert(u != v);

        auto span = neighborhood(u);
        auto low = std::lower_bound(span.first, span.second, v,
                                    utils::comp_v<V, B>);

        // if u, v does not exist when we are done
        if (low == span.second || low->first != v) return false;

        low->second = b;

        span = neighborhood(v);
        low = std::lower_bound(span.first, span.second, u, utils::comp_v<V, B>);

        assert(low->first == u);

        low->second = b;

        return true;
    }
};
}  // namespace dimod

#endif  // DIMOD_ADJARRAYBQM_H_
