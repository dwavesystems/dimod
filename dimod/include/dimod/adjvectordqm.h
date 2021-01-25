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

#ifndef DIMOD_ADJVECTORDQM_H_
#define DIMOD_ADJVECTORDQM_H_

#include <algorithm>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dimod/adjvectorbqm.h"
#include "dimod/utils.h"

namespace dimod {

template <class V, class B>
class AdjVectorDQM {
 public:
    using bias_type = B;
    using variable_type = V;
    using size_type = std::size_t;

    AdjVectorBQM<variable_type, bias_type> bqm_;
    std::vector<variable_type> case_starts_;  // len(adj_) + 1
    std::vector<std::vector<variable_type>> adj_;

    AdjVectorDQM() { case_starts_.push_back(0); }

    template <class DQM>
    explicit AdjVectorDQM(const DQM &dqm) {
        bqm_ = dqm.bqm_;
        case_starts_.insert(case_starts_.begin(), dqm.case_starts_.begin(),
                            dqm.case_starts_.end());
        adj_.resize(dqm.adj_.size());
        variable_type num_variables = dqm.num_variables();
        for (variable_type v = 0; v < num_variables; v++) {
            adj_[v].insert(adj_[v].begin(), dqm.adj_[v].begin(),
                           dqm.adj_[v].end());
        }
    }

    template <class io_variable_type, class io_bias_type>
    AdjVectorDQM(io_variable_type *case_starts, size_type num_variables,
                 io_bias_type *linear_biases, size_type num_cases,
                 io_variable_type *irow, io_variable_type *icol,
                 io_bias_type *quadratic_biases, size_type num_interactions) {
        // Set the BQM, linear biases will be added separately.
        if (num_interactions) {
            bqm_ = AdjVectorBQM<variable_type, bias_type>(
                    irow, icol, quadratic_biases, num_interactions, true);
        }

        // Accounting for the cases/variables at the end without interaction.
        while (bqm_.num_variables() < num_cases) {
            bqm_.add_variable();
        }
        assert(bqm_.num_variables() == num_cases);

        // Add the linear biases.
        for (variable_type ci = 0; ci < num_cases; ci++) {
            bqm_.set_linear(ci, linear_biases[ci]);
        }

        // Set the case starts.
        case_starts_.resize(num_variables + 1);
        for (variable_type v = 0; v < num_variables; v++) {
            case_starts_[v] = case_starts[v];
        }
        case_starts_[num_variables] = num_cases;

        // Fill the adjacency list for variables.
        std::vector<std::unordered_set<variable_type>> adjset;
        adjset.resize(num_variables);
        variable_type u = 0;
        variable_type num_total_cases = bqm_.num_variables();
        for (variable_type ci = 0; ci < num_total_cases; ci++) {
            while (ci >= case_starts_[u + 1]) {
                u++;
            }
            auto span = bqm_.neighborhood(ci);
            variable_type v = 0;
            while (span.first != span.second) {
                variable_type cj = (span.first)->first;
                while (cj >= case_starts_[v + 1]) {
                    v++;
                }
                adjset[u].insert(v);
                span.first++;
            }
        }

        adj_.resize(num_variables);
#pragma omp parallel for
        for (variable_type v = 0; v < num_variables; v++) {
            adj_[v].insert(adj_[v].begin(), adjset[v].begin(), adjset[v].end());
            std::sort(adj_[v].begin(), adj_[v].end());
        }
    }

    template <class io_variable_type, class io_bias_type>
    void to_coo(io_variable_type *case_starts, io_bias_type *linear_biases,
                io_variable_type *irow, io_variable_type *icol,
                io_bias_type *quadratic_biases) {
        variable_type num_variables = this->num_variables();
        variable_type num_total_cases = bqm_.num_variables();

        for (variable_type v = 0; v < num_variables; v++) {
            case_starts[v] = case_starts_[v];
        }

        size_type qi = 0;
        for (variable_type ci = 0; ci < num_total_cases; ci++) {
            linear_biases[ci] = bqm_.get_linear(ci);
            auto span = bqm_.neighborhood(ci);
            while ((span.first != span.second) && ((span.first)->first < ci)) {
                irow[qi] = ci;
                icol[qi] = (span.first)->first;
                quadratic_biases[qi] = (span.first)->second;
                span.first++;
                qi++;
            }
        }
    }

    bool self_loop_present() {
        variable_type num_variables = this->num_variables();
        for (variable_type v = 0; v < num_variables; v++) {
            for (variable_type ci = case_starts_[v],
                               ci_end = case_starts_[v + 1];
                 ci < ci_end; ci++) {
                auto span = bqm_.neighborhood(ci, case_starts_[v]);
                if ((span.first != span.second) &&
                    ((span.first)->first < case_starts_[v + 1])) {
                    return true;
                }
            }
        }
        return false;
    }

    bool connection_present(variable_type u, variable_type v) {
        bool connected = true;
        auto it = std::lower_bound(adj_[u].begin(), adj_[u].end(), v);
        if (it == adj_[u].end() || *it != v) {
            connected = false;
        }
        return connected;
    }

    void connect_variables(variable_type u, variable_type v) {
        auto low = std::lower_bound(adj_[u].begin(), adj_[u].end(), v);
        if (low == adj_[u].end() || *low != v) {
            adj_[u].insert(low, v);
            adj_[v].insert(std::lower_bound(adj_[v].begin(), adj_[v].end(), u),
                           u);
        }
    }

    size_type num_variables() { return adj_.size(); }

    size_type num_variable_interactions() {
        size_type num = 0;
        variable_type num_variables = this->num_variables();
        for (variable_type v = 0; v < num_variables; v++) {
            num += adj_[v].size();
        }
        return (num / 2);
    }

    size_type num_cases(variable_type v) {
        assert(v >= 0 && v < this->num_variables());
        return (case_starts_[v + 1] - case_starts_[v]);
    }

    size_type num_case_interactions() { return bqm_.num_interactions(); }

    variable_type add_variable(variable_type num_cases) {
        assert(num_cases > 0);
        variable_type v = adj_.size();
        adj_.resize(v + 1);
        for (variable_type n = 0; n < num_cases; n++) {
            bqm_.add_variable();
        }
        case_starts_.push_back(bqm_.num_variables());
        return v;
    }

    bias_type &linear_case(variable_type v, variable_type case_v) {
        assert(v >= 0 && v < this->num_variables());
        assert(case_v >= 0 && case_v < this->num_cases(v));
        return bqm_.linear(case_starts_[v] + case_v);
    }

    template <class io_bias_type>
    void get_linear(variable_type v, io_bias_type *biases) {
        assert(v >= 0 && v < this->num_variables());
        variable_type num_cases_v = this->num_cases(v);
        for (variable_type case_v = 0; case_v < num_cases_v; case_v++) {
            biases[case_v] = bqm_.linear(case_starts_[v] + case_v);
        }
    }

    template <class io_bias_type>
    void set_linear(variable_type v, io_bias_type *biases) {
        assert(v >= 0 && v < this->num_variables());
        variable_type num_cases_v = this->num_cases(v);
        for (variable_type case_v = 0; case_v < num_cases_v; case_v++) {
            bqm_.linear(case_starts_[v] + case_v) = biases[case_v];
        }
    }

    std::pair<bias_type, bool> get_quadratic_case(variable_type u,
                                                  variable_type case_u,
                                                  variable_type v,
                                                  variable_type case_v) {
        assert(u >= 0 && u < this->num_variables());
        assert(case_u >= 0 && case_u < this->num_cases(u));
        assert(v >= 0 && v < this->num_variables());
        assert(case_v >= 0 && case_v < this->num_cases(v));
        variable_type cu = case_starts_[u] + case_u;
        variable_type cv = case_starts_[v] + case_v;
        return bqm_.get_quadratic(cu, cv);
    }

    // Check if boolean type is still okay
    bool set_quadratic_case(variable_type u, variable_type case_u,
                            variable_type v, variable_type case_v,
                            bias_type bias) {
        assert(u >= 0 && u < this->num_variables());
        assert(case_u >= 0 && case_u < this->num_cases(u));
        assert(v >= 0 && v < this->num_variables());
        assert(case_v >= 0 && case_v < this->num_cases(v));
        if (u == v) {
            return false;
        }
        variable_type cu = case_starts_[u] + case_u;
        variable_type cv = case_starts_[v] + case_v;
        bqm_.set_quadratic(cu, cv, bias);
        connect_variables(u, v);
        return true;
    }

    // Returns false if there is no interaction among the variables.
    template <class io_bias_type>
    bool get_quadratic(variable_type u, variable_type v,
                       io_bias_type *quadratic_biases) {
        assert(u >= 0 && u < this->num_variables());
        assert(v >= 0 && v < this->num_variables());
        if (!connection_present(u, v)) {
            return false;
        }
        variable_type num_cases_u = num_cases(u);
        variable_type num_cases_v = num_cases(v);
        memset(quadratic_biases, 0,
               (size_t)num_cases_u * (size_t)num_cases_v *
                       sizeof(io_bias_type));
#pragma omp parallel for
        for (variable_type case_u = 0; case_u < num_cases_u; case_u++) {
            auto span = bqm_.neighborhood(case_starts_[u] + case_u,
                                          case_starts_[v]);
            while (span.first != span.second &&
                   (span.first)->first < case_starts_[v + 1]) {
                variable_type case_v = (span.first)->first - case_starts_[v];
                quadratic_biases[case_u * num_cases_v + case_v] =
                        (span.first)->second;
                span.first++;
            }
        }
        return true;
    }

    template <class io_bias_type>
    bool set_quadratic(variable_type u, variable_type v, io_bias_type *biases) {
        assert(u >= 0 && u < this->num_variables());
        assert(v >= 0 && v < this->num_variables());
        if (u == v) {
            return false;
        }
        variable_type num_cases_u = num_cases(u);
        variable_type num_cases_v = num_cases(v);
        // This cannot be parallelized since the vectors cannot be reshaped in
        // parallel.
        bool inserted = false;
        for (variable_type case_u = 0; case_u < num_cases_u; case_u++) {
            variable_type cu = case_starts_[u] + case_u;
            for (variable_type case_v = 0; case_v < num_cases_v; case_v++) {
                variable_type cv = case_starts_[v] + case_v;
                bias_type bias = biases[case_u * num_cases_v + case_v];
                if (bias != (bias_type)0) {
                    bqm_.set_quadratic(cu, cv, bias);
                    inserted = true;
                }
            }
        }

        if (inserted) {
            connect_variables(u, v);
            return true;
        } else {
            return false;
        }
    }

    template <class io_variable_type>
    double get_energy(io_variable_type *sample) {
        variable_type num_variables = this->num_variables();
        double sample_energy = 0;
        for (variable_type u = 0; u < num_variables; u++) {
            variable_type case_u = sample[u];
            assert(case_u < num_cases(u));
            variable_type cu = case_starts_[u] + case_u;
            sample_energy += bqm_.get_linear(cu);
            for (variable_type vi = 0; vi < adj_[u].size(); vi++) {
                variable_type v = adj_[u][vi];
                // We only care about lower triangle.
                if (v > u) {
                    break;
                }
                variable_type case_v = sample[v];
                variable_type cv = case_starts_[v] + case_v;
                auto out = bqm_.get_quadratic(cu, cv);
                if (out.second) {
                    sample_energy += out.first;
                }
            }
        }
        return sample_energy;
    }
};
}  // namespace dimod

#endif  // DIMOD_ADJVECTORDQM_H_
