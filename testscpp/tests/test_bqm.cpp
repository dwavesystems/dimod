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

#include <vector>

#include "../Catch2/single_include/catch2/catch.hpp"
#include "dimod/adjvectorbqm.h"

namespace dimod {

TEMPLATE_TEST_CASE("Tests for BQM Classes", "[bqm]",
                   (AdjVectorBQM<int, float>)) {
    SECTION("Test neighborhood()") {
        float Q[9] = {1.0, 0.0, 3.0, 2.0, 1.5, 6.0, 1.0, 0.0, 0.0};
        auto bqm = TestType(Q, 3);

        std::vector<typename TestType::variable_type> neighbors;
        std::vector<typename TestType::bias_type> biases;

        auto span = bqm.neighborhood(0);
        while (span.first != span.second) {
            neighbors.push_back(span.first->first);
            biases.push_back(span.first->second);
            ++span.first;
        }

        REQUIRE(neighbors.size() == 2);
        REQUIRE(neighbors[0] == 1);
        REQUIRE(neighbors[1] == 2);
        REQUIRE(biases[0] == 2.0);
        REQUIRE(biases[1] == 4.0);

        neighbors.clear();
        biases.clear();

        span = bqm.neighborhood(1);
        while (span.first != span.second) {
            neighbors.push_back(span.first->first);
            biases.push_back(span.first->second);
            ++span.first;
        }

        REQUIRE(neighbors.size() == 2);
        REQUIRE(neighbors[0] == 0);
        REQUIRE(neighbors[1] == 2);
        REQUIRE(biases[0] == 2.0);
        REQUIRE(biases[1] == 6.0);

        neighbors.clear();
        biases.clear();

        span = bqm.neighborhood(2);
        while (span.first != span.second) {
            neighbors.push_back(span.first->first);
            biases.push_back(span.first->second);
            ++span.first;
        }

        REQUIRE(neighbors.size() == 2);
        REQUIRE(neighbors[0] == 0);
        REQUIRE(neighbors[1] == 1);
        REQUIRE(biases[0] == 4.0);
        REQUIRE(biases[1] == 6.0);
    }

    SECTION("Test neighborhood() start") {
        float Q[25] = {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                       0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
        auto bqm = TestType(Q, 5);

        std::vector<typename TestType::variable_type> neighbors;
        std::vector<typename TestType::bias_type> biases;

        auto span = bqm.neighborhood(2, 0);
        while (span.first != span.second) {
            neighbors.push_back(span.first->first);
            biases.push_back(span.first->second);
            ++span.first;
        }

        REQUIRE(neighbors.size() == 3);
        REQUIRE(neighbors[0] == 0);
        REQUIRE(neighbors[1] == 3);
        REQUIRE(neighbors[2] == 4);
        REQUIRE(biases[0] == 1.0);
        REQUIRE(biases[1] == 1.0);
        REQUIRE(biases[2] == 1.0);

        neighbors.clear();
        biases.clear();

        span = bqm.neighborhood(2, 1);
        while (span.first != span.second) {
            neighbors.push_back(span.first->first);
            biases.push_back(span.first->second);
            ++span.first;
        }

        REQUIRE(neighbors.size() == 2);
        REQUIRE(neighbors[0] == 3);
        REQUIRE(neighbors[1] == 4);
        REQUIRE(biases[0] == 1.0);
        REQUIRE(biases[1] == 1.0);
    }

    SECTION("Test num_variable()") {
        float Q[4] = {1, 0, -1, 2};
        auto bqm = TestType(Q, 2);

        REQUIRE(bqm.num_variables() == 2);
    }

    SECTION("Test num_interactions()") {
        float Q[4] = {1, 0, -1, 2};
        auto bqm = TestType(Q, 2);

        REQUIRE(bqm.num_interactions() == 1);
    }

    SECTION("Test get_linear()") {
        float Q[4] = {1, 0, -1, 2};
        auto bqm = TestType(Q, 2);

        REQUIRE(bqm.get_linear(1) == 2);
    }

    SECTION("Test get_quadratic()") {
        float Q[4] = {1, 0, -1, 2};
        auto bqm = TestType(Q, 2);

        auto q = bqm.get_quadratic(0, 1);
        REQUIRE(q.first == -1);
        REQUIRE(q.second);
    }
}

TEMPLATE_TEST_CASE("Tests for Shapeable BQM Classes", "[shapeablebqm][bqm]",
                   (AdjVectorBQM<int, float>)) {
    SECTION("Test add_variable()") {
        auto bqm = TestType();
        bqm.add_variable();
        bqm.add_variable();
        REQUIRE(bqm.num_variables() == 2);
    }

    SECTION("Test COO construction empty") {
        int irow[4] = {0, 2, 0, 1};
        int icol[4] = {0, 2, 1, 2};
        float bias[4] = {.5, -2, 2, -3};

        auto bqm = TestType(&irow[0], &icol[0], &bias[0], 0);

        REQUIRE(bqm.num_variables() == 0);
        REQUIRE(bqm.num_interactions() == 0);
    }

    SECTION("Test COO construction from arrays") {
        int irow[4] = {0, 2, 0, 1};
        int icol[4] = {0, 2, 1, 2};
        float bias[4] = {.5, -2, 2, -3};

        auto bqm = TestType(&irow[0], &icol[0], &bias[0], 4);

        REQUIRE(bqm.num_variables() == 3);
        REQUIRE(bqm.linear(0) == .5);
        REQUIRE(bqm.linear(1) == 0.);
        REQUIRE(bqm.linear(2) == -2);

        REQUIRE(bqm.num_interactions() == 2);
        REQUIRE(bqm.get_quadratic(0, 1).first == 2);
        REQUIRE(bqm.get_quadratic(2, 1).first == -3);
        REQUIRE(!bqm.get_quadratic(0, 2).second);
    }

    SECTION("Test COO construction from arrays with single edge") {
        int irow[1] = {3};
        int icol[1] = {4};
        float bias[1] = {1};

        auto bqm = TestType(&irow[0], &icol[0], &bias[0], 1);

        REQUIRE(bqm.num_variables() == 5);
        REQUIRE(bqm.num_interactions() == 1);
        REQUIRE(bqm.get_quadratic(3, 4).first == 1);
    }

    SECTION("Test COO construction from arrays with duplicates") {
        int irow[6] = {0, 2, 0, 1, 0, 0};
        int icol[6] = {0, 2, 1, 2, 1, 0};
        float bias[6] = {.5, -2, 2, -3, 4, 1};

        auto bqm = TestType(&irow[0], &icol[0], &bias[0], 6);

        REQUIRE(bqm.num_variables() == 3);
        REQUIRE(bqm.linear(0) == 1.5);
        REQUIRE(bqm.linear(1) == 0.);
        REQUIRE(bqm.linear(2) == -2);

        REQUIRE(bqm.num_interactions() == 2);
        REQUIRE(bqm.get_quadratic(0, 1).first == 6);
        REQUIRE(bqm.get_quadratic(2, 1).first == -3);
        REQUIRE(!bqm.get_quadratic(0, 2).second);
    }

    SECTION("Test COO construction from arrays with multiple duplicates") {
        int irow[4] = {0, 1, 0, 1};
        int icol[4] = {1, 2, 1, 0};
        float bias[4] = {-1, 1, -2, -3};

        auto bqm = TestType(&irow[0], &icol[0], &bias[0], 4);

        REQUIRE(bqm.num_variables() == 3);
        REQUIRE(bqm.linear(0) == 0);
        REQUIRE(bqm.linear(1) == 0);
        REQUIRE(bqm.linear(2) == 0);

        REQUIRE(bqm.num_interactions() == 2);
        REQUIRE(bqm.get_quadratic(0, 1).first == -6);
        REQUIRE(bqm.get_quadratic(1, 0).first == -6);
        REQUIRE(bqm.get_quadratic(2, 1).first == 1);
        REQUIRE(bqm.get_quadratic(1, 2).first == 1);
        REQUIRE(!bqm.get_quadratic(0, 2).second);
        REQUIRE(!bqm.get_quadratic(2, 0).second);
    }

    SECTION("Test pop_variable()") {
        auto bqm = TestType();
        bqm.add_variable();
        bqm.pop_variable();
        REQUIRE(bqm.num_variables() == 0);
    }
}

}  // namespace dimod
