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
#include "dimod/adjmapbqm.h"
#include "dimod/adjarraybqm.h"

namespace dimod {

TEMPLATE_TEST_CASE("Tests for BQM Classes", 
                   "[bqm]", 
                   (AdjVectorBQM<int, float>), (AdjMapBQM<int, float>), (AdjArrayBQM<int, float>)) {

    SECTION("Test neighborhood()") {
        float Q[9] = {1.0, 0.0, 3.0,
                      2.0, 1.5, 6.0,
                      1.0, 0.0, 0.0};
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
        float Q[25] = {0.0, 0.0, 1.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0, 0.0};
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

        auto q = bqm.get_quadratic(0,1);
        REQUIRE(q.first == -1);
        REQUIRE(q.second);
    }

    SECTION("Test quadratic()") {
        float Q[4] = {1, 0, -1, 2};
        auto bqm = TestType(Q, 2);

        auto q = bqm.quadratic(0, 1);
        REQUIRE(q == -1);
        bqm.quadratic(1, 0) = 2;
        REQUIRE(bqm.quadratic(0, 1) == 2);
        auto&& x = bqm.quadratic(0, 1);
        x += x;
        REQUIRE(bqm.quadratic(0, 1) == 4);
    }
}

TEMPLATE_TEST_CASE("Tests for Shapeable BQM Classes", 
                   "[shapeablebqm][bqm]", 
                   (AdjVectorBQM<int, float>), (AdjMapBQM<int, float>)) {

    auto bqm = TestType();  
    bqm.add_variable();

    SECTION("Test add_variable()") {
        bqm.add_variable();
        REQUIRE(bqm.num_variables() == 2);         
    }

    SECTION("Test pop_variable()") {
        bqm.pop_variable();
        REQUIRE(bqm.num_variables() == 0); 
    }

    SECTION("Test quadratic()") {
        bqm.add_variable();
        bqm.quadratic(0, 1) = 1;
        REQUIRE(bqm.num_interactions() == 1);
    }
}

}   // namespace dimod
