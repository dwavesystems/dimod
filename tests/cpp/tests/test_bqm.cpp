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
//
// =============================================================================

#include "../Catch2/single_include/catch2/catch.hpp"
#include "../../../dimod/include/dimod/adjvectorbqm.h"
#include "../../../dimod/include/dimod/adjmapbqm.h"
#include "../../../dimod/include/dimod/adjarraybqm.h"

#include <iostream>

using namespace dimod;
using namespace std;

TEMPLATE_TEST_CASE("Tests for BQM Classes", 
                   "[bqm]", 
                   (AdjVectorBQM<int, float>), (AdjMapBQM<int, float>), (AdjArrayBQM<int, float>)) {

    float Q[4] = {1,0,-1,2};
    auto bqm = TestType(Q, 2);

    SECTION("Test num_variable()") {
        REQUIRE(bqm.num_variables() == 2);
    }

    SECTION("Test num_interactions()") {
        REQUIRE(bqm.num_interactions() == 1);
    }

    SECTION("Test get_linear()") {
        REQUIRE(bqm.get_linear(1) == 2);   
    }

    SECTION("Test get_quadratic()") {
        auto q = bqm.get_quadratic(0,1);
        REQUIRE(q.first == -1);
        REQUIRE(q.second);
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
}
