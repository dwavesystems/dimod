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
#include "../../../dimod/roof_duality/src/fix_variables.hpp"

namespace fix_variables_ {

TEST_CASE("Test roof_duality's fixQuboVariables()", "[roofduality]") {
    SECTION("Test invalid cases") {
        auto nonsquare_matrix = compressed_matrix::CompressedMatrix<double>(2,3);
        REQUIRE_THROWS_AS(
            fixQuboVariables(nonsquare_matrix, 2),
            std::invalid_argument
        );

        auto matrix = compressed_matrix::CompressedMatrix<double>(2,2);
        REQUIRE_THROWS_AS(
            fixQuboVariables(matrix, 0),
            std::invalid_argument
        );
    }

    SECTION("Test empty case") {
        auto empty_matrix = compressed_matrix::CompressedMatrix<double>(0,0);
        FixVariablesResult result = fixQuboVariables(empty_matrix, 2);

        REQUIRE(result.offset == 0);
        REQUIRE(result.fixedVars.empty());
        REQUIRE(result.newQ.numCols() == 0);
        REQUIRE(result.newQ.numRows() == 0);
    }
}
    
}   // namespace fix_variables_
