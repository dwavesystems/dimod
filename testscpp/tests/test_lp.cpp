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

#include <cstdio>

#include "catch2/catch.hpp"
#include "dimod/lp.h"

namespace dimod {

TEST_CASE("LP tests") {
    GIVEN("A simple LP file as a string") {
        std::FILE* tmpf = std::tmpfile();
        std::fputs("minimize\nx0 - 2 x1\nsubject to\nx0 + x1 = 1\nBinary\nx0\nEnd\n", tmpf);
        std::rewind(tmpf);
        auto model = lp::read<double>(tmpf);

        if (model.variable_labels == std::vector<std::string>{"x0", "x1"}) {
            THEN("it is read correctly") {
                REQUIRE(model.variable_labels.size() == 2);
                REQUIRE(model.model.num_variables() == 2);

                CHECK(model.model.objective.linear(0) == 1);
                CHECK(model.model.objective.linear(1) == -2);

                REQUIRE(model.constraint_labels == std::vector<std::string>{""});
                REQUIRE(model.model.num_constraints() == 1);

                CHECK(model.model.constraint_ref(0).linear(0) == 1);
                CHECK(model.model.constraint_ref(0).linear(1) == 1);
                CHECK(model.model.constraint_ref(0).is_linear());

                CHECK(model.model.vartype(0) == Vartype::BINARY);
                CHECK(model.model.vartype(1) == Vartype::REAL);
            }
        } else if (model.variable_labels == std::vector<std::string>{"x1", "x0"}) {
            THEN("it is read correctly") {
                REQUIRE(model.variable_labels.size() == 2);
                REQUIRE(model.model.num_variables() == 2);

                CHECK(model.model.objective.linear(1) == 1);
                CHECK(model.model.objective.linear(0) == -2);

                REQUIRE(model.constraint_labels == std::vector<std::string>{""});
                REQUIRE(model.model.num_constraints() == 1);

                CHECK(model.model.constraint_ref(0).linear(1) == 1);
                CHECK(model.model.constraint_ref(0).linear(0) == 1);
                CHECK(model.model.constraint_ref(0).is_linear());

                CHECK(model.model.vartype(1) == Vartype::BINARY);
                CHECK(model.model.vartype(0) == Vartype::REAL);
            }
        } else {
            CHECK(false);
        }
    }
}
}  // namespace dimod
