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

#include <iostream>
#include <vector>

#include "catch2/catch.hpp"
#include "dimod/utils.h"

namespace dimod {
namespace utils {

TEST_CASE("remove_by_index()") {
    GIVEN("A vector") {
        auto v = std::vector<int>{0, 1, 2, 3, 4, 5, 6};

        AND_GIVEN("some indices") {
            auto i = std::vector<int>{1, 3, 4};

            WHEN("We use remove_by_index() to shrink the vector") {
                v.erase(remove_by_index(v.begin(), v.end(), i.begin(), i.end()), v.end());

                THEN("The vector has the values we expect") {
                    REQUIRE_THAT(v, Catch::Approx(std::vector<int>{0, 2, 5, 6}));
                }
            }
        }

        AND_GIVEN("Some indices that are out-of-range") {
            auto i = std::vector<int>{5, 105};

            WHEN("We use remove_by_index() to shrink the vector") {
                v.erase(remove_by_index(v.begin(), v.end(), i.begin(), i.end()), v.end());

                THEN("The vector has the values we expect") {
                    REQUIRE_THAT(v, Catch::Approx(std::vector<int>{0, 1, 2, 3, 4, 6}));
                }
            }
        }

        AND_GIVEN("An empty indices vector") {
            auto i = std::vector<int>{};

            WHEN("We use remove_by_index() to shrink the vector") {
                v.erase(remove_by_index(v.begin(), v.end(), i.begin(), i.end()), v.end());

                THEN("The vector has the values we expect") {
                    REQUIRE_THAT(v, Catch::Approx(std::vector<int>{0, 1, 2, 3, 4, 5, 6}));
                }
            }
        }
    }
}

}  // namespace utils
}  // namespace dimod
