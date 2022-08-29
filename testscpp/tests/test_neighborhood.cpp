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

#include "../Catch2/single_include/catch2/catch.hpp"
#include "dimod/abc.h"

namespace dimod {

SCENARIO("Neighborhood can be manipulated") {
    GIVEN("An empty Neighborhood") {
        auto neighborhood = abc::Neighborhood<float, size_t>();

        WHEN("some variables/biases are emplaced") {
            neighborhood.emplace_back(0, .5);
            neighborhood.emplace_back(1, 1.5);
            neighborhood.emplace_back(3, -3);

            THEN("we can retrieve the biases with .at()") {
                REQUIRE(neighborhood.size() == 3);
                REQUIRE(neighborhood.at(0) == .5);
                REQUIRE(neighborhood.at(1) == 1.5);
                REQUIRE(neighborhood.at(3) == -3);

                // should throw an error
                REQUIRE_THROWS_AS(neighborhood.at(2), std::out_of_range);
                REQUIRE(neighborhood.size() == 3);
            }

            THEN("we can retrieve the biases with []") {
                REQUIRE(neighborhood.size() == 3);
                REQUIRE(neighborhood[0] == .5);
                REQUIRE(neighborhood[1] == 1.5);
                REQUIRE(neighborhood[2] == 0);  // created
                REQUIRE(neighborhood[3] == -3);
                REQUIRE(neighborhood.size() == 4);  // since 2 was inserted
            }

            THEN("we can retrieve the biases with .get()") {
                REQUIRE(neighborhood.size() == 3);
                REQUIRE(neighborhood.get(0) == .5);
                REQUIRE(neighborhood.get(1) == 1.5);
                REQUIRE(neighborhood.get(1, 2) == 1.5);  // use real value
                REQUIRE(neighborhood.get(2) == 0);
                REQUIRE(neighborhood.get(2, 1.5) == 1.5);  // use default
                REQUIRE(neighborhood.at(3) == -3);
                REQUIRE(neighborhood.size() == 3);  // should not change
            }

            THEN("we can modify the biases with []") {
                neighborhood[0] += 7;
                neighborhood[2] -= 3;

                REQUIRE(neighborhood.at(0) == 7.5);
                REQUIRE(neighborhood.at(2) == -3);
            }

            THEN("we can create a vector from the neighborhood") {
                std::vector<abc::Neighborhood<float, size_t>::value_type> pairs(
                        neighborhood.begin(), neighborhood.end());

                REQUIRE(pairs[0].v == 0);
                REQUIRE(pairs[0].bias == .5);
                REQUIRE(pairs[1].v == 1);
                REQUIRE(pairs[1].bias == 1.5);
                REQUIRE(pairs[2].v == 3);
                REQUIRE(pairs[2].bias == -3);
            }

            THEN("we can create a vector from the const neighborhood") {
                std::vector<abc::Neighborhood<float, size_t>::value_type> pairs(
                        neighborhood.begin(), neighborhood.end());

                REQUIRE(pairs[0].v == 0);
                REQUIRE(pairs[0].bias == .5);
                REQUIRE(pairs[1].v == 1);
                REQUIRE(pairs[1].bias == 1.5);
                REQUIRE(pairs[2].v == 3);
                REQUIRE(pairs[2].bias == -3);
            }

            THEN("we can modify the biases via the iterator") {
                auto it = neighborhood.begin();

                (*it).bias = 18;
                REQUIRE(neighborhood.at(0) == 18);

                it++;
                (*it).bias = -48;
                REQUIRE(neighborhood.at(1) == -48);

                ++it;
                it->bias = 104;
                REQUIRE(neighborhood.at(3) == 104);
            }

            THEN("we can erase some with an iterator") {
                neighborhood.erase(neighborhood.begin() + 1, neighborhood.end());
                REQUIRE(neighborhood.size() == 1);
            }
        }
    }
}

}  // namespace dimod
