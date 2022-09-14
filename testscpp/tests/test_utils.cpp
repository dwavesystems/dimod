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
#include <random>

#include "catch2/catch.hpp"
#include "dimod/utils.h"

namespace dimod {
namespace utils {

    TEST_CASE("Two vectors are zip-sorted", "[utils]") {
        std::default_random_engine generator;
        std::uniform_int_distribution<int> int_distribution(0, 100);
        std::uniform_real_distribution<double> signed_distribution(-100, 100);

        // 1000 random sortings
        for (int i = 0; i < 1000; i++) {
            int num_elements = int_distribution(generator);

            std::vector<int> control(num_elements);
            std::vector<int> response(num_elements);
            std::vector<std::pair<int, int>> v_pair(num_elements);

            for (int i = 0; i < num_elements; i++) {
                control[i] = i;
                response[i] = signed_distribution(generator);
            }

            std::shuffle(control.begin(), control.end(), generator);

            for (int i = 0; i < num_elements; i++) {
                v_pair[i] = {control[i], response[i]};
            }

            zip_sort(control, response);
            std::sort(v_pair.begin(), v_pair.end());

            REQUIRE(std::is_sorted(control.begin(), control.end()));

            for (int i = 0; i < num_elements; i++) {
                REQUIRE(v_pair[i].first == control[i]);
                REQUIRE(v_pair[i].second == response[i]);
            }
        }
    }

    TEST_CASE("Two vectors with duplicates are zip-sorted", "[utils]") {
        std::default_random_engine generator;
        std::uniform_int_distribution<int> int_distribution(0, 50);
        std::uniform_real_distribution<double> signed_distribution(-100, 100);

        // 10 random sortings
        for (int i = 0; i < 10; i++) {
            int num_elements = int_distribution(generator);

            std::vector<int> control(2 * num_elements);
            std::vector<int> response(2 * num_elements);
            std::vector<std::pair<int, int>> v_pair(2 * num_elements);

            for (int i = 0; i < num_elements; i++) {
                control[2 * i] = i;
                control[2 * i + 1] = i;
            }

            std::shuffle(control.begin(), control.end(), generator);

            for (std::size_t i = 0; i < control.size(); i++) {
                response[i] = control[i];
                v_pair[i] = {control[i], response[i]};
            }

            zip_sort(control, response);
            std::sort(v_pair.begin(), v_pair.end());

            REQUIRE(std::is_sorted(control.begin(), control.end()));

            for (int i = 0; i < num_elements; i++) {
                REQUIRE(v_pair[i].first == control[i]);
                REQUIRE(v_pair[i].second == response[i]);
            }
        }
    }

}  // namespace utils
}  // namespace dimod
