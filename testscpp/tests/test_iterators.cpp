// Copyright 2022 D-Wave Systems Inc.
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

#include "catch2/catch.hpp"
#include "dimod/iterators.h"

namespace dimod {
namespace iterators {

TEST_CASE("Test ConstraintIterator") {
    GIVEN("A vector of shared_ptr to ints") {
        std::vector<std::shared_ptr<int>> vec;

        for (int i = 0; i < 10; ++i) {
            vec.push_back(std::make_shared<int>(i));
        }

        WHEN("we build a ConstraintIterator from the vector's interators") {
            auto it = make_constraint_iterator(vec.begin());

            THEN("we can do comparisons") {
                CHECK(it == make_constraint_iterator(vec.begin()));
                CHECK(it != make_constraint_iterator(vec.end()));

                CHECK(it < make_constraint_iterator(vec.end()));
                CHECK(!(it < make_constraint_iterator(vec.begin())));

                CHECK(make_constraint_iterator(vec.end()) > it);
                CHECK(!(make_constraint_iterator(vec.begin()) > it));

                CHECK(it <= make_constraint_iterator(vec.end()));
                CHECK(it <= make_constraint_iterator(vec.begin()));

                CHECK(!(it >= make_constraint_iterator(vec.end())));
                CHECK(it >= make_constraint_iterator(vec.begin()));
            }

            THEN("we can increment and decrement") {
                CHECK(++it == make_constraint_iterator(vec.begin() + 1));

                CHECK(it++ == make_constraint_iterator(vec.begin() + 1));
                CHECK(it == make_constraint_iterator(vec.begin() + 2));

                CHECK(--it == make_constraint_iterator(vec.begin() + 1));

                CHECK(it-- == make_constraint_iterator(vec.begin() + 1));
                CHECK(it == make_constraint_iterator(vec.begin()));
            }
        }

        THEN("we can add and subtract") {
            auto it1 = make_constraint_iterator(vec.begin() + 3);
            auto it2 = make_constraint_iterator(vec.begin());
            CHECK(it1 - it2 == 3);
        }

        AND_GIVEN("the beginning and end") {
            auto it = make_constraint_iterator(vec.begin());
            auto end = make_constraint_iterator(vec.end());

            THEN("we can iterate over, accessing by deref and []") {
                for (int i = 0; it != end; ++it, ++i) {
                    CHECK(*it == i);
                    CHECK(it[0] == i);
                }
            }
        }
    }
}

}  // namespace iterators
}  // namespace dimod
