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

#include "catch2/catch.hpp"
#include "dimod/quadratic_model.h"

namespace dimod {

// todo: reorganize other tests
TEST_CASE("QuadraticModel tests") {
    GIVEN("a linear QM with mixed vartypes") {
        auto qm = QuadraticModel<double>();
        qm.add_variable(Vartype::BINARY);
        qm.add_variable(Vartype::INTEGER, -1, 1);
        qm.add_variable(Vartype::REAL, -2, 2);
        qm.add_variable(Vartype::REAL, -3, 3);
        qm.add_variable(Vartype::SPIN);
        qm.set_offset(5);
        qm.set_linear(0, {0, -1, +2, -3, +4});

        WHEN("we use remove_variable()") {
            qm.remove_variable(2);

            THEN("the variable is removed and the model is reindexed") {
                REQUIRE(qm.num_variables() == 4);
                REQUIRE(qm.num_interactions() == 0);
                CHECK(qm.offset() == 5);
                CHECK(qm.linear(0) == 0);
                CHECK(qm.linear(1) == -1);
                CHECK(qm.linear(2) == -3);  // this was reindexed
                CHECK(qm.linear(3) == 4);   // this was reindexed
                CHECK(qm.vartype(0) == Vartype::BINARY);
                CHECK(qm.vartype(1) == Vartype::INTEGER);
                CHECK(qm.vartype(2) == Vartype::REAL);  // this was reindexed
                CHECK(qm.vartype(3) == Vartype::SPIN);  // this was reindexed
                CHECK(qm.lower_bound(1) == -1);
                CHECK(qm.lower_bound(2) == -3);  // this was reindexed
                CHECK(qm.upper_bound(1) == 1);
                CHECK(qm.upper_bound(2) == 3);  // this was reindexed
            }
        }

        WHEN("we use remove_variables()") {
            qm.remove_variables({2});

            THEN("the variable is removed and the model is reindexed") {
                REQUIRE(qm.num_variables() == 4);
                REQUIRE(qm.num_interactions() == 0);
                CHECK(qm.offset() == 5);
                CHECK(qm.linear(0) == 0);
                CHECK(qm.linear(1) == -1);
                CHECK(qm.linear(2) == -3);  // this was reindexed
                CHECK(qm.linear(3) == 4);   // this was reindexed
                CHECK(qm.vartype(0) == Vartype::BINARY);
                CHECK(qm.vartype(1) == Vartype::INTEGER);
                CHECK(qm.vartype(2) == Vartype::REAL);  // this was reindexed
                CHECK(qm.vartype(3) == Vartype::SPIN);  // this was reindexed
                CHECK(qm.lower_bound(1) == -1);
                CHECK(qm.lower_bound(2) == -3);  // this was reindexed
                CHECK(qm.upper_bound(1) == 1);
                CHECK(qm.upper_bound(2) == 3);  // this was reindexed
            }
        }

        WHEN("we use fix_variable()") {
            qm.fix_variable(2, -1);
            THEN("the variable is removed, its biases distributed and the model is reindexed") {
                REQUIRE(qm.num_variables() == 4);
                REQUIRE(qm.num_interactions() == 0);
                CHECK(qm.offset() == 3);  // linear biases -1*2 was added to offset
                CHECK(qm.linear(0) == 0);
                CHECK(qm.linear(1) == -1);
                CHECK(qm.linear(2) == -3);  // this was reindexed
                CHECK(qm.linear(3) == 4);   // this was reindexed
                CHECK(qm.vartype(0) == Vartype::BINARY);
                CHECK(qm.vartype(1) == Vartype::INTEGER);
                CHECK(qm.vartype(2) == Vartype::REAL);  // this was reindexed
                CHECK(qm.vartype(3) == Vartype::SPIN);  // this was reindexed
                CHECK(qm.lower_bound(1) == -1);
                CHECK(qm.lower_bound(2) == -3);  // this was reindexed
                CHECK(qm.upper_bound(1) == 1);
                CHECK(qm.upper_bound(2) == 3);  // this was reindexed
            }
        }
    }
}

SCENARIO("A small quadratic model can be manipulated", "[qm]") {
    GIVEN("An empty quadratic model with float biases") {
        auto qm = QuadraticModel<float>();

        WHEN("an integer, spin and binary variable are added without explicit "
             "bounds") {
            auto v_int = qm.add_variable(Vartype::INTEGER);
            auto v_bin = qm.add_variable(Vartype::BINARY);
            auto v_spn = qm.add_variable(Vartype::SPIN);

            THEN("the defaults bounds chosen appropriately") {
                REQUIRE(qm.num_variables() == 3);

                CHECK(qm.vartype(v_int) == Vartype::INTEGER);
                CHECK(qm.lower_bound(v_int) == 0);
                CHECK(qm.upper_bound(v_int) == 16777215);

                CHECK(qm.vartype(v_bin) == Vartype::BINARY);
                CHECK(qm.lower_bound(v_bin) == 0);
                CHECK(qm.upper_bound(v_bin) == 1);

                CHECK(qm.vartype(v_spn) == Vartype::SPIN);
                CHECK(qm.lower_bound(v_spn) == -1);
                CHECK(qm.upper_bound(v_spn) == 1);
            }
        }
    }

    GIVEN("An empty quadratic model with double biases") {
        auto qm = QuadraticModel<double>();

        WHEN("an integer, spin and binary variable are added without explicit "
             "bounds") {
            auto v_int = qm.add_variable(Vartype::INTEGER);
            auto v_bin = qm.add_variable(Vartype::BINARY);
            auto v_spn = qm.add_variable(Vartype::SPIN);

            THEN("the defaults bounds chosen appropriately") {
                REQUIRE(qm.num_variables() == 3);

                CHECK(qm.vartype(v_int) == Vartype::INTEGER);
                CHECK(qm.lower_bound(v_int) == 0);
                CHECK(qm.upper_bound(v_int) == 9007199254740991);

                CHECK(qm.vartype(v_bin) == Vartype::BINARY);
                CHECK(qm.lower_bound(v_bin) == 0);
                CHECK(qm.upper_bound(v_bin) == 1);

                CHECK(qm.vartype(v_spn) == Vartype::SPIN);
                CHECK(qm.lower_bound(v_spn) == -1);
                CHECK(qm.upper_bound(v_spn) == 1);
            }

            THEN("the linear biases default to 0 and there are no quadratic") {
                REQUIRE(qm.num_variables() == 3);

                CHECK(qm.linear(v_int) == 0);
                CHECK(qm.linear(v_bin) == 0);
                CHECK(qm.linear(v_spn) == 0);

                CHECK(qm.num_interactions() == 0);
                CHECK_THROWS_AS(qm.quadratic_at(v_int, v_bin), std::out_of_range);
                CHECK_THROWS_AS(qm.quadratic_at(v_int, v_spn), std::out_of_range);
                CHECK_THROWS_AS(qm.quadratic_at(v_bin, v_spn), std::out_of_range);
            }

            AND_WHEN("we set some quadratic biases") {
                qm.set_quadratic(v_int, v_bin, 1.5);
                qm.set_quadratic(v_bin, v_spn, -3);

                THEN("we can read them back out") {
                    REQUIRE(qm.num_variables() == 3);
                    CHECK(qm.quadratic(v_int, v_bin) == 1.5);
                    CHECK(qm.quadratic(v_bin, v_spn) == -3);
                    CHECK_THROWS_AS(qm.quadratic_at(v_int, v_spn), std::out_of_range);
                }
            }

            AND_WHEN("we set some quadratic biases on self-loops") {
                qm.set_quadratic(v_int, v_int, 1.5);
                CHECK_THROWS_AS(qm.set_quadratic(v_bin, v_bin, -3), std::domain_error);
                CHECK_THROWS_AS(qm.set_quadratic(v_spn, v_spn, -3), std::domain_error);

                THEN("we can read them back out") {
                    REQUIRE(qm.num_variables() == 3);
                    CHECK(qm.quadratic(v_int, v_int) == 1.5);
                }
            }
        }

        WHEN("we add an integer variable with a self-loops") {
            auto v = qm.add_variable(Vartype::INTEGER);

            qm.set_quadratic(v, v, 1.5);

            THEN("it is accounted for correctly in num_interactions") {
                CHECK(qm.num_interactions() == 1);
            }
            THEN("we can retrieve the quadratic bias") { CHECK(qm.quadratic(v, v) == 1.5); }

            AND_WHEN("we add another variable with another self-loop") {
                auto u = qm.add_variable(Vartype::INTEGER);

                qm.add_quadratic(u, u, -2);

                THEN("it is accounted for correctly in num_interactions") {
                    CHECK(qm.num_interactions() == 2);
                }
                THEN("we can retrieve the quadratic bias") {
                    CHECK(qm.quadratic(v, v) == 1.5);
                    CHECK(qm.quadratic(u, u) == -2);
                }
            }
        }

        WHEN("we add a real variable with a self-loop") {
            auto v = qm.add_variable(Vartype::REAL);
            qm.set_quadratic(v, v, 1.5);

            THEN("it is accounted for correctly in num_interactions") {
                CHECK(qm.num_interactions() == 1);
            }
            THEN("we can retrieve the quadratic bias") { CHECK(qm.quadratic(v, v) == 1.5); }

            AND_WHEN("we add another variable with another self-loop") {
                auto u = qm.add_variable(Vartype::REAL);

                qm.add_quadratic(u, u, -2);

                THEN("it is accounted for correctly in num_interactions") {
                    CHECK(qm.num_interactions() == 2);
                }
                THEN("we can retrieve the quadratic bias") {
                    CHECK(qm.quadratic(v, v) == 1.5);
                    CHECK(qm.quadratic(u, u) == -2);
                }
            }
        }

        WHEN("we add two integer variables with an interaction") {
            auto u = qm.add_variable(Vartype::INTEGER);
            auto v = qm.add_variable(Vartype::INTEGER);

            qm.set_quadratic(u, v, 1);

            AND_WHEN("we calculate the energy of a sample with large biases") {
                // https://github.com/dwavesystems/dimod/issues/982
                std::vector<std::int64_t> samples = {4294967296, 4294967296};

                THEN("we get the value we expect") {
                    CHECK(qm.energy(samples.begin()) == Approx(4294967296.0 * 4294967296.0));
                }
            }
        }

        WHEN("the quadratic model is resized") {
            qm.resize(10, Vartype::BINARY);

            THEN("missing variables are given the provided vartype") {
                CHECK(qm.num_variables() == 10);

                for (size_t v = 0; v < qm.num_variables(); ++v) {
                    CHECK(qm.vartype(v) == Vartype::BINARY);
                }
            }

            AND_WHEN("we shrink it again") {
                qm.resize(5);

                THEN("it is shrunk accordingly") {
                    CHECK(qm.num_variables() == 5);

                    for (size_t v = 0; v < qm.num_variables(); ++v) {
                        CHECK(qm.vartype(v) == Vartype::BINARY);
                    }
                }
            }

            AND_WHEN("we add more variables of other vartypes") {
                qm.resize(15, Vartype::SPIN);
                qm.resize(20, Vartype::INTEGER, 0, 5);

                THEN("missing variables are given the provided vartype and/or "
                     "bounds") {
                    CHECK(qm.num_variables() == 20);

                    for (size_t v = 0; v < 10; ++v) {
                        CHECK(qm.vartype(v) == Vartype::BINARY);
                    }
                    for (size_t v = 10; v < 15; ++v) {
                        CHECK(qm.vartype(v) == Vartype::SPIN);
                    }
                    for (size_t v = 15; v < 20; ++v) {
                        CHECK(qm.vartype(v) == Vartype::INTEGER);
                        CHECK(qm.lower_bound(v) == 0);
                        CHECK(qm.upper_bound(v) == 5);
                    }
                }
            }
        }
    }
}

SCENARIO("A quadratic model can be constructed from a binary quadratic model", "[bqm]") {
    GIVEN("A binary quadratic model") {
        auto bqm = BinaryQuadraticModel<float>(3, Vartype::SPIN);
        bqm.set_linear(0, {4, 0, -2});
        bqm.set_quadratic(0, 1, 1.5);
        bqm.set_quadratic(1, 2, -3);
        bqm.set_offset(5);

        WHEN("a quadratic model is constructed from it") {
            auto qm = QuadraticModel<float>(bqm);

            THEN("the biases etc are passed in") {
                REQUIRE(qm.num_variables() == 3);
                REQUIRE(qm.num_interactions() == 2);

                CHECK(qm.linear(0) == 4);
                CHECK(qm.linear(1) == 0);
                CHECK(qm.linear(2) == -2);

                CHECK(qm.vartype(0) == Vartype::SPIN);
                CHECK(qm.vartype(1) == Vartype::SPIN);
                CHECK(qm.vartype(2) == Vartype::SPIN);

                CHECK(qm.quadratic(0, 1) == 1.5);
                CHECK(qm.quadratic(1, 2) == -3);
                CHECK_THROWS_AS(qm.quadratic_at(0, 2), std::out_of_range);
            }
        }

        WHEN("a quadratic model with a different type is constructed from it") {
            auto qm = QuadraticModel<double>(bqm);

            THEN("the biases etc are passed in") {
                REQUIRE(qm.num_variables() == 3);
                REQUIRE(qm.num_interactions() == 2);

                CHECK(qm.linear(0) == 4);
                CHECK(qm.linear(1) == 0);
                CHECK(qm.linear(2) == -2);

                CHECK(qm.vartype(0) == Vartype::SPIN);
                CHECK(qm.vartype(1) == Vartype::SPIN);
                CHECK(qm.vartype(2) == Vartype::SPIN);

                CHECK(qm.quadratic(0, 1) == 1.5);
                CHECK(qm.quadratic(1, 2) == -3);
                CHECK_THROWS_AS(qm.quadratic_at(0, 2), std::out_of_range);
            }
        }
    }
}

SCENARIO("The variables of a quadratic model can have their vartypes changed", "[qm]") {
    GIVEN("A quadratic model with a spin and a binary variable") {
        auto qm = QuadraticModel<double>();
        auto s = qm.add_variable(Vartype::SPIN);
        auto x = qm.add_variable(Vartype::BINARY);

        qm.set_linear(s, 2);
        qm.set_linear(x, 4);
        qm.set_quadratic(s, x, 3);

        WHEN("The spin variable is changed to binary") {
            qm.change_vartype(Vartype::BINARY, s);

            THEN("the biases update appropriately") {
                CHECK(qm.linear(s) == 4);
                CHECK(qm.linear(x) == 1);
                CHECK(qm.quadratic(s, x) == 6);
                CHECK(qm.quadratic(x, s) == 6);
                CHECK(qm.offset() == -2);

                CHECK(qm.upper_bound(s) == 1);
                CHECK(qm.lower_bound(s) == 0);
                CHECK(qm.vartype(s) == Vartype::BINARY);
            }
        }

        WHEN("The binary variable is changed to spin") {
            CHECK(qm.linear(s) == 2);
            CHECK(qm.linear(x) == 4);

            qm.change_vartype(Vartype::SPIN, x);
            auto t = x;

            THEN("the biases update appropriately") {
                CHECK(qm.linear(s) == 7. / 2);
                CHECK(qm.linear(t) == 2);
                CHECK(qm.quadratic(s, t) == 3. / 2);
                CHECK(qm.quadratic(t, s) == 3. / 2);
                CHECK(qm.offset() == 2);

                CHECK(qm.upper_bound(t) == +1);
                CHECK(qm.lower_bound(t) == -1);
                CHECK(qm.vartype(t) == Vartype::SPIN);
            }
        }

        WHEN("The spin variable is changed to integer") {
            qm.change_vartype(Vartype::INTEGER, s);

            THEN("the biases update appropriately") {
                CHECK(qm.linear(s) == 4);
                CHECK(qm.linear(x) == 1);
                CHECK(qm.quadratic(s, x) == 6);
                CHECK(qm.quadratic(x, s) == 6);
                CHECK(qm.offset() == -2);

                CHECK(qm.upper_bound(s) == 1);
                CHECK(qm.lower_bound(s) == 0);
                CHECK(qm.vartype(s) == Vartype::INTEGER);
            }
        }

        WHEN("The vartype  and bounds are changed manually ") {
            qm.set_vartype(s, Vartype::BINARY);
            qm.set_lower_bound(s, -2);
            qm.set_upper_bound(s, 2);

            THEN("the biases do not update") {
                CHECK(qm.linear(s) == 2);
                CHECK(qm.linear(x) == 4);
                CHECK(qm.quadratic(s, x) == 3);
                CHECK(qm.offset() == 0);
                CHECK(qm.vartype(s) == Vartype::BINARY);
                CHECK(qm.lower_bound(s) == -2);
                CHECK(qm.upper_bound(s) == 2);
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: the size of quadratic models in bytes can be determined", "[qm]",
                   double, float) {
    GIVEN("a binary quadratic model") {
        auto bqm = dimod::BinaryQuadraticModel<TestType>(5, dimod::Vartype::BINARY);
        bqm.add_quadratic(0, 1, 1.5);
        bqm.add_quadratic(1, 2, 1.5);
        bqm.add_quadratic(2, 3, 1.5);

        struct term {
            int v;
            TestType bias;
        };

        auto pair_size = sizeof(term);

        // this assumption is not guaranteed across compilers, but it
        // is required for this test to make sense
        CHECK(pair_size == 2 * sizeof(TestType));

        THEN("we can determine the number of bytes used by the elements") {
            CHECK(bqm.nbytes() == bqm.num_variables() * sizeof(TestType)            // linear
                                          + 2 * bqm.num_interactions() * pair_size  // quadratic
                                          + sizeof(TestType));                      // offset
            CHECK(bqm.nbytes(true) >= bqm.nbytes());
        }

        AND_GIVEN("a quadratic model") {
            auto qm = dimod::QuadraticModel<TestType>(bqm);

            struct varinfo_type {
                Vartype vartype;
                TestType lb;
                TestType ub;
            };

            THEN("we can determine the number of bytes used by the elements") {
                CHECK(qm.nbytes() ==
                      qm.num_variables() * sizeof(TestType)                  // linear
                              + 2 * qm.num_interactions() * pair_size        // quadratic
                              + sizeof(TestType)                             // offset
                              + qm.num_variables() * sizeof(varinfo_type));  // vartypes
                CHECK(qm.nbytes(true) >= qm.nbytes());
            }
        }
    }
}

SCENARIO("quadratic models with square terms") {
    GIVEN("a quadratic model with square terms") {
        auto qm = QuadraticModel<double>();
        auto i = qm.add_variable(Vartype::INTEGER);
        auto j = qm.add_variable(Vartype::INTEGER);

        qm.add_quadratic(i, i, 1);
        qm.add_quadratic(i, j, 2);
        qm.add_quadratic(j, j, 3);

        WHEN("the energy is calculated") {
            std::vector<int> sample = {5, -1};
            auto en = qm.energy(sample.begin());
            THEN("the energy incorporates the square terms") { CHECK(en == 18); }
        }
    }
}

SCENARIO("quadratic models can be swapped", "[qm]") {
    GIVEN("two quadratic models") {
        auto qm0 = dimod::QuadraticModel<double>();
        auto s = qm0.add_variable(Vartype::SPIN);
        auto x = qm0.add_variable(Vartype::BINARY);
        auto i = qm0.add_variable(Vartype::INTEGER, -5, 5);

        qm0.set_linear(s, 1);
        qm0.set_linear(x, 2);
        qm0.set_linear(i, 3);
        qm0.add_quadratic(s, x, 4);
        qm0.add_quadratic(s, i, 5);
        qm0.add_quadratic(x, i, 6);
        qm0.add_quadratic(i, i, 7);
        qm0.set_offset(8);

        auto qm1 = dimod::QuadraticModel<double>();
        auto t = qm1.add_variable(Vartype::SPIN);
        auto y = qm1.add_variable(Vartype::BINARY);
        auto j = qm1.add_variable(Vartype::INTEGER, -10, 10);

        qm1.set_linear(t, -1);
        qm1.set_linear(y, -2);
        qm1.set_linear(j, -3);
        qm1.add_quadratic(t, y, -4);
        qm1.add_quadratic(t, j, -5);
        qm1.add_quadratic(x, j, -6);
        qm1.add_quadratic(j, j, -7);

        WHEN("the swap() method is called") {
            std::swap(qm0, qm1);

            THEN("their contents are swapped") {
                CHECK(qm0.linear(s) == -1);
                CHECK(qm1.linear(t) == +1);

                CHECK(qm0.quadratic(t, y) == -4);
                CHECK(qm1.quadratic(s, x) == +4);

                CHECK(qm0.offset() == 0);
                CHECK(qm1.offset() == 8);

                CHECK(qm0.lower_bound(i) == -10);
                CHECK(qm1.upper_bound(j) == 5);
            }
        }
    }
}

SCENARIO("quadratic models can have their interactions filtered") {
    GIVEN("a binary quadratic model") {
        auto bqm = dimod::BinaryQuadraticModel<double>(5, dimod::Vartype::BINARY);
        bqm.add_quadratic(0, 1, 1.5);
        bqm.add_quadratic(1, 2, 2.5);
        bqm.add_quadratic(2, 3, 3.5);

        WHEN("We filter all interactions with variable 1") {
            CHECK(bqm.remove_interactions([](int u, int v, double) { return u == 1 || v == 1; }) ==
                  2);

            CHECK(bqm.quadratic(0, 1) == 0);
            CHECK(bqm.quadratic(1, 2) == 0);
            CHECK(bqm.quadratic(2, 3) == 3.5);

            CHECK(bqm.quadratic(1, 0) == 0);
            CHECK(bqm.quadratic(2, 1) == 0);
            CHECK(bqm.quadratic(3, 2) == 3.5);
        }
    }
}
}  // namespace dimod
