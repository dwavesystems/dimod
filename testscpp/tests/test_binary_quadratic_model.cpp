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
#include "dimod/binary_quadratic_model.h"

namespace dimod {

// todo: reorganize other tests
TEST_CASE("BinaryQuadraticModel tests") {
    GIVEN("a linear BQM") {
        auto bqm = BinaryQuadraticModel<double>(5, Vartype::SPIN);
        bqm.set_offset(5);
        bqm.set_linear(0, {0, -1, +2, -3, +4});

        WHEN("we use remove_variable()") {
            bqm.remove_variable(3);

            THEN("the variable is removed and the model is reindexed") {
                REQUIRE(bqm.num_variables() == 4);
                REQUIRE(bqm.num_interactions() == 0);
                CHECK(bqm.linear(0) == 0);
                CHECK(bqm.linear(1) == -1);
                CHECK(bqm.linear(2) == 2);
                CHECK(bqm.linear(3) == 4);  // this was reindexed
                CHECK(bqm.offset() == 5);
            }
        }

        WHEN("we use fix_variable()") {
            bqm.fix_variable(2, -1);
            THEN("the variable is removed, its biases distributed and the model is reindexed") {
                REQUIRE(bqm.num_variables() == 4);
                REQUIRE(bqm.num_interactions() == 0);
                CHECK(bqm.offset() == 3);  // linear biases -1*2 was added to offset
                CHECK(bqm.linear(0) == 0);
                CHECK(bqm.linear(1) == -1);
                CHECK(bqm.linear(2) == -3);  // this was reindexed
                CHECK(bqm.linear(3) == 4);   // this was reindexed
            }
        }

        WHEN("we use energy()") {
            auto sample = std::vector<int>{0, 1, 1, 0, 1};
            double energy = bqm.energy(sample.begin());
            THEN("the energy is calculated appropriately") {
                CHECK(energy == -1 + 2 + 4 + bqm.offset());
            }
        }
    }

    GIVEN("a sparse quadratic BQM") {
        auto bqm = BinaryQuadraticModel<double>(5, Vartype::SPIN);
        bqm.set_offset(5);
        bqm.set_linear(0, {0, -1, +2, -3, +4});
        bqm.add_quadratic({0, 1, 2, 3}, {1, 2, 3, 4}, {1, 12, 23, 34});

        WHEN("we use remove_variable()") {
            bqm.remove_variable(2);

            THEN("everything is reindexed") {
                REQUIRE(bqm.num_variables() == 4);
                REQUIRE(bqm.num_interactions() == 2);
                CHECK(bqm.linear(0) == 0);
                CHECK(bqm.linear(1) == -1);
                CHECK(bqm.linear(2) == -3);  // this was reindexed
                CHECK(bqm.linear(3) == 4);   // this was reindexed
                CHECK(bqm.quadratic(0, 1) == 1);
                CHECK(bqm.quadratic(2, 3) == 34);  // this was reindexed
                CHECK(bqm.offset() == 5);
            }
        }

        AND_GIVEN("another identical BQM") {
            auto bqm2 = BinaryQuadraticModel<double>(5, Vartype::SPIN);
            bqm2.set_offset(5);
            bqm2.set_linear(0, {0, -1, +2, -3, +4});
            bqm2.add_quadratic({0, 1, 2, 3}, {1, 2, 3, 4}, {1, 12, 23, 34});

            THEN("they test equal") {
                CHECK(bqm.is_equal(bqm2));
            }
        }

        WHEN("we use resize() to shrink the BQM") {
            bqm.resize(3);

            THEN("the variables and interactions are updated") {
                REQUIRE(bqm.num_variables() == 3);
                REQUIRE(bqm.num_interactions() == 2);
                CHECK(bqm.linear(0) == 0);
                CHECK(bqm.linear(1) == -1);
                CHECK(bqm.linear(2) == 2);
                CHECK(bqm.quadratic(0, 1) == 1);
                CHECK(bqm.quadratic(1, 2) == 12);
            }
        }
    }

    GIVEN("a clique BQM") {
        auto bqm = BinaryQuadraticModel<double>(5, Vartype::SPIN);
        bqm.set_offset(5);
        bqm.set_linear(0, {0, -1, +2, -3, +4});
        for (size_t u = 0; u < bqm.num_variables(); ++u) {
            for (size_t v = u + 1; v < bqm.num_variables(); ++v) {
                bqm.add_quadratic(u, v, u * 10 + v);
            }
        }

        WHEN("we use remove_variable()") {
            bqm.remove_variable(3);

            THEN("everything is reindexed") {
                REQUIRE(bqm.num_variables() == 4);
                REQUIRE(bqm.num_interactions() == 6);
                CHECK(bqm.linear(0) == 0);
                CHECK(bqm.linear(1) == -1);
                CHECK(bqm.linear(2) == 2);
                CHECK(bqm.linear(3) == 4);  // this was reindexed
                CHECK(bqm.quadratic(0, 1) == 1);
                CHECK(bqm.quadratic(0, 2) == 2);
                CHECK(bqm.quadratic(0, 3) == 4);
                CHECK(bqm.quadratic(1, 2) == 12);
                CHECK(bqm.quadratic(1, 3) == 14);  // this was reindexed
                CHECK(bqm.quadratic(2, 3) == 24);  // this was reindexed
                CHECK(bqm.offset() == 5);
            }
        }
    }
}

TEST_CASE("BinaryQuadraticModel constructors") {
    GIVEN("a linear BQM") {
        auto bqm = BinaryQuadraticModel<double>(5, Vartype::SPIN);
        bqm.set_linear(0, {0, -1, +2, -3, +4});
        bqm.set_offset(5);

        AND_GIVEN("another BQM with the same values") {
            auto expected = BinaryQuadraticModel<double>(5, Vartype::SPIN);
            expected.set_linear(0, {0, -1, +2, -3, +4});
            expected.set_offset(5);

            WHEN("another BQM is constructed from it using the assignment operator") {
                auto cpy = BinaryQuadraticModel<double>(bqm);

                THEN("all of the values are copied appropriately") {
                    CHECK(cpy.is_equal(expected));
                }

                AND_WHEN("the original BQM is modified") {
                    bqm.set_linear(1, 1);
                    bqm.add_quadratic(1, 2, 2);
                    bqm.set_offset(-5);

                    THEN("the copy is not also modified") {
                        CHECK(cpy.is_equal(expected));
                        CHECK(!bqm.is_equal(expected));  // sanity check
                    }
                }
            }

            WHEN("another BQM is constructed from it using the copy constructor") {
                BinaryQuadraticModel<double> cpy(bqm);

                THEN("all of the values are copied appropriately") {
                    CHECK(cpy.is_equal(expected));
                }

                AND_WHEN("the original BQM is modified") {
                    bqm.set_linear(1, 1);
                    bqm.add_quadratic(1, 2, 2);
                    bqm.set_offset(-5);

                    THEN("the copy is not also modified") {
                        CHECK(cpy.is_equal(expected));
                        CHECK(!bqm.is_equal(expected));  // sanity check
                    }
                }
            }
        }
    }
}

TEST_CASE("BinaryQuadraticModel quadratic iteration") {
    GIVEN("a linear BQM") {
        auto bqm = BinaryQuadraticModel<double>(5, Vartype::SPIN);
        bqm.set_linear(0, {0, -1, +2, -3, +4});
        bqm.set_offset(5);

        THEN("quadric iteration should return nothing") {
            CHECK(bqm.cbegin_quadratic() == bqm.cend_quadratic());
        }

        THEN("neighborhood iteration should return nothing") {
            for (std::size_t v = 0; v < bqm.num_variables(); ++v) {
                CHECK(bqm.cbegin_neighborhood(v) == bqm.cend_neighborhood(v));
            }
        }
    }

    GIVEN("a linear BQM that was once quadratic") {
        // as a technical detail, as of August 2022, this leaves the structure
        // for an adjacency in-place. If that changes in the future and this
        // test isn't removed then it does no harm
        auto bqm = BinaryQuadraticModel<double>(5, Vartype::SPIN);
        bqm.set_linear(0, {0, -1, +2, -3, +4});
        bqm.set_offset(5);
        bqm.add_quadratic(0, 1, 1);
        bqm.remove_interaction(0, 1);

        THEN("quadric iteration should return nothing") {
            CHECK(bqm.cbegin_quadratic() == bqm.cend_quadratic());
        }
    }

    GIVEN("a BQM with one quadratic interaction") {
        auto bqm = BinaryQuadraticModel<double>(5, Vartype::SPIN);
        bqm.set_linear(0, {0, -1, +2, -3, +4});
        bqm.set_offset(5);
        bqm.add_quadratic(1, 3, 5);

        THEN("our quadratic iterator has one value in it") {
            std::vector<int> row;
            std::vector<int> col;
            std::vector<double> biases;
            for (auto it = bqm.cbegin_quadratic(); it != bqm.cend_quadratic(); ++it) {
                row.push_back(it->u);
                col.push_back(it->v);
                biases.push_back(it->bias);
            }
            CHECK(row == std::vector<int>{3});
            CHECK(col == std::vector<int>{1});
            CHECK(biases == std::vector<double>{5});
        }
    }

    GIVEN("a BQM with several quadratic interactions") {
        auto bqm = BinaryQuadraticModel<double>(5, Vartype::SPIN);
        bqm.set_linear(0, {0, -1, +2, -3, +4});
        bqm.set_offset(5);
        bqm.add_quadratic({0, 0, 0, 3, 3}, {2, 3, 4, 4, 1}, {1, 2, 3, 4, 5});

        THEN("our quadratic iterator will correctly return the lower triangle") {
            std::vector<int> row;
            std::vector<int> col;
            std::vector<double> biases;
            // also sneak in a post-fix operator test
            for (auto it = bqm.cbegin_quadratic(); it != bqm.cend_quadratic(); it++) {
                row.push_back(it->u);
                col.push_back(it->v);
                biases.push_back(it->bias);
            }
            CHECK(row == std::vector<int>{2, 3, 3, 4, 4});
            CHECK(col == std::vector<int>{0, 0, 1, 0, 3});
            CHECK(biases == std::vector<double>{1, 2, 5, 3, 4});
        }
    }
}

TEST_CASE("BinaryQuadraticModel scale") {
    GIVEN("a bqm with linear, quadratic interactions and an offset") {
        auto bqm = BinaryQuadraticModel<double>(3, Vartype::BINARY);
        bqm.set_offset(2);
        bqm.set_linear(0, {1, 2, 3});
        bqm.set_quadratic(0, 1, 1);
        bqm.set_quadratic(1, 2, 2);

        WHEN("we scale it") {
            bqm.scale(5.5);

            THEN("all biases and the offset are scaled") {
                REQUIRE(bqm.offset() == 11.0);
                REQUIRE(bqm.linear(0) == 5.5);
                REQUIRE(bqm.linear(1) == 11.0);
                REQUIRE(bqm.linear(2) == 16.5);
                REQUIRE(bqm.quadratic(0, 1) == 5.5);
                REQUIRE(bqm.quadratic(1, 2) == 11.0);
            }
        }
    }
}

TEST_CASE("BinaryQuadraticModel vartype and bounds") {
    GIVEN("an empty BINARY BQM") {
        auto bqm = BinaryQuadraticModel<double>(Vartype::BINARY);

        THEN("the vartype and bounds are binary") {
            CHECK(bqm.vartype() == Vartype::BINARY);
            CHECK(bqm.lower_bound() == 0);
            CHECK(bqm.upper_bound() == 1);
        }
    }

    GIVEN("an empty SPIN BQM") {
        auto bqm = BinaryQuadraticModel<double>(Vartype::SPIN);

        THEN("the vartype and bounds are spin") {
            CHECK(bqm.vartype() == Vartype::SPIN);
            CHECK(bqm.lower_bound() == -1);
            CHECK(bqm.upper_bound() == +1);
        }
    }

    GIVEN("an empty default BQM") {
        auto bqm = BinaryQuadraticModel<double>();

        THEN("the vartype and bounds are binary") {
            CHECK(bqm.vartype() == Vartype::BINARY);
            CHECK(bqm.lower_bound() == 0);
            CHECK(bqm.upper_bound() == 1);
        }
    }
}

TEST_CASE("BinaryQuadraticModel swap") {
    GIVEN("two BQMs of different vartypes") {
        auto bqm1 = BinaryQuadraticModel<double>(3, Vartype::BINARY);
        bqm1.set_offset(2);
        bqm1.set_linear(0, {1, 2, 3});
        bqm1.set_quadratic(0, 1, 1);
        bqm1.set_quadratic(1, 2, 2);

        auto bqm2 = BinaryQuadraticModel<double>(3, Vartype::SPIN);
        bqm2.set_offset(-2);
        bqm2.set_linear(0, {-1, -2, -3});
        bqm2.set_quadratic(0, 1, -1);
        bqm2.set_quadratic(1, 2, -2);

        WHEN("we swap them") {
            std::swap(bqm1, bqm2);

            THEN("everything is moved appropriately") {
                CHECK(bqm1.linear(0) == -1);
                CHECK(bqm1.linear(1) == -2);
                CHECK(bqm1.linear(2) == -3);
                CHECK(bqm2.linear(0) == 1);
                CHECK(bqm2.linear(1) == 2);
                CHECK(bqm2.linear(2) == 3);

                CHECK(bqm1.quadratic(0, 1) == -1);
                CHECK(bqm1.quadratic(1, 2) == -2);
                CHECK(bqm2.quadratic(0, 1) == 1);
                CHECK(bqm2.quadratic(1, 2) == 2);

                CHECK(bqm1.offset() == -2);
                CHECK(bqm2.offset() == 2);

                CHECK(bqm1.vartype() == Vartype::SPIN);
                CHECK(bqm2.vartype() == Vartype::BINARY);
            }
        }
    }
}

TEMPLATE_TEST_CASE_SIG("Scenario: BinaryQuadraticModel tests", "[qmbase][bqm]",
                       ((typename Bias, Vartype vartype), Bias, vartype), (double, Vartype::BINARY),
                       (double, Vartype::SPIN), (float, Vartype::BINARY), (float, Vartype::SPIN)) {
    GIVEN("an empty BQM") {
        auto bqm = BinaryQuadraticModel<Bias>(vartype);

        WHEN("the bqm is resized") {
            bqm.resize(10);

            THEN("it will have the correct number of variables with 0 bias") {
                REQUIRE(bqm.num_variables() == 10);
                REQUIRE(bqm.num_interactions() == 0);
                for (auto v = 0u; v < bqm.num_variables(); ++v) {
                    REQUIRE(bqm.linear(v) == 0);
                }
            }
        }

        AND_GIVEN("some COO-formatted arrays") {
            int irow[4] = {0, 2, 0, 1};
            int icol[4] = {0, 2, 1, 2};
            float bias[4] = {.5, -2, 2, -3};
            std::size_t length = 4;

            WHEN("we add the biases with add_quadratic") {
                bqm.add_quadratic(&irow[0], &icol[0], &bias[0], length);

                THEN("it takes its values from the arrays") {
                    REQUIRE(bqm.num_variables() == 3);

                    if (bqm.vartype() == Vartype::SPIN) {
                        REQUIRE(bqm.linear(0) == 0);
                        REQUIRE(bqm.linear(1) == 0);
                        REQUIRE(bqm.linear(2) == 0);
                        REQUIRE(bqm.offset() == -1.5);
                    } else {
                        REQUIRE(bqm.vartype() == Vartype::BINARY);
                        REQUIRE(bqm.linear(0) == .5);
                        REQUIRE(bqm.linear(1) == 0);
                        REQUIRE(bqm.linear(2) == -2);
                        REQUIRE(bqm.offset() == 0);
                    }

                    REQUIRE(bqm.num_interactions() == 2);
                    REQUIRE(bqm.quadratic(0, 1) == 2);
                    REQUIRE(bqm.quadratic(2, 1) == -3);
                    REQUIRE_THROWS_AS(bqm.quadratic_at(0, 2), std::out_of_range);
                }
            }
        }

        AND_GIVEN("some COO-formatted arrays with duplicates") {
            int irow[6] = {0, 2, 0, 1, 0, 0};
            int icol[6] = {0, 2, 1, 2, 1, 0};
            float bias[6] = {.5, -2, 2, -3, 4, 1};
            std::size_t length = 6;

            WHEN("we add the biases with add_quadratic") {
                bqm.add_quadratic(&irow[0], &icol[0], &bias[0], length);

                THEN("it combines duplicate values") {
                    REQUIRE(bqm.num_variables() == 3);

                    if (bqm.vartype() == Vartype::SPIN) {
                        REQUIRE(bqm.linear(0) == 0);
                        REQUIRE(bqm.linear(1) == 0);
                        REQUIRE(bqm.linear(2) == 0);
                        REQUIRE(bqm.offset() == -.5);
                    } else {
                        REQUIRE(bqm.vartype() == Vartype::BINARY);
                        REQUIRE(bqm.linear(0) == 1.5);
                        REQUIRE(bqm.linear(1) == 0.);
                        REQUIRE(bqm.linear(2) == -2);
                        REQUIRE(bqm.offset() == 0);
                    }

                    REQUIRE(bqm.num_interactions() == 2);
                    REQUIRE(bqm.quadratic(0, 1) == 6);
                    REQUIRE(bqm.quadratic(2, 1) == -3);
                    REQUIRE_THROWS_AS(bqm.quadratic_at(0, 2), std::out_of_range);
                }
            }
        }

        AND_GIVEN("some COO-formatted arrays with multiple duplicates") {
            int irow[4] = {0, 1, 0, 1};
            int icol[4] = {1, 2, 1, 0};
            float bias[4] = {-1, 1, -2, -3};
            std::size_t length = 4;

            WHEN("we add the biases with add_quadratic") {
                bqm.add_quadratic(&irow[0], &icol[0], &bias[0], length);

                THEN("it combines duplicate values") {
                    REQUIRE(bqm.num_variables() == 3);
                    REQUIRE(bqm.linear(0) == 0);
                    REQUIRE(bqm.linear(1) == 0);
                    REQUIRE(bqm.linear(2) == 0);

                    REQUIRE(bqm.num_interactions() == 2);
                    REQUIRE(bqm.quadratic(0, 1) == -6);
                    REQUIRE(bqm.quadratic(1, 0) == -6);
                    REQUIRE(bqm.quadratic(2, 1) == 1);
                    REQUIRE(bqm.quadratic(1, 2) == 1);
                    REQUIRE_THROWS_AS(bqm.quadratic_at(0, 2), std::out_of_range);
                    REQUIRE_THROWS_AS(bqm.quadratic_at(2, 0), std::out_of_range);
                }
            }
        }
    }

    GIVEN("a BQM constructed from a dense array") {
        float Q[9] = {1, 0, 3, 2, 1, 0, 1, 0, 0};
        int num_variables = 3;

        auto bqm = BinaryQuadraticModel<Bias>(Q, num_variables, vartype);

        THEN("it handles the diagonal according to its vartype") {
            REQUIRE(bqm.num_variables() == 3);

            if (bqm.vartype() == Vartype::SPIN) {
                REQUIRE(bqm.linear(0) == 0);
                REQUIRE(bqm.linear(1) == 0);
                REQUIRE(bqm.linear(2) == 0);
                REQUIRE(bqm.offset() == 2);
            } else {
                REQUIRE(bqm.vartype() == Vartype::BINARY);
                REQUIRE(bqm.linear(0) == 1);
                REQUIRE(bqm.linear(1) == 1);
                REQUIRE(bqm.linear(2) == 0);
                REQUIRE(bqm.offset() == 0);
            }
        }

        THEN("it gets its quadratic from the off-diagonal") {
            REQUIRE(bqm.num_interactions() == 2);

            // test both forward and backward
            REQUIRE(bqm.quadratic(0, 1) == 2);
            REQUIRE(bqm.quadratic(1, 0) == 2);
            REQUIRE(bqm.quadratic(0, 2) == 4);
            REQUIRE(bqm.quadratic(2, 0) == 4);
            REQUIRE(bqm.quadratic(1, 2) == 0);
            REQUIRE(bqm.quadratic(2, 1) == 0);

            // ignores 0s
            REQUIRE_THROWS_AS(bqm.quadratic_at(1, 2), std::out_of_range);
            REQUIRE_THROWS_AS(bqm.quadratic_at(2, 1), std::out_of_range);
        }

        THEN("we can iterate over the neighborhood of 0") {
            std::size_t count = 0;
            for (auto it = bqm.cbegin_neighborhood(0); it != bqm.cend_neighborhood(0);
                 ++it, ++count) {
                CHECK(bqm.quadratic_at(0, it->v) == it->bias);
            }
            CHECK(count == 2);
        }

        WHEN("we iterate over the quadratic biases") {
            auto first = bqm.cbegin_quadratic();
            auto last = bqm.cend_quadratic();
            THEN("we read out the lower triangle") {
                CHECK(first->u == 1);
                CHECK(first->v == 0);
                CHECK(first->bias == 2);
                CHECK((*first).u == 1);
                CHECK((*first).v == 0);
                CHECK((*first).bias == 2);

                ++first;

                CHECK(first->u == 2);
                CHECK(first->v == 0);
                CHECK(first->bias == 4);

                first++;

                CHECK(first == last);
            }
        }
    }

    GIVEN("a BQM with five variables, two interactions and an offset") {
        auto bqm = BinaryQuadraticModel<Bias>(5, vartype);
        bqm.set_linear(0, {1, -3.25, 0, 3, -4.5});
        bqm.set_quadratic(0, 3, -1);
        bqm.set_quadratic(3, 1, 5.6);
        bqm.set_quadratic(0, 1, 1.6);
        bqm.set_offset(-3.8);

        AND_GIVEN("the set of all possible five variable samples") {
            // there are smarter ways to do this but it's simple
            std::vector<std::vector<int>> spn_samples;
            std::vector<std::vector<int>> bin_samples;
            for (auto i = 0; i < 1 << bqm.num_variables(); ++i) {
                std::vector<int> bin_sample;
                std::vector<int> spn_sample;
                for (size_t v = 0; v < bqm.num_variables(); ++v) {
                    bin_sample.push_back((i >> v) & 1);
                    spn_sample.push_back(2 * ((i >> v) & 1) - 1);
                }

                bin_samples.push_back(bin_sample);
                spn_samples.push_back(spn_sample);
            }

            std::vector<double> energies;
            if (vartype == Vartype::SPIN) {
                for (auto& sample : spn_samples) {
                    energies.push_back(bqm.energy(sample.begin()));
                }
            } else {
                for (auto& sample : bin_samples) {
                    energies.push_back(bqm.energy(sample.begin()));
                }
            }

            WHEN("we change the vartype to spin") {
                bqm.change_vartype(Vartype::SPIN);
                CHECK(bqm.vartype() == Vartype::SPIN);
                THEN("the energies will match") {
                    for (size_t si = 0; si < energies.size(); ++si) {
                        REQUIRE(energies[si] == Approx(bqm.energy(spn_samples[si].begin())));
                    }
                }
            }

            WHEN("we change the vartype to binary") {
                bqm.change_vartype(Vartype::BINARY);
                CHECK(bqm.vartype() == Vartype::BINARY);
                THEN("the energies will match") {
                    for (size_t si = 0; si < energies.size(); ++si) {
                        REQUIRE(energies[si] == Approx(bqm.energy(bin_samples[si].begin())));
                    }
                }
            }
        }
    }
}
}  // namespace dimod
