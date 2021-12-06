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

#include "../Catch2/single_include/catch2/catch.hpp"
#include "dimod/quadratic_model.h"

namespace dimod {

TEMPLATE_TEST_CASE_SIG("Scenario: BinaryQuadraticModel tests", "[qmbase][bqm]",
                       ((typename Bias, Vartype vartype), Bias, vartype),
                       (double, Vartype::BINARY), (double, Vartype::SPIN),
                       (float, Vartype::BINARY), (float, Vartype::SPIN)) {
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

        AND_GIVEN("a scale factor for biases and offset") {
            double scale_factor = 5.5;

            WHEN("we call scale on a filled BQM") {
                bqm.offset() = 2;
                bqm.resize(3);
                bqm.linear(0) = 1;
                bqm.linear(1) = 2;
                bqm.linear(2) = 3;
                bqm.set_quadratic(0,1,1);
                bqm.set_quadratic(1,2,2);

                bqm.scale(scale_factor);

                THEN("all biases, interaction, and offset are scaled") {
                    REQUIRE(bqm.offset() == 11.0);
                    REQUIRE(bqm.linear(0) == 5.5);
                    REQUIRE(bqm.linear(1) == 11.0);
                    REQUIRE(bqm.linear(2) == 16.5);
                    REQUIRE(bqm.quadratic(0,1) == 5.5);
                    REQUIRE(bqm.quadratic(1,2) == 11.0);
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
                    REQUIRE_THROWS_AS(bqm.quadratic_at(0, 2),
                                      std::out_of_range);
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
                    REQUIRE_THROWS_AS(bqm.quadratic_at(0, 2),
                                      std::out_of_range);
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
                    REQUIRE_THROWS_AS(bqm.quadratic_at(0, 2),
                                      std::out_of_range);
                    REQUIRE_THROWS_AS(bqm.quadratic_at(2, 0),
                                      std::out_of_range);
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

        THEN("we can iterate over the neighborhood") {
            auto span = bqm.neighborhood(0);
            auto pairs = std::vector<std::pair<std::size_t, Bias>>(span.first,
                                                                   span.second);

            REQUIRE(pairs[0].first == 1);
            REQUIRE(pairs[0].second == 2);
            REQUIRE(pairs[1].first == 2);
            REQUIRE(pairs[1].second == 4);
            REQUIRE(pairs.size() == 2);
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
        bqm.linear(0) = 1;
        bqm.linear(1) = -3.25;
        bqm.linear(2) = 0;
        bqm.linear(3) = 3;
        bqm.linear(4) = -4.5;
        bqm.set_quadratic(0, 3, -1);
        bqm.set_quadratic(3, 1, 5.6);
        bqm.set_quadratic(0, 1, 1.6);
        bqm.offset() = -3.8;

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
                    energies.push_back(bqm.energy(sample));
                }
            } else {
                for (auto& sample : bin_samples) {
                    energies.push_back(bqm.energy(sample));
                }
            }

            WHEN("we change the vartype to spin") {
                bqm.change_vartype(Vartype::SPIN);

                THEN("the energies will match") {
                    for (size_t si = 0; si < energies.size(); ++si) {
                        REQUIRE(energies[si] ==
                                Approx(bqm.energy(spn_samples[si])));
                    }
                }
            }

            WHEN("we change the vartype to binary") {
                bqm.change_vartype(Vartype::BINARY);
                THEN("the energies will match") {
                    for (size_t si = 0; si < energies.size(); ++si) {
                        REQUIRE(energies[si] ==
                                Approx(bqm.energy(bin_samples[si])));
                    }
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_SIG(
        "Scenario: BQMs can be combined", "[bqm]",
        ((typename B0, typename B1, Vartype vartype), B0, B1, vartype),
        (float, float, Vartype::BINARY), (double, float, Vartype::BINARY),
        (float, double, Vartype::BINARY), (double, double, Vartype::BINARY),
        (float, float, Vartype::SPIN), (double, float, Vartype::SPIN),
        (float, double, Vartype::SPIN), (double, double, Vartype::SPIN)) {
    GIVEN("a BQM with 3 variables") {
        auto bqm0 = BinaryQuadraticModel<B0>(3, vartype);
        bqm0.linear(2) = -1;
        bqm0.set_quadratic(0, 1, 1.5);
        bqm0.set_quadratic(0, 2, -2);
        bqm0.set_quadratic(1, 2, 7);
        bqm0.offset() = -4;

        AND_GIVEN("a BQM with 5 variables and the same vartype") {
            auto bqm1 = BinaryQuadraticModel<B1>(5, vartype);
            bqm1.linear(0) = 1;
            bqm1.linear(1) = -3.25;
            bqm1.linear(2) = 2;
            bqm1.linear(3) = 3;
            bqm1.linear(4) = -4.5;
            bqm1.set_quadratic(0, 1, 5.6);
            bqm1.set_quadratic(0, 3, -1);
            bqm1.set_quadratic(1, 2, 1.6);
            bqm1.set_quadratic(3, 4, -25);
            bqm1.offset() = -3.8;

            WHEN("the first is updated with the second") {
                bqm0.add_bqm(bqm1);

                THEN("the biases are added") {
                    REQUIRE(bqm0.num_variables() == 5);
                    REQUIRE(bqm0.num_interactions() == 5);

                    CHECK(bqm0.offset() == Approx(-7.8));

                    CHECK(bqm0.linear(0) == Approx(1));
                    CHECK(bqm0.linear(1) == Approx(-3.25));
                    CHECK(bqm0.linear(2) == Approx(1));
                    CHECK(bqm0.linear(3) == Approx(3));
                    CHECK(bqm0.linear(4) == Approx(-4.5));

                    CHECK(bqm0.quadratic(0, 1) == Approx(7.1));
                    CHECK(bqm0.quadratic(0, 2) == Approx(-2));
                    CHECK(bqm0.quadratic(0, 3) == Approx(-1));
                    CHECK(bqm0.quadratic(1, 2) == Approx(8.6));
                    CHECK(bqm0.quadratic(3, 4) == Approx(-25));
                }
            }

            WHEN("the second is updated with the first") {
                bqm1.add_bqm(bqm0);

                THEN("the biases are added") {
                    REQUIRE(bqm1.num_variables() == 5);
                    REQUIRE(bqm1.num_interactions() == 5);

                    CHECK(bqm1.offset() == Approx(-7.8));

                    CHECK(bqm1.linear(0) == Approx(1));
                    CHECK(bqm1.linear(1) == Approx(-3.25));
                    CHECK(bqm1.linear(2) == Approx(1));
                    CHECK(bqm1.linear(3) == Approx(3));
                    CHECK(bqm1.linear(4) == Approx(-4.5));

                    CHECK(bqm1.quadratic(0, 1) == Approx(7.1));
                    CHECK(bqm1.quadratic(0, 2) == Approx(-2));
                    CHECK(bqm1.quadratic(0, 3) == Approx(-1));
                    CHECK(bqm1.quadratic(1, 2) == Approx(8.6));
                    CHECK(bqm1.quadratic(3, 4) == Approx(-25));
                }
            }
        }

        AND_GIVEN("a BQM with 5 variables and a different vartype") {
            Vartype vt;
            if (vartype == Vartype::SPIN) {
                vt = Vartype::BINARY;
            } else {
                vt = Vartype::SPIN;
            }

            auto bqm1 = BinaryQuadraticModel<B1>(5, vt);
            bqm1.linear(0) = 1;
            bqm1.linear(1) = -3.25;
            bqm1.linear(2) = 2;
            bqm1.linear(3) = 3;
            bqm1.linear(4) = -4.5;
            bqm1.set_quadratic(0, 1, 5.6);
            bqm1.set_quadratic(0, 3, -1);
            bqm1.set_quadratic(1, 2, 1.6);
            bqm1.set_quadratic(3, 4, -25);
            bqm1.offset() = -3.8;

            WHEN("the first is updated with the second") {
                auto bqm0_cp = BinaryQuadraticModel<B0>(bqm0);
                auto bqm1_cp = BinaryQuadraticModel<B1>(bqm1);

                bqm0.add_bqm(bqm1);

                THEN("it was as if the vartype was changed first") {
                    bqm1_cp.change_vartype(vartype);
                    bqm0_cp.add_bqm(bqm1_cp);

                    REQUIRE(bqm0.num_variables() == bqm0_cp.num_variables());
                    REQUIRE(bqm0.num_interactions() ==
                            bqm0_cp.num_interactions());
                    REQUIRE(bqm0.offset() == Approx(bqm0_cp.offset()));
                    for (auto u = 0u; u < bqm0.num_variables(); ++u) {
                        REQUIRE(bqm0.linear(u) == Approx(bqm0_cp.linear(u)));

                        auto span = bqm0.neighborhood(u);
                        for (auto it = span.first; it != span.second; ++it) {
                            REQUIRE((*it).second == Approx(bqm0_cp.quadratic_at(
                                                            u, (*it).first)));
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("One bqm can be added to another") {
    GIVEN("Two BQMs of different vartypes") {
        auto bin = BinaryQuadraticModel<double>(2, Vartype::BINARY);
        bin.linear(0) = .3;
        bin.set_quadratic(0, 1, -1);

        auto spn = BinaryQuadraticModel<double>(2, Vartype::SPIN);
        spn.linear(1) = -1;
        spn.set_quadratic(0, 1, 1);
        spn.offset() = 1.2;

        WHEN("the spin one is added to the binary") {
            bin.add_bqm(spn);

            THEN("the combined model is correct") {
                REQUIRE(bin.num_variables() == 2);
                CHECK(bin.num_interactions() == 1);

                CHECK(bin.linear(0) == -1.7);
                CHECK(bin.linear(1) == -4);

                CHECK(bin.quadratic(0, 1) == 3);
            }
        }

        WHEN("the spin one is added to the binary one with an offset") {
            std::vector<int> mapping = {1, 2};
            bin.add_bqm(spn, mapping);

            THEN("the combined model is correct") {
                REQUIRE(bin.num_variables() == 3);
                CHECK(bin.num_interactions() == 2);

                CHECK(bin.linear(0) == .3);
                CHECK(bin.linear(1) == -2);
                CHECK(bin.linear(2) == -4);

                CHECK(bin.quadratic(0, 1) == -1);
                CHECK(bin.quadratic(1, 2) == 4);
                CHECK_THROWS_AS(bin.quadratic_at(0, 2), std::out_of_range);
            }
        }
    }

    GIVEN("Two BQMs of the same vartype") {
        auto bqm0 = BinaryQuadraticModel<double>(2, Vartype::SPIN);
        bqm0.linear(0) = -1;
        bqm0.set_quadratic(0, 1, 1.5);
        bqm0.offset() = 1.5;

        auto bqm1 = BinaryQuadraticModel<double>(3, Vartype::SPIN);
        bqm1.linear(0) = -2;
        bqm1.linear(2) = 3;
        bqm1.set_quadratic(0, 1, 5);
        bqm1.set_quadratic(2, 1, 1);
        bqm1.offset() = 1.5;

        WHEN("the spin one is added to the binary one with a mapping") {
            std::vector<int> mapping = {0, 1, 2};
            bqm0.add_bqm(bqm1, mapping);

            THEN("the biases are summed") {
                REQUIRE(bqm0.num_variables() == 3);
                CHECK(bqm0.num_interactions() == 2);

                CHECK(bqm0.offset() == 3);

                CHECK(bqm0.linear(0) == -3);
                CHECK(bqm0.linear(1) == 0);
                CHECK(bqm0.linear(2) == 3);

                CHECK(bqm0.quadratic(0, 1) == 6.5);
                CHECK(bqm0.quadratic(1, 2) == 1);
                CHECK_THROWS_AS(bqm0.quadratic_at(0, 2), std::out_of_range);
            }
        }
    }
}

SCENARIO("Neighborhood can be manipulated") {
    GIVEN("An empty Neighborhood") {
        auto neighborhood = Neighborhood<float, size_t>();

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
                std::vector<std::pair<size_t, float>> pairs(
                        neighborhood.begin(), neighborhood.end());

                REQUIRE(pairs[0].first == 0);
                REQUIRE(pairs[0].second == .5);
                REQUIRE(pairs[1].first == 1);
                REQUIRE(pairs[1].second == 1.5);
                REQUIRE(pairs[2].first == 3);
                REQUIRE(pairs[2].second == -3);
            }

            THEN("we can create a vector from the const neighborhood") {
                std::vector<std::pair<size_t, float>> pairs(
                        neighborhood.cbegin(), neighborhood.cend());

                REQUIRE(pairs[0].first == 0);
                REQUIRE(pairs[0].second == .5);
                REQUIRE(pairs[1].first == 1);
                REQUIRE(pairs[1].second == 1.5);
                REQUIRE(pairs[2].first == 3);
                REQUIRE(pairs[2].second == -3);
            }

            THEN("we can modify the biases via the iterator") {
                auto it = neighborhood.begin();

                (*it).second = 18;
                REQUIRE(neighborhood.at(0) == 18);

                it++;
                (*it).second = -48;
                REQUIRE(neighborhood.at(1) == -48);

                ++it;
                it->second = 104;
                REQUIRE(neighborhood.at(3) == 104);
            }

            THEN("we can erase some with an iterator") {
                neighborhood.erase(neighborhood.begin() + 1,
                                   neighborhood.end());
                REQUIRE(neighborhood.size() == 1);
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
                CHECK_THROWS_AS(qm.quadratic_at(v_int, v_bin),
                                std::out_of_range);
                CHECK_THROWS_AS(qm.quadratic_at(v_int, v_spn),
                                std::out_of_range);
                CHECK_THROWS_AS(qm.quadratic_at(v_bin, v_spn),
                                std::out_of_range);
            }

            AND_WHEN("we set some quadratic biases") {
                qm.set_quadratic(v_int, v_bin, 1.5);
                qm.set_quadratic(v_bin, v_spn, -3);

                THEN("we can read them back out") {
                    REQUIRE(qm.num_variables() == 3);
                    CHECK(qm.quadratic(v_int, v_bin) == 1.5);
                    CHECK(qm.quadratic(v_bin, v_spn) == -3);
                    CHECK_THROWS_AS(qm.quadratic_at(v_int, v_spn),
                                    std::out_of_range);
                }
            }

            AND_WHEN("we set some quadratic biases on self-loops") {
                qm.set_quadratic(v_int, v_int, 1.5);
                CHECK_THROWS_AS(qm.set_quadratic(v_bin, v_bin, -3),
                                std::domain_error);
                CHECK_THROWS_AS(qm.set_quadratic(v_spn, v_spn, -3),
                                std::domain_error);

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
            THEN("we can retrieve the quadratic bias") {
                CHECK(qm.quadratic(v, v) == 1.5);
            }

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
            THEN("we can retrieve the quadratic bias") {
                CHECK(qm.quadratic(v, v) == 1.5);
            }

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
              CHECK(qm.energy(samples) == Approx(1.8446744069414584e+19));
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

SCENARIO("A quadratic model can be constructed from a binary quadratic model",
         "[bqm]") {
    GIVEN("A binary quadratic model") {
        auto bqm = BinaryQuadraticModel<float>(3, Vartype::SPIN);
        bqm.linear(0) = 4;
        bqm.linear(2) = -2;
        bqm.set_quadratic(0, 1, 1.5);
        bqm.set_quadratic(1, 2, -3);
        bqm.offset() = 5;

        WHEN("a quadratic model is constructed from it") {
            auto qm = QuadraticModel<float>(bqm);

            THEN("the biases etc are passed in") {
                REQUIRE(qm.num_variables() == 3);

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

SCENARIO("The variables of a quadratic model can have their vartypes changed",
         "[qm]") {
    GIVEN("A quadratic model with a spin and a binary variable") {
        auto qm = QuadraticModel<double>();
        auto s = qm.add_variable(Vartype::SPIN);
        auto x = qm.add_variable(Vartype::BINARY);

        qm.linear(s) = 2;
        qm.linear(x) = 4;
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
            qm.vartype(s) = Vartype::BINARY;
            qm.lower_bound(s) = -2;
            qm.upper_bound(s) = 2;

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

SCENARIO("variables in a quadratic model can be swapped", "[qm]") {
    GIVEN("a quadratic model with a binary, integer, and spin variable") {
        auto qm = dimod::QuadraticModel<double>();
        auto s = qm.add_variable(Vartype::SPIN);
        auto x = qm.add_variable(Vartype::BINARY);
        auto i = qm.add_variable(Vartype::INTEGER, -5, 5);

        qm.linear(s) = 1;
        qm.linear(x) = 2;
        qm.linear(i) = 3;

        AND_GIVEN("that all the variables have interactions") {
            qm.set_quadratic(s, x, 4);
            qm.set_quadratic(s, i, 5);
            qm.set_quadratic(x, i, 6);

            WHEN("we swap two of the variables") {
                qm.swap_variables(s, i);

                THEN("everything moves accordingly") {
                    CHECK(qm.linear(i) == 1);
                    CHECK(qm.linear(x) == 2);
                    CHECK(qm.linear(s) == 3);
                    CHECK(qm.quadratic(i, x) == 4);
                    CHECK(qm.quadratic(x, i) == 4);
                    CHECK(qm.quadratic(s, i) == 5);
                    CHECK(qm.quadratic(i, s) == 5);
                    CHECK(qm.quadratic(s, x) == 6);
                    CHECK(qm.quadratic(x, s) == 6);
                    CHECK(qm.lower_bound(s) == -5);
                    CHECK(qm.upper_bound(s) == 5);
                    CHECK(qm.lower_bound(i) == -1);
                    CHECK(qm.upper_bound(i) == +1);
                    CHECK(qm.vartype(s) == Vartype::INTEGER);
                    CHECK(qm.vartype(x) == Vartype::BINARY);
                    CHECK(qm.vartype(i) == Vartype::SPIN);
                    CHECK(qm.num_interactions(i) == 2);
                    CHECK(qm.num_interactions(x) == 2);
                    CHECK(qm.num_interactions(s) == 2);
                }
            }
        }

        AND_GIVEN("sparse interactions") {
            qm.set_quadratic(s, x, 4);

            WHEN("we swap two of the variables") {
                qm.swap_variables(s, i);

                THEN("everything moves accordingly") {
                    CHECK(qm.linear(i) == 1);
                    CHECK(qm.linear(x) == 2);
                    CHECK(qm.linear(s) == 3);
                    CHECK(qm.quadratic(i, x) == 4);
                    CHECK(qm.quadratic(x, i) == 4);
                    CHECK_THROWS_AS(qm.quadratic_at(s, i), std::out_of_range);
                    CHECK_THROWS_AS(qm.quadratic_at(i, s), std::out_of_range);
                    CHECK_THROWS_AS(qm.quadratic_at(s, x), std::out_of_range);
                    CHECK_THROWS_AS(qm.quadratic_at(x, s), std::out_of_range);
                    CHECK(qm.lower_bound(s) == -5);
                    CHECK(qm.upper_bound(s) == 5);
                    CHECK(qm.lower_bound(i) == -1);
                    CHECK(qm.upper_bound(i) == +1);
                    CHECK(qm.vartype(s) == Vartype::INTEGER);
                    CHECK(qm.vartype(x) == Vartype::BINARY);
                    CHECK(qm.vartype(i) == Vartype::SPIN);
                    CHECK(qm.num_interactions(i) == 1);
                    CHECK(qm.num_interactions(x) == 1);
                    CHECK(qm.num_interactions(s) == 0);
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE(
        "Scenario: the size of quadratic models in bytes can be determined",
        "[qm]", double, float) {
    GIVEN("a binary quadratic model") {
        auto bqm = dimod::BinaryQuadraticModel<TestType>(
                5, dimod::Vartype::BINARY);
        bqm.add_quadratic(0, 1, 1.5);
        bqm.add_quadratic(1, 2, 1.5);
        bqm.add_quadratic(2, 3, 1.5);

        THEN("we can determine the number of bytes used by the elements") {
            CHECK(bqm.nbytes() ==
                  (bqm.num_variables() + 2 * bqm.num_interactions()) *
                                  sizeof(TestType) +
                          2 * bqm.num_interactions() * sizeof(int) + sizeof(TestType));
            CHECK(bqm.nbytes(true) >= bqm.nbytes());
        }

        AND_GIVEN("a quadratic model") {
            auto qm = dimod::QuadraticModel<TestType>(bqm);

            THEN("we can determine the number of bytes used by the elements") {
                CHECK(qm.nbytes() ==
                      (qm.num_variables() + 2 * qm.num_interactions()) *
                                      sizeof(TestType) +
                              2 * qm.num_interactions() * sizeof(int) +
                              qm.num_variables() *
                                      sizeof(dimod::VarInfo<TestType>) +
                              sizeof(TestType));
                CHECK(qm.nbytes(true) >= qm.nbytes());
            }
        }
    }
}
}  // namespace dimod
