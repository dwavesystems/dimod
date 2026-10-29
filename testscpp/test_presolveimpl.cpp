// Copyright 2023 D-Wave Systems Inc.
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
#include "dimod/constrained_quadratic_model.h"
#include "presolveimpl.hpp"

namespace dwave {

using PresolverImpl = presolve::PresolverImpl<double, int, double>;
using ConstrainedQuadraticModel = dimod::ConstrainedQuadraticModel<double, int>;

TEST_CASE("Test construction", "[presolve][impl]") {
    SECTION("Default constructor") {
        auto pre = PresolverImpl();
        CHECK_FALSE(pre.model().num_variables());
        CHECK_FALSE(pre.model().num_constraints());
        CHECK(pre.techniques == presolve::TechniqueFlags::Default);
        CHECK(pre.feasibility() == presolve::Feasibility::Unknown);
    }

    SECTION("Empty model") {
        auto pre = PresolverImpl(ConstrainedQuadraticModel());
        CHECK_FALSE(pre.model().num_variables());
        CHECK_FALSE(pre.model().num_constraints());
        CHECK(pre.techniques == presolve::TechniqueFlags::Default);
        CHECK(pre.feasibility() == presolve::Feasibility::Unknown);
    }
}

TEST_CASE("Test apply()", "[presolve][impl]") {
    SECTION("A CQM with a single self-loop") {
        auto cqm = ConstrainedQuadraticModel();
        auto i = cqm.add_variable(dimod::Vartype::INTEGER);
        auto c = cqm.add_linear_constraint({}, {}, dimod::Sense::LE, 0);
        cqm.constraint_ref(c).set_quadratic(i, i, 1);

        WHEN("We call apply()") {
            auto pre = PresolverImpl(cqm);
            pre.techniques = presolve::TechniqueFlags::DomainPropagation;
            pre.apply();

            THEN("The model is not infeasible") {
                CHECK(pre.feasibility() != presolve::Feasibility::Infeasible);
                CHECK(pre.model().num_variables() == 2.0);
                CHECK(pre.model().num_constraints() == 2.0);
            }
        }
    }
}

TEST_CASE("Test normalization_fix_bounds", "[presolve][impl]") {
    GIVEN("A CQM with valid bounds") {
        auto cqm = ConstrainedQuadraticModel();
        cqm.add_variable(dimod::Vartype::REAL, -5, +5);
        cqm.add_variable(dimod::Vartype::INTEGER, -104, 123);
        cqm.add_variable(dimod::Vartype::BINARY);

        THEN("normalization_fix_bounds() does nothing") {
            auto pre = PresolverImpl(cqm);
            CHECK(!pre.normalization_fix_bounds());
        }
    }

    GIVEN("A CQM with invalid bounds") {
        auto cqm = ConstrainedQuadraticModel();
        auto v = cqm.add_variable(dimod::Vartype::REAL, -5, +5);
        cqm.set_lower_bound(v, 10);

        THEN("normalization_fix_bounds() throws an error") {
            auto pre = PresolverImpl(cqm);
            CHECK_THROWS_AS(pre.normalization_fix_bounds(), presolve::InvalidModelError);
        }
    }

    GIVEN("A CQM with fractional integer bounds") {
        auto cqm = ConstrainedQuadraticModel();
        auto v = cqm.add_variable(dimod::Vartype::INTEGER, -5.5, 7.4);

        WHEN("We give it to the presolver and run normalization_fix_bounds()") {
            auto pre = PresolverImpl(cqm);
            CHECK(pre.normalization_fix_bounds());

            THEN("the bounds are adjusted") {
                CHECK(pre.model().lower_bound(v) == -5.0);
                CHECK(pre.model().upper_bound(v) == +7.0);
            }
        }
    }

    GIVEN("A CQM with invalid fractional integer bounds") {
        auto cqm = ConstrainedQuadraticModel();
        cqm.add_variable(dimod::Vartype::INTEGER, 5.1, 5.9);

        THEN("normalization_fix_bounds() throws an error") {
            auto pre = PresolverImpl(cqm);
            CHECK_THROWS_AS(pre.normalization_fix_bounds(), presolve::InvalidModelError);
        }
    }

    GIVEN("A CQM with binary variables with loose bounds") {
        auto cqm = ConstrainedQuadraticModel();
        auto v = cqm.add_variable(dimod::Vartype::BINARY);
        cqm.set_lower_bound(v, -5);
        cqm.set_upper_bound(v, +5);

        WHEN("We give it to the presolver and run normalization_fix_bounds()") {
            auto pre = PresolverImpl(cqm);
            CHECK(pre.normalization_fix_bounds());

            THEN("the bounds are tightened") {
                CHECK(pre.model().lower_bound(v) == 0.0);
                CHECK(pre.model().upper_bound(v) == 1.0);
            }
        }
    }

    GIVEN("A CQM with binary variables with invalid bounds") {
        auto cqm = ConstrainedQuadraticModel();
        auto v = cqm.add_variable(dimod::Vartype::BINARY);
        cqm.set_lower_bound(v, -5);
        cqm.set_upper_bound(v, -7);

        THEN("normalization_fix_bounds() throws an error") {
            auto pre = PresolverImpl(cqm);
            CHECK_THROWS_AS(pre.normalization_fix_bounds(), presolve::InvalidModelError);
        }
    }
}

TEST_CASE("Test normalization_check_nan", "[presolve][impl]") {
    const double nan = std::numeric_limits<double>::quiet_NaN();

    SECTION("Linear objective with NaNs") {
        auto cqm = ConstrainedQuadraticModel();
        cqm.add_variable(dimod::BINARY);
        cqm.objective.set_linear(0, nan);
        CHECK_THROWS_AS(PresolverImpl::normalization_check_nan(cqm.objective),
                        presolve::InvalidModelError);
    }

    SECTION("Linear constraint with NaNs") {
        auto cqm = ConstrainedQuadraticModel();
        cqm.add_variable(dimod::BINARY);
        cqm.add_linear_constraint({0}, {nan}, dimod::Sense::EQ, 0);
        CHECK_THROWS_AS(PresolverImpl::normalization_check_nan(cqm.constraint_ref(0)),
                        presolve::InvalidModelError);
    }

    SECTION("Quadratic constraint with NaNs") {
        auto cqm = ConstrainedQuadraticModel();
        cqm.add_variables(dimod::BINARY, 2);
        auto c = cqm.add_linear_constraint({0, 1}, {1, 1}, dimod::Sense::EQ, 0);
        cqm.constraint_ref(c).add_quadratic(0, 1, nan);
        CHECK_THROWS_AS(PresolverImpl::normalization_check_nan(cqm.constraint_ref(0)),
                        presolve::InvalidModelError);
    }

    SECTION("Constraint with NaN rhs") {
        auto cqm = ConstrainedQuadraticModel();
        cqm.add_variable(dimod::BINARY);
        cqm.add_linear_constraint({0}, {1}, dimod::Sense::EQ, nan);
        CHECK_THROWS_AS(PresolverImpl::normalization_check_nan(cqm.constraint_ref(0)),
                        presolve::InvalidModelError);
    }

    SECTION("Constraint with NaN weight") {
        auto cqm = ConstrainedQuadraticModel();
        cqm.add_variable(dimod::BINARY);
        auto c = cqm.add_linear_constraint({0}, {1}, dimod::Sense::EQ, 0);
        cqm.constraint_ref(c).set_weight(nan);
        CHECK_THROWS_AS(PresolverImpl::normalization_check_nan(cqm.constraint_ref(0)),
                        presolve::InvalidModelError);
    }

    SECTION("Variables with NaN bounds") {
        auto cqm = ConstrainedQuadraticModel();
        auto v = cqm.add_variable(dimod::INTEGER);
        cqm.set_lower_bound(v, nan);
        auto pre = PresolverImpl(cqm);
        CHECK_THROWS_AS(pre.normalization_check_nan(), presolve::InvalidModelError);
    }
}

TEST_CASE("Test normalization_flip_constraints", "[presolve][impl]") {
    GIVEN("A CQM with three constraints of different senses") {
        auto cqm = ConstrainedQuadraticModel();
        auto i = cqm.add_variable(dimod::Vartype::INTEGER);
        auto j = cqm.add_variable(dimod::Vartype::INTEGER, -5, 5);
        auto k = cqm.add_variable(dimod::Vartype::INTEGER, -10, 10);

        auto c1 = cqm.add_linear_constraint({i, j}, {1, 2}, dimod::Sense::EQ, 3);
        auto c2 = cqm.add_linear_constraint({j, k}, {4, 5}, dimod::Sense::LE, 6);
        auto c3 = cqm.add_linear_constraint({i, k}, {7, 8}, dimod::Sense::GE, 9);

        cqm.constraint_ref(c1).add_quadratic(i, j, 10);
        cqm.constraint_ref(c2).add_quadratic(j, k, 11);
        cqm.constraint_ref(c3).add_quadratic(i, k, 12);

        cqm.constraint_ref(c1).add_offset(13);
        cqm.constraint_ref(c2).add_offset(14);
        cqm.constraint_ref(c3).add_offset(15);

        WHEN("We give it to the presolver and run normalization_remove_self_loops()") {
            auto pre = PresolverImpl(cqm);
            CHECK(pre.normalization_flip_constraints());

            THEN("The EQ and LE constraints are not changed") {
                for (auto ci : {c1, c2}) {
                    auto oldconstraint = cqm.constraint_ref(ci);
                    auto newconstraint = pre.model().constraint_ref(ci);

                    CHECK(oldconstraint.num_variables() == newconstraint.num_variables());
                    CHECK(oldconstraint.num_interactions() == newconstraint.num_interactions());
                    for (std::size_t v = 0; v < cqm.num_variables(); ++v) {
                        CHECK(oldconstraint.linear(v) == newconstraint.linear(v));
                    }
                    for (std::size_t u = 0; u < cqm.num_variables(); ++u) {
                        for (std::size_t v = 0; v < cqm.num_variables(); ++v) {
                            CHECK(oldconstraint.quadratic(u, v) == newconstraint.quadratic(u, v));
                        }
                    }
                    CHECK(oldconstraint.offset() == newconstraint.offset());
                    CHECK(oldconstraint.sense() == newconstraint.sense());
                    CHECK(oldconstraint.rhs() == newconstraint.rhs());
                }
            }

            THEN("The GE constraint is flipped") {
                auto oldconstraint = cqm.constraint_ref(c3);
                auto newconstraint = pre.model().constraint_ref(c3);

                CHECK(oldconstraint.num_variables() == newconstraint.num_variables());
                CHECK(oldconstraint.num_interactions() == newconstraint.num_interactions());
                for (std::size_t v = 0; v < cqm.num_variables(); ++v) {
                    CHECK(oldconstraint.linear(v) == -newconstraint.linear(v));
                }
                for (std::size_t u = 0; u < cqm.num_variables(); ++u) {
                    for (std::size_t v = 0; v < cqm.num_variables(); ++v) {
                        CHECK(oldconstraint.quadratic(u, v) == -newconstraint.quadratic(u, v));
                    }
                }
                CHECK(oldconstraint.offset() == -newconstraint.offset());
                CHECK(dimod::Sense::LE == newconstraint.sense());
                CHECK(oldconstraint.rhs() == -newconstraint.rhs());
            }
        }
    }
}

TEST_CASE("Test normalization_remove_invalid_markers", "[presolve][impl]") {
    GIVEN("A CQM with overlapping discrete constraints") {
        auto cqm = ConstrainedQuadraticModel();
        cqm.add_variables(dimod::Vartype::BINARY, 8);
        auto c0 = cqm.add_linear_constraint({0, 1, 2}, {1, 1, 1}, dimod::Sense::EQ, 1);  // overlap
        auto c1 = cqm.add_linear_constraint({2, 3, 4}, {1, 1, 1}, dimod::Sense::EQ, 1);  // overlap
        auto c2 = cqm.add_linear_constraint({5, 6, 7}, {1, 1, 1}, dimod::Sense::EQ, 1);  // not

        for (auto ci : {c0, c1, c2}) {
            cqm.constraint_ref(ci).mark_discrete();
        }

        WHEN("We give it to the presolver and run normalization_remove_invalid_markers()") {
            auto pre = PresolverImpl(cqm);
            CHECK(pre.normalization_remove_invalid_markers());

            THEN("Only the valid discrete constraints are still marked") {
                CHECK(pre.model().constraint_ref(c0).marked_discrete() ^
                      pre.model().constraint_ref(c1).marked_discrete());
                CHECK(pre.model().constraint_ref(c2).marked_discrete());
            }
        }
    }

    GIVEN("A CQM with a mis-marked constraint") {
        auto cqm = ConstrainedQuadraticModel();
        cqm.add_variables(dimod::Vartype::BINARY, 9);  // need the constraints to not overlap
        auto c0 = cqm.add_linear_constraint({0, 1, 2}, {1, 1, 1}, dimod::Sense::EQ, 1);  // valid
        auto c1 = cqm.add_linear_constraint({3, 4, 5}, {1, 2, 1}, dimod::Sense::EQ, 1);  // invalid
        auto c2 = cqm.add_linear_constraint({6, 7, 8}, {2, 2, 2}, dimod::Sense::EQ, 2);  // valid
        // NB: there are other cases we could test, but ultimately we'd just be testing
        // constraint.is_onehot() so let's rely on dimod to test that method thoroughly.

        for (auto ci : {c0, c1, c2}) {
            cqm.constraint_ref(ci).mark_discrete();
        }

        WHEN("We give it to the presolver and run normalization_remove_invalid_markers()") {
            auto pre = PresolverImpl(cqm);
            CHECK(pre.normalization_remove_invalid_markers());

            THEN("Only the valid discrete constraints are still marked") {
                CHECK(pre.model().constraint_ref(c0).marked_discrete());
                CHECK(!pre.model().constraint_ref(c1).marked_discrete());
                CHECK(pre.model().constraint_ref(c2).marked_discrete());
            }
        }
    }
}

TEST_CASE("Test normalization_remove_self_loops", "[presolve][impl]") {
    GIVEN("A CQM with a single integer self-loop") {
        auto cqm = ConstrainedQuadraticModel();
        auto v = cqm.add_variable(dimod::Vartype::INTEGER, -5, +5);
        cqm.objective.add_linear(v, 2);
        cqm.objective.add_quadratic(v, v, -7);
        cqm.objective.add_offset(3);

        WHEN("We give it to the presolver and run normalization_remove_self_loops()") {
            auto pre = PresolverImpl(cqm);
            CHECK(pre.normalization_remove_self_loops());

            THEN("The single integer variable is now two") {
                REQUIRE(pre.model().num_variables() == 2);
                REQUIRE(pre.model().num_constraints() == 1);

                CHECK(pre.model().objective.linear(0) == 2.);
                CHECK(pre.model().objective.linear(1) == 0.);
                CHECK(pre.model().objective.quadratic(0, 0) == 0.);
                CHECK(pre.model().objective.quadratic(1, 1) == 0.);
                CHECK(pre.model().objective.quadratic(0, 1) == -7.);
                CHECK(pre.model().objective.offset() == 3.);
                CHECK(pre.model().vartype(v) == dimod::Vartype::INTEGER);
                CHECK(pre.model().lower_bound(v) == -5);
                CHECK(pre.model().upper_bound(v) == +5);

                CHECK(pre.model().constraint_ref(0).linear(0) +
                              pre.model().constraint_ref(0).linear(1) ==
                      pre.model().constraint_ref(0).rhs());
                CHECK(pre.model().constraint_ref(0).linear(0));
            }
        }
    }

    GIVEN("A CQM with several self-loops") {
        auto cqm = ConstrainedQuadraticModel();
        auto i = cqm.add_variable(dimod::Vartype::INTEGER);
        auto j = cqm.add_variable(dimod::Vartype::INTEGER, -5, 5);
        auto k = cqm.add_variable(dimod::Vartype::INTEGER, -10, 10);

        auto c0 = cqm.add_constraint();
        cqm.constraint_ref(c0).add_quadratic(i, i, 2);
        cqm.constraint_ref(c0).add_quadratic(j, j, 3);

        auto c1 = cqm.add_constraint();
        cqm.constraint_ref(c1).add_quadratic(j, j, 4);
        cqm.constraint_ref(c1).add_quadratic(k, k, 5);

        WHEN("We give it to the presolver and run normalization_remove_self_loops()") {
            auto pre = PresolverImpl(cqm);
            CHECK(pre.normalization_remove_self_loops());

            THEN("Three new variables and two constraints are added") {
                REQUIRE(pre.model().num_variables() == 6);
                REQUIRE(pre.model().num_constraints() == 5);
            }
        }
    }
}

TEST_CASE("Test normalization_replace_inf", "[presolve][impl]") {
    const double inf = std::numeric_limits<double>::infinity();

    SECTION("Linear objective with inf") {
        auto cqm = ConstrainedQuadraticModel();
        auto v = cqm.add_variable(dimod::BINARY);
        cqm.objective.set_linear(v, inf);

        CHECK(PresolverImpl::normalization_replace_inf(cqm.objective));
        CHECK(cqm.objective.linear(v) == PresolverImpl::INF);
    }

    SECTION("Linear objective with -inf") {
        auto cqm = ConstrainedQuadraticModel();
        auto v = cqm.add_variable(dimod::BINARY);
        cqm.objective.set_linear(v, -inf);

        CHECK(PresolverImpl::normalization_replace_inf(cqm.objective));
        CHECK(cqm.objective.linear(v) == -PresolverImpl::INF);
    }

    SECTION("Linear objective with -inf offset") {
        auto cqm = ConstrainedQuadraticModel();
        cqm.add_variable(dimod::BINARY);
        cqm.objective.set_offset(-inf);

        CHECK(PresolverImpl::normalization_replace_inf(cqm.objective));
        CHECK(cqm.objective.offset() == -PresolverImpl::INF);
    }

    SECTION("Quadratic objective with -inf") {
        auto cqm = ConstrainedQuadraticModel();
        auto u = cqm.add_variable(dimod::BINARY);
        auto v = cqm.add_variable(dimod::BINARY);
        cqm.objective.set_quadratic(u, v, -inf);

        CHECK(PresolverImpl::normalization_replace_inf(cqm.objective));
        CHECK(cqm.objective.quadratic(u, v) == -PresolverImpl::INF);
    }

    SECTION("Constraint with inf rhs") {
        auto cqm = ConstrainedQuadraticModel();
        cqm.add_variable(dimod::BINARY);
        cqm.add_linear_constraint({0}, {1}, dimod::Sense::EQ, inf);

        CHECK(PresolverImpl::normalization_replace_inf(cqm.constraint_ref(0)));
        CHECK(cqm.constraint_ref(0).rhs() == PresolverImpl::INF);
    }

    SECTION("Constraint with -inf rhs") {
        auto cqm = ConstrainedQuadraticModel();
        cqm.add_variable(dimod::BINARY);
        cqm.add_linear_constraint({0}, {1}, dimod::Sense::EQ, -inf);

        CHECK(PresolverImpl::normalization_replace_inf(cqm.constraint_ref(0)));
        CHECK(cqm.constraint_ref(0).rhs() == -PresolverImpl::INF);
    }

    SECTION("inf lower bound") {
        auto cqm = ConstrainedQuadraticModel();
        auto v = cqm.add_variable(dimod::INTEGER);
        cqm.set_lower_bound(v, -inf);
        cqm.set_upper_bound(v, +inf);

        auto pre = PresolverImpl(cqm);
        CHECK(pre.normalize());

        CHECK(pre.model().lower_bound(v) == -PresolverImpl::INF);
        CHECK(pre.model().upper_bound(v) == +PresolverImpl::INF);

        CHECK(pre.feasibility() != presolve::Feasibility::Infeasible);
    }
}

TEST_CASE("Test normalization_spin_to_binary", "[presolve][impl]") {
    SECTION("Empty model") {
        auto pre = PresolverImpl(ConstrainedQuadraticModel());
        CHECK(!pre.normalization_spin_to_binary());
        CHECK_FALSE(pre.model().num_variables());
        CHECK_FALSE(pre.model().num_constraints());
        CHECK(pre.techniques == presolve::TechniqueFlags::Default);
        CHECK(pre.feasibility() == presolve::Feasibility::Unknown);
    }

    GIVEN("A CQM with a single spin variable") {
        auto cqm = ConstrainedQuadraticModel();
        auto v = cqm.add_variable(dimod::Vartype::SPIN);
        cqm.objective.set_linear(v, 1.5);
        cqm.objective.set_offset(-10);
        auto c = cqm.add_linear_constraint({v}, {-2}, dimod::Sense::EQ, 3);

        WHEN("We give it to the presolver and run normalization_spin_to_binary()") {
            auto pre = PresolverImpl(cqm);
            CHECK(pre.normalization_spin_to_binary());

            THEN("The model will have one binary variable and the energies will match") {
                REQUIRE(pre.model().num_variables() == 1);
                REQUIRE(pre.model().num_constraints() == 1);

                CHECK(pre.model().vartype(v) == dimod::Vartype::BINARY);

                auto bin_sample = std::vector<int>{0};
                auto spin_sample = std::vector<int>{-1};

                CHECK(pre.model().objective.energy(bin_sample.begin()) ==
                      cqm.objective.energy(spin_sample.begin()));
                CHECK(pre.model().constraint_ref(c).energy(bin_sample.begin()) ==
                      cqm.constraint_ref(c).energy(spin_sample.begin()));
            }
        }
    }
}

TEST_CASE("Test technique_clear_redundant_constraint", "[presolve][impl]") {
    GIVEN("A CQM with linear constraint that cannot be satisfied") {
        auto cqm = ConstrainedQuadraticModel();
        cqm.add_variables(dimod::Vartype::INTEGER, 3, 5, 100);
        auto c = cqm.add_linear_constraint({0, 1, 2}, {1, 1, 1}, dimod::Sense::LE, 0);

        WHEN("We remove redundant constraints") {
            auto pre = PresolverImpl(cqm);
            pre.techniques = presolve::TechniqueFlags::RemoveRedundantConstraints;
            CHECK(!pre.normalize());
            CHECK(!pre.presolve());

            THEN("the model is infeasible and the constraint is not removed") {
                CHECK(pre.feasibility() == presolve::Feasibility::Infeasible);

                REQUIRE(pre.model().num_constraints() == 1);
                REQUIRE(pre.model().constraint_ref(c).num_variables() == 3);
                CHECK(pre.model().constraint_ref(c).linear(0) == 1.0);
                CHECK(pre.model().constraint_ref(c).linear(0) == 1.0);
                CHECK(pre.model().constraint_ref(c).linear(0) == 1.0);
                CHECK(pre.model().constraint_ref(c).sense() == dimod::Sense::LE);
                CHECK(pre.model().constraint_ref(c).rhs() == 0.0);
            }
        }

        WHEN("We mark that constraint is soft") {
            cqm.constraint_ref(c).set_weight(5);

            AND_WHEN("We remove redundant constraints") {
                auto pre = PresolverImpl(cqm);
                pre.techniques = presolve::TechniqueFlags::RemoveRedundantConstraints;
                CHECK(!pre.normalize());
                CHECK(!pre.presolve());

                THEN("the model is not infeasible and the soft constraint is still present") {
                    CHECK(pre.feasibility() != presolve::Feasibility::Infeasible);

                    REQUIRE(pre.model().num_constraints() == 1);
                    REQUIRE(pre.model().constraint_ref(c).num_variables() == 3);
                    CHECK(pre.model().constraint_ref(c).linear(0) == 1.0);
                    CHECK(pre.model().constraint_ref(c).linear(0) == 1.0);
                    CHECK(pre.model().constraint_ref(c).linear(0) == 1.0);
                    CHECK(pre.model().constraint_ref(c).sense() == dimod::Sense::LE);
                    CHECK(pre.model().constraint_ref(c).rhs() == 0.0);
                }
            }
        }
    }
}

TEST_CASE("Test technique_domain_propagation", "[presolve][impl]") {
    GIVEN("An empty constraint") {
        auto cqm = ConstrainedQuadraticModel();
        cqm.add_constraint();

        THEN("Domain propagation does nothing") {
            auto pre = PresolverImpl(cqm);
            CHECK(!pre.technique_domain_propagation(pre.model().constraint_ref(0)));
        }
    }

    GIVEN("A CQM with a single REAL variable in [-100, 200]") {
        auto cqm = ConstrainedQuadraticModel();
        auto v = cqm.add_variable(dimod::Vartype::REAL, -100, +200);

        AND_GIVEN("A v <= -105 constraint that makes the model infeasible") {
            auto c = cqm.add_linear_constraint({v}, {1}, dimod::Sense::LE, -105);

            WHEN("We apply domain propagation") {
                auto pre = PresolverImpl(cqm);
                CHECK(!pre.technique_domain_propagation(pre.model().constraint_ref(c)));

                THEN("The bounds aren't changed but the model is marked infeasible") {
                    CHECK(pre.model().lower_bound(v) == -100.0);
                    CHECK(pre.model().upper_bound(v) == +200.0);
                    CHECK(pre.feasibility() == presolve::Feasibility::Infeasible);
                }
            }
        }

        AND_GIVEN("A v >= 201 constraint that makes the model infeasible") {
            auto c = cqm.add_linear_constraint({v}, {1}, dimod::Sense::GE, 201);

            WHEN("We apply domain propagation") {
                auto pre = PresolverImpl(cqm);
                CHECK(!pre.technique_domain_propagation(pre.model().constraint_ref(c)));

                THEN("The bounds aren't changed but the model is marked infeasible") {
                    CHECK(pre.model().lower_bound(v) == -100.0);
                    CHECK(pre.model().upper_bound(v) == +200.0);
                    CHECK(pre.feasibility() == presolve::Feasibility::Infeasible);
                }
            }
        }

        AND_GIVEN("A v == 201 constraint that makes the model infeasible") {
            auto c = cqm.add_linear_constraint({v}, {1}, dimod::Sense::EQ, 201);

            WHEN("We apply domain propagation") {
                auto pre = PresolverImpl(cqm);
                CHECK(!pre.technique_domain_propagation(pre.model().constraint_ref(c)));

                THEN("The bounds aren't changed but the model is marked infeasible") {
                    CHECK(pre.model().lower_bound(v) == -100.0);
                    CHECK(pre.model().upper_bound(v) == +200.0);
                    CHECK(pre.feasibility() == presolve::Feasibility::Infeasible);
                }
            }
        }

        AND_GIVEN("A v <= 3 constraint") {
            auto c = cqm.add_linear_constraint({v}, {1}, dimod::Sense::LE, 3);

            WHEN("We apply domain propagation") {
                auto pre = PresolverImpl(cqm);
                CHECK(pre.technique_domain_propagation(pre.model().constraint_ref(c)));

                THEN("The upper bound is tightened") {
                    CHECK(pre.model().lower_bound(v) == -100.0);
                    CHECK(pre.model().upper_bound(v) == 3.0);
                    CHECK(pre.feasibility() == presolve::Feasibility::Unknown);
                }
            }
        }

        AND_GIVEN("A v <= 203 constraint") {
            auto c = cqm.add_linear_constraint({v}, {1}, dimod::Sense::LE, 203);

            WHEN("We apply domain propagation") {
                auto pre = PresolverImpl(cqm);
                CHECK(!pre.technique_domain_propagation(pre.model().constraint_ref(c)));

                THEN("The bounds don't change") {
                    CHECK(pre.model().lower_bound(v) == -100.0);
                    CHECK(pre.model().upper_bound(v) == +200.0);
                    CHECK(pre.feasibility() == presolve::Feasibility::Unknown);
                }
            }
        }

        AND_GIVEN("A 6*v >= 30 constraint") {
            auto c = cqm.add_linear_constraint({v}, {6}, dimod::Sense::GE, 30);

            // because of normalization we should never need to handle this case,
            // but it's a nice test case so no harm
            WHEN("We apply domain propagation") {
                auto pre = PresolverImpl(cqm);
                CHECK(pre.technique_domain_propagation(pre.model().constraint_ref(c)));

                THEN("The lower bound is tightened") {
                    CHECK(pre.model().lower_bound(v) == 5.0);
                    CHECK(pre.model().upper_bound(v) == 200.0);
                    CHECK(pre.feasibility() == presolve::Feasibility::Unknown);
                }
            }

            WHEN("We normalize and then apply domain propagation") {
                auto pre = PresolverImpl(cqm);
                CHECK(pre.normalize());
                CHECK(pre.technique_domain_propagation(pre.model().constraint_ref(c)));

                THEN("The lower bound is tightened") {
                    CHECK(pre.model().lower_bound(v) == 5.0);
                    CHECK(pre.model().upper_bound(v) == 200.0);
                    CHECK(pre.feasibility() == presolve::Feasibility::Unknown);
                }
            }
        }

        AND_GIVEN("A v >= -105 constraint") {
            auto c = cqm.add_linear_constraint({v}, {1}, dimod::Sense::GE, -105);

            // because of normalization we should never need to handle this case,
            // but it's a nice test case so no harm
            WHEN("We apply domain propagation") {
                auto pre = PresolverImpl(cqm);
                CHECK(!pre.technique_domain_propagation(pre.model().constraint_ref(c)));

                THEN("The bounds don't change") {
                    CHECK(pre.model().lower_bound(v) == -100);
                    CHECK(pre.model().upper_bound(v) == 200.0);
                    CHECK(pre.feasibility() == presolve::Feasibility::Unknown);
                }
            }

            WHEN("We normalize and then apply domain propagation") {
                auto pre = PresolverImpl(cqm);
                CHECK(pre.normalize());
                CHECK(!pre.technique_domain_propagation(pre.model().constraint_ref(c)));

                THEN("The bounds don't change") {
                    CHECK(pre.model().lower_bound(v) == -100);
                    CHECK(pre.model().upper_bound(v) == 200.0);
                    CHECK(pre.feasibility() == presolve::Feasibility::Unknown);
                }
            }
        }

        AND_GIVEN("A 6*v == 30 constraint") {
            auto c = cqm.add_linear_constraint({v}, {6}, dimod::Sense::EQ, 30);

            WHEN("We apply domain propagation") {
                auto pre = PresolverImpl(cqm);
                CHECK(pre.technique_domain_propagation(pre.model().constraint_ref(c)));

                THEN("The upper and lower bound are updated") {
                    CHECK(pre.model().lower_bound(v) == 5.0);
                    CHECK(pre.model().upper_bound(v) == 5.0);
                    CHECK(pre.feasibility() == presolve::Feasibility::Unknown);
                }
            }
        }

        AND_GIVEN("A -6*v == 30 constraint") {
            auto c = cqm.add_linear_constraint({v}, {-6}, dimod::Sense::EQ, 30);

            WHEN("We apply domain propagation") {
                auto pre = PresolverImpl(cqm);
                CHECK(pre.technique_domain_propagation(pre.model().constraint_ref(c)));

                THEN("The upper and lower bound are updated") {
                    CHECK(pre.model().lower_bound(v) == -5.0);
                    CHECK(pre.model().upper_bound(v) == -5.0);
                    CHECK(pre.feasibility() == presolve::Feasibility::Unknown);
                }
            }
        }
    }

    GIVEN("A CQM with a binary variable x and a continuous variable v") {
        auto cqm = ConstrainedQuadraticModel();
        auto x = cqm.add_variable(dimod::Vartype::BINARY);
        auto v = cqm.add_variable(dimod::Vartype::REAL, -50, +50);

        AND_GIVEN("A 2x + 4v <= 20 constraint") {
            auto c = cqm.add_linear_constraint({x, v}, {2, 4}, dimod::Sense::LE, 20);

            WHEN("We apply domain propagation") {
                auto pre = PresolverImpl(cqm);
                CHECK(pre.technique_domain_propagation(pre.model().constraint_ref(c)));

                THEN("The upper bound of v is updated") {
                    CHECK(pre.model().lower_bound(x) == 0.0);
                    CHECK(pre.model().upper_bound(x) == 1.0);
                    CHECK(pre.model().lower_bound(v) == -50.0);
                    CHECK(pre.model().upper_bound(v) == 5.0);
                    CHECK(pre.feasibility() == presolve::Feasibility::Unknown);
                }
            }
        }

        AND_GIVEN("A -2x + 4v <= 20 constraint") {
            auto c = cqm.add_linear_constraint({x, v}, {-2, 4}, dimod::Sense::LE, 20);

            WHEN("We apply domain propagation") {
                auto pre = PresolverImpl(cqm);
                CHECK(pre.technique_domain_propagation(pre.model().constraint_ref(c)));

                THEN("The upper bound of v is updated") {
                    CHECK(pre.model().lower_bound(x) == 0.0);
                    CHECK(pre.model().upper_bound(x) == 1.0);
                    CHECK(pre.model().lower_bound(v) == -50.0);
                    CHECK(pre.model().upper_bound(v) == 5.5);
                    CHECK(pre.feasibility() == presolve::Feasibility::Unknown);
                }
            }
        }
    }

    GIVEN("A CQM with a binary variable x and an integer variable v") {
        auto cqm = ConstrainedQuadraticModel();
        auto x = cqm.add_variable(dimod::Vartype::BINARY);
        auto v = cqm.add_variable(dimod::Vartype::INTEGER, -50, +50);

        AND_GIVEN("A 2x + 4v <= 20 constraint") {
            auto c = cqm.add_linear_constraint({x, v}, {2, 4}, dimod::Sense::LE, 20);

            WHEN("We apply domain propagation") {
                auto pre = PresolverImpl(cqm);
                CHECK(pre.technique_domain_propagation(pre.model().constraint_ref(c)));

                THEN("The upper bound of v is updated") {
                    CHECK(pre.model().lower_bound(x) == 0.0);
                    CHECK(pre.model().upper_bound(x) == 1.0);
                    CHECK(pre.model().lower_bound(v) == -50.0);
                    CHECK(pre.model().upper_bound(v) == 5.0);
                    CHECK(pre.feasibility() == presolve::Feasibility::Unknown);
                }
            }
        }

        AND_GIVEN("A -2x + 4v <= 20 constraint") {
            auto c = cqm.add_linear_constraint({x, v}, {-2, 4}, dimod::Sense::LE, 20);

            WHEN("We apply domain propagation") {
                auto pre = PresolverImpl(cqm);
                CHECK(pre.technique_domain_propagation(pre.model().constraint_ref(c)));

                THEN("The upper bound of v is updated") {
                    CHECK(pre.model().lower_bound(x) == 0.0);
                    CHECK(pre.model().upper_bound(x) == 1.0);
                    CHECK(pre.model().lower_bound(v) == -50.0);
                    CHECK(pre.model().upper_bound(v) == 5.0);  // floor(5.5)
                    CHECK(pre.feasibility() == presolve::Feasibility::Unknown);
                }
            }
        }
    }

    GIVEN("A CQM with an equality constraint over two binary variables") {
        auto cqm = ConstrainedQuadraticModel();
        auto x = cqm.add_variable(dimod::Vartype::BINARY);
        auto y = cqm.add_variable(dimod::Vartype::BINARY);
        auto c = cqm.add_linear_constraint({x, y}, {1, -1}, dimod::Sense::EQ, 0);

        WHEN("We apply domain propagation") {
            auto pre = PresolverImpl(cqm);
            CHECK(!pre.technique_domain_propagation(pre.model().constraint_ref(c)));

            THEN("nothing happens") {
                CHECK(pre.model().num_variables() == 2.0);
                CHECK(pre.model().constraint_ref(c).linear(x) == 1.0);
                CHECK(pre.model().constraint_ref(c).linear(y) == -1.0);
                CHECK(pre.feasibility() != presolve::Feasibility::Infeasible);
            }
        }
    }

    GIVEN("A CQM with two cont. variables and two <= constraints that can strengthen eachother") {
        auto cqm = ConstrainedQuadraticModel();
        auto a = cqm.add_variable(dimod::Vartype::REAL);
        auto b = cqm.add_variable(dimod::Vartype::REAL);
        auto c0 = cqm.add_linear_constraint({a, b}, {1, -2}, dimod::Sense::LE, -100);
        auto c1 = cqm.add_linear_constraint({a, b}, {-2, 1}, dimod::Sense::LE, -100);

        auto pre = PresolverImpl(cqm);

        WHEN("we apply domain propagation to the a - 2b <= -100 constraint") {
            pre.technique_domain_propagation(pre.model().constraint_ref(c0));

            THEN("It uses it to strengthen the bound on b") {
                CHECK(pre.model().lower_bound(a) == 0.0);  // unchanged
                CHECK(pre.model().lower_bound(b) == 50.0);

                AND_WHEN("we then apply domain propagation to the b - 2a <= -100 constraint") {
                    pre.technique_domain_propagation(pre.model().constraint_ref(c1));

                    THEN("It strengthens the bound on a") {
                        CHECK(pre.model().lower_bound(a) == 75.0);
                        CHECK(pre.model().lower_bound(b) == 50.0);  // unchanged
                    }
                }
            }
        }
    }

    GIVEN("A CQM with two cont. variables and two >= constraints that can strengthen eachother") {
        auto cqm = ConstrainedQuadraticModel();
        auto a = cqm.add_variable(dimod::Vartype::REAL);
        auto b = cqm.add_variable(dimod::Vartype::REAL);

        // NB: this is actually infeasible
        auto c0 = cqm.add_linear_constraint({a, b}, {1, -2}, dimod::Sense::GE, 100);
        auto c1 = cqm.add_linear_constraint({a, b}, {-2, 1}, dimod::Sense::GE, 100);

        auto pre = PresolverImpl(cqm);

        WHEN("we apply domain propagation to the a - 2b >= 100 constraint") {
            pre.technique_domain_propagation(pre.model().constraint_ref(c0));

            THEN("It uses it to strengthen the bound on a") {
                CHECK(pre.model().lower_bound(a) == 100.0);
                CHECK(pre.model().lower_bound(b) == 0.0);  // unchanged

                AND_WHEN("we then apply domain propagation to the second constraint") {
                    pre.technique_domain_propagation(pre.model().constraint_ref(c1));

                    THEN("It strengthens the bound on a") {
                        CHECK(pre.model().lower_bound(a) == 100.0);  // unchanged
                        CHECK(pre.model().lower_bound(b) == 300.0);
                    }
                }
            }
        }
    }
}

TEST_CASE("Test technique_remove_small_biases", "[presolve][impl]") {
    GIVEN("A linear CQM with small biases") {
        auto cqm = ConstrainedQuadraticModel();
        cqm.add_variables(dimod::Vartype::BINARY, 2);
        cqm.objective.set_linear(0, 1e-11);
        cqm.objective.set_linear(1, -4);
        cqm.add_linear_constraint({1, 0}, {1e-11, 4}, dimod::Sense::LE, 100);

        WHEN("We apply technique_remove_small_biases()") {
            CHECK(PresolverImpl::technique_remove_small_biases(cqm.objective));
            CHECK(PresolverImpl::technique_remove_small_biases(cqm.constraint_ref(0)));

            THEN("The variables with small linear biases are removed from the expressions") {
                CHECK(cqm.objective.variables() == std::vector{1});
                CHECK(cqm.constraint_ref(0).variables() == std::vector{0});
            }
        }

        WHEN("We add quadratic biases to the expressions") {
            cqm.objective.set_quadratic(0, 1, 1);
            cqm.constraint_ref(0).set_quadratic(0, 1, -1);

            AND_WHEN("We apply technique_remove_small_biases()") {
                CHECK(PresolverImpl::technique_remove_small_biases(cqm.objective));
                CHECK(PresolverImpl::technique_remove_small_biases(cqm.constraint_ref(0)));

                THEN("The variables with small linear biases are not removed from the expressions "
                     "but their linear biases are zeroed") {
                    CHECK(cqm.objective.variables() == std::vector{0, 1});
                    CHECK(cqm.constraint_ref(0).variables() == std::vector{1, 0});
                    CHECK(cqm.objective.linear(0) == 0.0);
                    CHECK(cqm.constraint_ref(0).linear(1) == 0.0);
                }
            }
        }

        WHEN("We add small quadratic biases to the expressions") {
            cqm.objective.set_quadratic(0, 1, 1e-11);
            cqm.constraint_ref(0).set_quadratic(0, 1, -1e-11);

            AND_WHEN("We apply technique_remove_small_biases()") {
                CHECK(PresolverImpl::technique_remove_small_biases(cqm.objective));
                CHECK(PresolverImpl::technique_remove_small_biases(cqm.constraint_ref(0)));

                THEN("The variables with small biases are removed from the expressions") {
                    CHECK(cqm.objective.variables() == std::vector{1});
                    CHECK(cqm.constraint_ref(0).variables() == std::vector{0});
                }
            }
        }

        WHEN("We run PresolverImpl::apply() without any techniques loaded") {
            auto pre = PresolverImpl(cqm);
            pre.techniques = presolve::TechniqueFlags::None;
            CHECK(!pre.apply());
        }

        WHEN("We run PresolverImpl::apply() with RemoveSmallBiases loaded") {
            auto pre = PresolverImpl(cqm);
            pre.techniques = presolve::TechniqueFlags::RemoveSmallBiases;
            CHECK(pre.apply());

            THEN("The variables with small linear biases are removed from the expressions") {
                CHECK(pre.model().objective.variables() == std::vector{1});
                CHECK(pre.model().constraint_ref(0).variables() == std::vector{0});
            }
        }
    }

    GIVEN("A linear CQM with biases that could be conditionally removed") {
        auto cqm = ConstrainedQuadraticModel();
        cqm.add_variables(dimod::Vartype::REAL, 3);

        cqm.set_lower_bound(0, -1e-5);
        cqm.set_upper_bound(0, 0);
        cqm.set_lower_bound(1, -1e-5);
        cqm.set_upper_bound(1, 0);

        // x0 should trigger the conditional removal, x1 should not, x2 should trigger the
        // first test, but not the second so shouldn't be removed
        cqm.add_linear_constraint({1, 0, 2}, {1, .0003, .0003}, dimod::Sense::LE, 100);

        // none should be removed from the objective
        cqm.objective.set_linear(0, .0003);
        cqm.objective.set_linear(2, .0003);
        cqm.objective.set_linear(1, 1);

        WHEN("We apply technique_remove_small_biases()") {
            CHECK(!PresolverImpl::technique_remove_small_biases(cqm.objective));
            CHECK(PresolverImpl::technique_remove_small_biases(cqm.constraint_ref(0)));

            THEN("The variables with small linear biases are removed from the constraint") {
                CHECK(cqm.constraint_ref(0).variables() == std::vector{1, 2});
                CHECK(cqm.constraint_ref(0).linear(1) == 1.0);
                CHECK(cqm.constraint_ref(0).linear(2) == .0003);
                CHECK(!cqm.constraint_ref(0).offset());
                CHECK(cqm.constraint_ref(0).rhs() == Approx(100 - 1e-5 * .0003));
            }

            THEN("The objective is not changed") {
                CHECK(cqm.objective.variables() == std::vector{0, 2, 1});
            }
        }

        WHEN("We add quadratic bias") {
            cqm.constraint_ref(0).add_quadratic(0, 1, 1);

            AND_WHEN("We apply technique_remove_small_biases()") {
                PresolverImpl::technique_remove_small_biases(cqm.objective);

                THEN("There is no conditional removal") {
                    CHECK(cqm.constraint_ref(0).variables() == std::vector{1, 0, 2});
                }
            }
        }
    }
}
}  // namespace dwave
