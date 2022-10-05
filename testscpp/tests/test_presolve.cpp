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
#include "dimod/constrained_quadratic_model.h"
#include "dimod/presolve.h"


namespace dimod {

SCENARIO("constrained quadratic models can be presolved") {
    GIVEN("a cqm with some trivial issues") {
        auto cqm = ConstrainedQuadraticModel<double>();

        // v0 has default bounds, but a constraint restricting it
        auto v0 = cqm.add_variable(Vartype::INTEGER);
        auto c0 = cqm.add_linear_constraint({v0}, {1}, Sense::LE, 5);  // v0 <= 5

        // v1 has bounds that implicitly fix it
        auto v1 = cqm.add_variable(Vartype::INTEGER, 5, 5);

        // v2 has default bounds, but an equality constraint that fixes it
        auto v2 = cqm.add_variable(Vartype::INTEGER);
        auto c2 = cqm.add_linear_constraint({v2}, {1}, Sense::EQ, 7);

        // v3 has default bounds, but two inequality constraints that fix it
        auto v3 = cqm.add_variable(Vartype::INTEGER);
        auto c3a = cqm.add_linear_constraint({v3}, {1}, Sense::LE, 5.5);
        auto c3b = cqm.add_linear_constraint({v3}, {1}, Sense::GE, 4.5);

        WHEN("passed into a presolver") {
            auto presolver = presolve::PreSolver<ConstrainedQuadraticModel<double>>(std::move(cqm));

            THEN("it has moved correctly") {
                const auto& newcqm = presolver.model();

                REQUIRE(newcqm.num_variables() == 4);
                REQUIRE(newcqm.num_constraints() == 4);

                CHECK(newcqm.vartype(v0) == Vartype::INTEGER);
                CHECK(newcqm.lower_bound(v0) == 0);
                CHECK(newcqm.upper_bound(v0) ==
                      vartype_limits<double, Vartype::INTEGER>::default_max());
                CHECK(newcqm.constraint_ref(c0).linear(v0) == 1);
                CHECK(newcqm.constraint_ref(c0).sense() == Sense::LE);
                CHECK(newcqm.constraint_ref(c0).rhs() == 5);

                CHECK(newcqm.vartype(v1) == Vartype::INTEGER);
                CHECK(newcqm.lower_bound(v1) == 5);
                CHECK(newcqm.upper_bound(v1) == 5);

                CHECK(newcqm.vartype(v2) == Vartype::INTEGER);
                CHECK(newcqm.lower_bound(v2) == 0);
                CHECK(newcqm.upper_bound(v2) ==
                      vartype_limits<double, Vartype::INTEGER>::default_max());
                CHECK(newcqm.constraint_ref(c2).linear(v2) == 1);
                CHECK(newcqm.constraint_ref(c2).sense() == Sense::EQ);
                CHECK(newcqm.constraint_ref(c2).rhs() == 7);

                CHECK(newcqm.vartype(v3) == Vartype::INTEGER);
                CHECK(newcqm.lower_bound(v3) == 0);
                CHECK(newcqm.upper_bound(v3) ==
                      vartype_limits<double, Vartype::INTEGER>::default_max());
                CHECK(newcqm.constraint_ref(c3a).linear(v3) == 1);
                CHECK(newcqm.constraint_ref(c3a).sense() == Sense::LE);
                CHECK(newcqm.constraint_ref(c3a).rhs() == 5.5);
                CHECK(newcqm.constraint_ref(c3b).linear(v3) == 1);
                CHECK(newcqm.constraint_ref(c3b).sense() == Sense::GE);
                CHECK(newcqm.constraint_ref(c3b).rhs() == 4.5);
            }

            AND_WHEN("the default presolving is applied") {
                presolver.load_default_presolvers();
                presolver.apply();

                THEN("most of the constraints/variables are removed") {
                    CHECK(presolver.model().num_constraints() == 0);
                    CHECK(presolver.model().num_variables() == 1);
                }

                AND_WHEN("we then undo the transformation") {
                    auto original = presolver.postsolver().apply(std::vector<int>{3});
                    CHECK(original == std::vector<int>{3, 5, 7, 5});
                }
            }
        }
    }

    GIVEN("a cqm with some spin variables") {
        auto cqm = ConstrainedQuadraticModel<double>();
        cqm.add_variables(Vartype::SPIN, 5);
        for (size_t v = 0; v < 5; ++v) {
            for (size_t u = v + 1; u < 5; ++u) {
                cqm.objective.set_quadratic(u, v, 1);
            }
        }

        WHEN("the default presolving is applied") {
            auto presolver = presolve::PreSolver<ConstrainedQuadraticModel<double>>(std::move(cqm));
            presolver.load_default_presolvers();
            presolver.apply();

            THEN("most of the constraints/variables are removed") {
                CHECK(presolver.model().num_constraints() == 0);
                CHECK(presolver.model().num_variables() == 5);

                for (size_t v = 0; v < 5; ++v) {
                    CHECK(presolver.model().vartype(v) == Vartype::BINARY);
                }

                AND_WHEN("we then undo the transformation") {
                    auto original = presolver.postsolver().apply(std::vector<int>{0, 1, 0, 1, 0});
                    CHECK(original == std::vector<int>{-1, +1, -1, +1, -1});
                }
            }
        }
    }

    //     WHEN("passed through the identity presolver") {
    //         auto presolver = presolve::PreSolver<double>(cqm);
    //         presolver.apply();

    //         THEN("nothing has changed") {
    //             // CHECK(presolver.result.status == unchanged)

    //             const auto& newcqm = presolver.model();

    //             REQUIRE(newcqm.num_variables() == 4);
    //             REQUIRE(newcqm.num_constraints() == 4);

    //             CHECK(newcqm.vartype(v0) == Vartype::INTEGER);
    //             CHECK(newcqm.lower_bound(v0) == 0);
    //             CHECK(newcqm.upper_bound(v0) ==
    //                   vartype_limits<double, Vartype::INTEGER>::default_max());
    //             CHECK(newcqm.constraint_ref(c0).linear(v0) == 1);
    //             CHECK(newcqm.constraint_ref(c0).sense() == Sense::LE);
    //             CHECK(newcqm.constraint_ref(c0).rhs() == 5);

    //             CHECK(newcqm.vartype(v1) == Vartype::INTEGER);
    //             CHECK(newcqm.lower_bound(v1) == 5);
    //             CHECK(newcqm.upper_bound(v1) == 5);

    //             CHECK(newcqm.vartype(v2) == Vartype::INTEGER);
    //             CHECK(newcqm.lower_bound(v2) == 0);
    //             CHECK(newcqm.upper_bound(v2) ==
    //                   vartype_limits<double, Vartype::INTEGER>::default_max());
    //             CHECK(newcqm.constraint_ref(c2).linear(v2) == 1);
    //             CHECK(newcqm.constraint_ref(c2).sense() == Sense::EQ);
    //             CHECK(newcqm.constraint_ref(c2).rhs() == 7);

    //             CHECK(newcqm.vartype(v3) == Vartype::INTEGER);
    //             CHECK(newcqm.lower_bound(v3) == 0);
    //             CHECK(newcqm.upper_bound(v3) ==
    //                   vartype_limits<double, Vartype::INTEGER>::default_max());
    //             CHECK(newcqm.constraint_ref(c3a).linear(v3) == 1);
    //             CHECK(newcqm.constraint_ref(c3a).sense() == Sense::LE);
    //             CHECK(newcqm.constraint_ref(c3a).rhs() == 5.5);
    //             CHECK(newcqm.constraint_ref(c3b).linear(v3) == 1);
    //             CHECK(newcqm.constraint_ref(c3b).sense() == Sense::GE);
    //             CHECK(newcqm.constraint_ref(c3b).rhs() == 4.5);
    //         }
    //     }

    //     WHEN("passed through the trivial presolver") {
    //         auto presolver = presolve::PreSolver<double>(cqm);
    //         presolver.add_presolver<presolve::techniques::TrivialPresolver<double>>();
    //         presolver.apply();

    //         THEN("several constraints/variables are removed") {
    //             const auto& newcqm = presolver.model();
    //             const auto& postsolver = presolver.postsolver();

    //             CHECK(newcqm.num_constraints() == 0);

    //             CHECK(newcqm.num_variables() == 1);

    //             // see if we restore the original problem
    //             auto reduced = std::vector<int>{3};

    //             auto original = postsolver.apply(reduced);

    //             CHECK(original == std::vector<int>{3, 5, 7, 5});
    //         }
    //     }
    // }
}


}  // namespace dimod
