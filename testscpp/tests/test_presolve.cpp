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

#include "../Catch2/single_include/catch2/catch.hpp"
#include "dimod/constrained.h"
#include "dimod/presolver.h"

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

        WHEN("passed through the identity presolver") {
            auto presolver = presolve::PreSolver<double>(cqm);
            presolver.apply();

            THEN("nothing has changed") {
                // CHECK(presolver.result.status == unchanged)

                const auto& newcqm = presolver.model();

                REQUIRE(newcqm.num_variables() == 4);
                REQUIRE(newcqm.num_constraints() == 4);

                CHECK(newcqm.vartype(v0) == Vartype::INTEGER);
                CHECK(newcqm.lower_bound(v0) == 0);
                CHECK(newcqm.upper_bound(v0) ==
                      vartype_limits<double, Vartype::INTEGER>::default_max());
                CHECK(newcqm.constraints[c0].linear(v0) == 1);
                CHECK(newcqm.constraints[c0].sense() == Sense::LE);
                CHECK(newcqm.constraints[c0].rhs() == 5);

                CHECK(newcqm.vartype(v1) == Vartype::INTEGER);
                CHECK(newcqm.lower_bound(v1) == 5);
                CHECK(newcqm.upper_bound(v1) == 5);

                CHECK(newcqm.vartype(v2) == Vartype::INTEGER);
                CHECK(newcqm.lower_bound(v2) == 0);
                CHECK(newcqm.upper_bound(v2) ==
                      vartype_limits<double, Vartype::INTEGER>::default_max());
                CHECK(newcqm.constraints[c2].linear(v2) == 1);
                CHECK(newcqm.constraints[c2].sense() == Sense::EQ);
                CHECK(newcqm.constraints[c2].rhs() == 7);

                CHECK(newcqm.vartype(v3) == Vartype::INTEGER);
                CHECK(newcqm.lower_bound(v3) == 0);
                CHECK(newcqm.upper_bound(v3) ==
                      vartype_limits<double, Vartype::INTEGER>::default_max());
                CHECK(newcqm.constraints[c3a].linear(v3) == 1);
                CHECK(newcqm.constraints[c3a].sense() == Sense::LE);
                CHECK(newcqm.constraints[c3a].rhs() == 5.5);
                CHECK(newcqm.constraints[c3b].linear(v3) == 1);
                CHECK(newcqm.constraints[c3b].sense() == Sense::GE);
                CHECK(newcqm.constraints[c3b].rhs() == 4.5);
            }
        }

        WHEN("passed through the trivial presolver") {
            auto presolver = presolve::PreSolver<double>(cqm);
            presolver.add_presolver<presolve::techniques::TrivialPresolver<double>>();
            presolver.apply();

            THEN("several constraints/variables are removed") {
                const auto& newcqm = presolver.model();
                const auto& postsolver = presolver.postsolver();

                CHECK(newcqm.num_constraints() == 0);

                CHECK(newcqm.num_variables() == 1);

                // see if we restore the original problem
                auto reduced = std::vector<int>{3};

                auto original = postsolver.apply(reduced);

                CHECK(original == std::vector<int>{3, 5, 7, 5});
            }
        }
    }
}

}  // namespace dimod
