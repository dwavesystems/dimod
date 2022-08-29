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

namespace dimod {

SCENARIO("constrained quadratic models") {
    GIVEN("an empty CQM") {
        auto cqm = ConstrainedQuadraticModel<double>();

        THEN("some basic properties can be discovered") {
            CHECK(cqm.num_variables() == 0);
            CHECK(cqm.num_constraints() == 0);
        }

        THEN("the cqm can be copied") {
            auto cqm2 = cqm;
        }

        WHEN("the objective is set via a QM") {
            auto qm = QuadraticModel<float>();
            qm.add_variable(Vartype::SPIN);
            qm.add_variable(Vartype::REAL, -5, +17);
            qm.set_linear(0, {1.5, -5});
            qm.add_quadratic(0, 1, -2);
            qm.set_offset(5);

            cqm.set_objective(qm);

            REQUIRE(cqm.num_variables() == 2);
            CHECK(cqm.objective().num_variables() == 2);
            REQUIRE(cqm.objective().num_interactions() == 1);
            CHECK(cqm.objective().linear(0) == 1.5);
            CHECK(cqm.objective().linear(1) == -5);
            CHECK(cqm.objective().quadratic_at(0, 1) == -2);
            CHECK(cqm.objective().offset() == 5);
            CHECK(cqm.objective().lower_bound(0) == qm.lower_bound(0));
            CHECK(cqm.objective().upper_bound(0) == qm.upper_bound(0));
            CHECK(cqm.objective().lower_bound(1) == qm.lower_bound(1));
            CHECK(cqm.objective().upper_bound(1) == qm.upper_bound(1));
        }

        WHEN("one constraint is added") {
            cqm.add_variable(Vartype::INTEGER, -5, 5);
            cqm.add_variables(9, Vartype::BINARY);
            REQUIRE(cqm.num_variables() == 10);

            std::vector<int> variables {2, 4, 7};
            std::vector<float> biases {20, 40, 70};

            auto c0 = cqm.add_linear_constraint(variables, biases, Sense::LE, 5);

            REQUIRE(cqm.num_constraints() == 1);
            CHECK(cqm.constraints[c0].linear(0) == 0);
            CHECK(cqm.constraints[c0].linear(2) == 20);
            CHECK(cqm.constraints[c0].linear(4) == 40);
            CHECK(cqm.constraints[c0].linear(7) == 70);
            CHECK(cqm.constraints[c0].lower_bound(0) == -5);
            CHECK(cqm.constraints[c0].upper_bound(0) == +5);
            CHECK(cqm.constraints[c0].vartype(0) == Vartype::INTEGER);
            CHECK(cqm.constraints[c0].lower_bound(2) == 0);
            CHECK(cqm.constraints[c0].upper_bound(2) == 1);
            CHECK(cqm.constraints[c0].vartype(2) == Vartype::BINARY);

            CHECK(cqm.constraints[0].vartype(2) == Vartype::BINARY);
        }

        WHEN("10 variables and two empty constraints are added") {
            cqm.add_constraints(2, Sense::EQ);
            cqm.add_variables(10, Vartype::INTEGER);

            THEN("quadratic biases can be added") {
                REQUIRE(cqm.num_constraints() == 2);

                cqm.constraints[0].add_quadratic(0, 1, 1.5);

                CHECK(cqm.constraints[0].num_interactions() == 1);
                CHECK(cqm.constraints[0].num_interactions(0) == 1);
                CHECK(cqm.constraints[0].num_interactions(1) == 1);
                CHECK(cqm.constraints[0].quadratic(0, 1) == 1.5);
                CHECK(cqm.constraints[0].quadratic(1, 0) == 1.5);
            }
        }
    }

    GIVEN("a CQM with an objective and a constraint") {
        auto cqm = ConstrainedQuadraticModel<double>();

        auto qm = QuadraticModel<double>();
        qm.add_variables(Vartype::INTEGER, 5);
        qm.add_variables(Vartype::BINARY, 5);
        qm.set_linear(0, {0, 1, -2, 3, -4, 5, -6, 7, -8, 9});
        qm.add_quadratic(0, 1, 1);
        qm.add_quadratic(1, 2, 2);
        qm.set_offset(5);
        cqm.set_objective(qm);

        cqm.add_constraints(1, Sense::EQ);
        cqm.constraints[0].set_linear(2, 2);
        cqm.constraints[0].set_linear(5, -5);
        cqm.constraints[0].add_quadratic(2, 4, 8);
        cqm.constraints[0].set_offset(4);

        // todo: test Move, copy, constructors and operators
    }
}


}  // namespace dimod
