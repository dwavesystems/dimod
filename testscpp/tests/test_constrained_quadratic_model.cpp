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
#include "dimod/quadratic_model.h"

namespace dimod {

SCENARIO("ConstrainedQuadraticModel  tests") {
    GIVEN("an empty CQM") {
        auto cqm = ConstrainedQuadraticModel<double>();

        THEN("some basic properties can be discovered") {
            CHECK(cqm.num_variables() == 0);
            CHECK(cqm.num_constraints() == 0);
        }

        THEN("the cqm can be copied") {
            auto cqm2 = cqm;

            THEN("some basic properties can be discovered") {
                CHECK(cqm2.num_variables() == 0);
                CHECK(cqm2.num_constraints() == 0);
            }
        }

        WHEN("we add variables") {
            cqm.add_variables(Vartype::INTEGER, 5);
            cqm.add_variables(Vartype::REAL, 3, -10, 10);
            cqm.add_variable(Vartype::SPIN);
            cqm.add_variable(Vartype::BINARY, 0, 1);

            THEN("we can read them off appropriately") {
                REQUIRE(cqm.num_variables() == 10);

                for (int i = 0; i < 5; ++i) CHECK(cqm.vartype(i) == Vartype::INTEGER);
                for (int i = 5; i < 8; ++i) CHECK(cqm.vartype(i) == Vartype::REAL);
                for (int i = 5; i < 8; ++i) CHECK(cqm.lower_bound(i) == -10);
                for (int i = 5; i < 8; ++i) CHECK(cqm.upper_bound(i) == +10);
                CHECK(cqm.vartype(8) == Vartype::SPIN);
                CHECK(cqm.vartype(9) == Vartype::BINARY);
            }

            AND_WHEN("we try to set the linear biases in the objective") {
                cqm.objective.set_linear(0, 1.5);

                THEN("it is reflected in the model") {
                    REQUIRE(cqm.objective.num_variables() == 1);
                    CHECK(cqm.objective.linear(0) == 1.5);
                }
            }
        }

        AND_GIVEN("a quadratic model") {
            auto qm = QuadraticModel<double>();
            auto u = qm.add_variable(Vartype::INTEGER, -5, 5);
            auto v = qm.add_variable(Vartype::BINARY);
            qm.set_linear(u, 1);
            qm.set_linear(v, 2);
            qm.set_quadratic(u, v, 1.5);
            qm.set_offset(10);

            WHEN("we set the objective via the set_objective() method") {
                cqm.set_objective(qm);

                THEN("the objective updates appropriately") {
                    REQUIRE(cqm.objective.num_variables() == 2);
                    REQUIRE(cqm.objective.num_interactions() == 1);
                    CHECK(cqm.objective.linear(0) == 1);
                    CHECK(cqm.objective.linear(1) == 2);
                    CHECK(cqm.objective.quadratic(0, 1) == 1.5);
                    CHECK(cqm.objective.offset() == 10);
                    CHECK(cqm.lower_bound(0) == -5);
                    CHECK(cqm.upper_bound(0) == 5);
                    CHECK(cqm.vartype(0) == Vartype::INTEGER);
                    CHECK(cqm.vartype(1) == Vartype::BINARY);
                }
            }

            WHEN("we set the objective via the set_objective() with a relabel") {
                cqm.add_variable(Vartype::BINARY);
                cqm.add_variable(Vartype::INTEGER, -5, 5);
                cqm.set_objective(qm, std::vector<int>{1, 0});

                THEN("the objective updates appropriately") {
                    REQUIRE(cqm.objective.num_variables() == 2);
                    REQUIRE(cqm.objective.num_interactions() == 1);
                    CHECK(cqm.objective.linear(1) == 1);
                    CHECK(cqm.objective.linear(0) == 2);
                    CHECK(cqm.objective.quadratic(0, 1) == 1.5);
                    CHECK(cqm.objective.offset() == 10);
                    CHECK(cqm.lower_bound(1) == -5);
                    CHECK(cqm.upper_bound(1) == 5);
                    CHECK(cqm.vartype(1) == Vartype::INTEGER);
                    CHECK(cqm.vartype(0) == Vartype::BINARY);
                }

                AND_WHEN("that objective is overwritten") {
                    auto qm2 = QuadraticModel<double>();
                    qm2.add_variable(Vartype::BINARY);
                    qm2.set_linear(0, 10);
                    cqm.set_objective(qm2, std::vector<int>{0});

                    THEN("everything updates as expected") {
                        REQUIRE(cqm.objective.num_variables() == 1);
                        REQUIRE(cqm.objective.num_interactions() == 0);
                        CHECK(cqm.objective.linear(1) == 0);
                        CHECK(cqm.objective.linear(0) == 10);
                        CHECK(cqm.objective.quadratic(0, 1) == 0);
                        CHECK(cqm.objective.offset() == 0);
                        CHECK(cqm.lower_bound(1) == -5);
                        CHECK(cqm.upper_bound(1) == 5);
                        CHECK(cqm.vartype(1) == Vartype::INTEGER);
                        CHECK(cqm.vartype(0) == Vartype::BINARY);
                    }
                }
            }
        }

        WHEN("we add an empty constraint") {
            auto c0 = cqm.add_constraint();

            THEN("that constraint is well formed") {
                CHECK(cqm.constraint_ref(c0).num_variables() == 0);
                CHECK(cqm.constraint_ref(c0).num_interactions() == 0);
                CHECK(!cqm.constraint_ref(c0).is_soft());
                CHECK(cqm.constraint_ref(c0).rhs() == 0);
            }

            AND_WHEN("we manipulate that constraint") {
                auto u = cqm.add_variable(Vartype::BINARY);
                auto v = cqm.add_variable(Vartype::INTEGER, -5, 5);
                cqm.constraint_ref(c0).add_linear(u, 1.5);
                cqm.constraint_ref(c0).add_linear(v, 12.5);
            }
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
            CHECK(cqm.objective.num_variables() == 2);
            REQUIRE(cqm.objective.num_interactions() == 1);
            CHECK(cqm.objective.linear(0) == 1.5);
            CHECK(cqm.objective.linear(1) == -5);
            CHECK(cqm.objective.quadratic_at(0, 1) == -2);
            CHECK(cqm.objective.offset() == 5);
            CHECK(cqm.objective.lower_bound(0) == qm.lower_bound(0));
            CHECK(cqm.objective.upper_bound(0) == qm.upper_bound(0));
            CHECK(cqm.objective.lower_bound(1) == qm.lower_bound(1));
            CHECK(cqm.objective.upper_bound(1) == qm.upper_bound(1));
        }

        WHEN("10 variables and two empty constraints are added") {
            cqm.add_constraints(2);
            cqm.add_variable(Vartype::INTEGER);
            cqm.add_variable(Vartype::INTEGER);

            THEN("quadratic biases can be added") {
                REQUIRE(cqm.num_constraints() == 2);

                cqm.constraint_ref(0).add_quadratic(0, 1, 1.5);

                CHECK(cqm.constraint_ref(0).num_interactions() == 1);
                CHECK(cqm.constraint_ref(0).num_interactions(0) == 1);
                CHECK(cqm.constraint_ref(0).num_interactions(1) == 1);
                CHECK(cqm.constraint_ref(0).quadratic(0, 1) == 1.5);
                CHECK(cqm.constraint_ref(0).quadratic(1, 0) == 1.5);
            }
        }
    }

    GIVEN("a CQM with an objective and a constraint") {
        auto cqm = ConstrainedQuadraticModel<double>();

        // auto qm = QuadraticModel<double>();
        cqm.add_variables(Vartype::INTEGER, 5, 0, 5);
        cqm.add_variables(Vartype::BINARY, 5);
        // qm.set_linear(0, {0, 1, -2, 3, -4, 5, -6, 7, -8, 9});
        // qm.add_quadratic(0, 1, 1);
        // qm.add_quadratic(1, 2, 2);
        // qm.set_offset(5);
        // cqm.set_objective(qm);

        // cqm.add_constraints(1, Sense::EQ);
        // cqm.constraint_ref(0).set_linear(2, 2);
        // cqm.constraint_ref(0).set_linear(5, -5);
        // cqm.constraint_ref(0).add_quadratic(2, 4, 8);
        // cqm.constraint_ref(0).set_offset(4);

        // todo: test Move, copy, constructors and operators
    }

    GIVEN("a CQM with 30 variables") {
        auto cqm = ConstrainedQuadraticModel<double>();

        cqm.add_variables(Vartype::BINARY, 10);
        cqm.add_variables(Vartype::INTEGER, 10, -5, 5);
        cqm.add_variables(Vartype::REAL, 10, -100, 105);

        WHEN("we add a constraint over only a few variables") {
            auto c0 = cqm.add_constraint();

            auto& constraint = cqm.constraint_ref(c0);

            constraint.add_linear(5, 15);
            constraint.add_linear(7, 105);
            constraint.add_linear(3, 5);
            constraint.add_quadratic(5, 7, 56);
            constraint.add_quadratic(3, 7, 134);

            THEN("we can iterate over the quadratic interactions") {
                auto it = constraint.cbegin_quadratic();

                CHECK(it->u == 7);
                CHECK(it->v == 5);
                CHECK(it->bias == 56);
                it++;
                CHECK(it->u == 3);
                CHECK(it->v == 7);
                CHECK(it->bias == 134);
                it++;
                CHECK(it == constraint.cend_quadratic());
            }

            THEN("we can iterate over the neighborhoods") {
                auto it = constraint.cbegin_neighborhood(7);

                CHECK(it->v == 5);
                CHECK(it->bias == 56);
                it++;
                CHECK(it->v == 3);
                CHECK(it->bias == 134);
                it++;
                CHECK(it == constraint.cend_neighborhood(7));
            }
        }

        WHEN("we use new_constraint()") {
            auto constraint = cqm.new_constraint();

            THEN("the constraint references the CQM, but is disconnected") {
                CHECK(constraint.vartype(4) == Vartype::BINARY);
                CHECK(cqm.num_constraints() == 0);
            }

            AND_WHEN("that constraint is modified and moved") {
                constraint.add_quadratic(5, 6, 10);
                constraint.set_offset(5);
                constraint.set_linear(16, -15);

                cqm.add_constraint(std::move(constraint));

                THEN("it has been moved correctly") {
                    REQUIRE(cqm.num_constraints() == 1);
                    REQUIRE(cqm.constraint_ref(0).num_variables() == 3);

                    CHECK(cqm.constraint_ref(0).linear(16) == -15);
                    CHECK(cqm.constraint_ref(0).quadratic(5, 6) == 10);
                    CHECK(cqm.constraint_ref(0).offset() == 5);
                }
            }
        }
    }

    GIVEN("a CQM with a contraint") {
        auto cqm = ConstrainedQuadraticModel<double>();
        cqm.add_variables(Vartype::BINARY, 3);

        auto c0 = cqm.add_constraint();
        cqm.constraint_ref(c0).set_linear(2, 1);
        cqm.constraint_ref(c0).set_sense(Sense::GE);
        cqm.constraint_ref(c0).set_rhs(1);

        std::vector<int> sample{0, 0, 1};
        CHECK(cqm.constraint_ref(c0).energy(sample.begin()) == 1);

        WHEN("we clear it") {
            cqm.clear();

            THEN("it is empty") {
                CHECK(cqm.num_variables() == 0);
                CHECK(cqm.num_constraints() == 0);
                CHECK(cqm.objective.num_variables() == 0);
            }
        }
    }

    GIVEN("A CQM with several constraints") {
        auto cqm = ConstrainedQuadraticModel<double>();
        cqm.add_variables(Vartype::BINARY, 7);

        auto c0 = cqm.add_constraint();
        auto c1 = cqm.add_constraint();
        auto c2 = cqm.add_constraint();

        cqm.constraint_ref(c0).add_linear(0, 1.5);
        cqm.constraint_ref(c0).add_linear(1, 2.5);
        cqm.constraint_ref(c0).add_linear(2, 3.5);

        cqm.constraint_ref(c1).add_linear(2, 4.5);
        cqm.constraint_ref(c1).add_linear(3, 5.5);
        cqm.constraint_ref(c1).add_linear(4, 6.5);

        cqm.constraint_ref(c2).add_linear(5, 8.5);
        cqm.constraint_ref(c2).add_linear(4, 7.5);

        THEN("we can read off the values as expected") {
            const auto& const2 = cqm.constraint_ref(c2);
            REQUIRE(const2.num_variables() == 2);
            CHECK(const2.linear(4) == 7.5);
            CHECK(const2.linear(5) == 8.5);
            CHECK(const2.variables() == std::vector<int>{5, 4});
        }

        WHEN("constraint c1 is removed") {
            cqm.remove_constraint(c1);

            THEN("all variables are preserved, but one constraint is removed") {
                CHECK(cqm.num_variables() == 7);
                CHECK(cqm.num_constraints() == 2);
                CHECK(cqm.constraint_ref(1).linear(4) == 7.5);
            }
        }

        WHEN("a variable is removed") {
            cqm.remove_variable(3);

            THEN("everything is updated appropriately") {
                REQUIRE(cqm.num_variables() == 6);

                const auto& const0 = cqm.constraint_ref(c0);
                REQUIRE(const0.num_variables() == 3);
                CHECK(const0.linear(0) == 1.5);
                CHECK(const0.linear(1) == 2.5);
                CHECK(const0.linear(2) == 3.5);
                CHECK(const0.variables() == std::vector<int>{0, 1, 2});

                const auto& const1 = cqm.constraint_ref(c1);
                REQUIRE(const1.num_variables() == 2);
                CHECK(const1.linear(2) == 4.5);
                CHECK(const1.linear(3) == 6.5);                       // reindexed
                CHECK(const1.variables() == std::vector<int>{2, 3});  // partly reindexed

                const auto& const2 = cqm.constraint_ref(c2);
                REQUIRE(const2.num_variables() == 2);
                CHECK(const2.linear(3) == 7.5);                       // reindexed
                CHECK(const2.linear(4) == 8.5);                       // reindexed
                CHECK(const2.variables() == std::vector<int>{4, 3});  // reindexed
            }
        }
    }

    GIVEN("A CQM with two constraints") {
        auto cqm = ConstrainedQuadraticModel<double>();
        auto x = cqm.add_variable(Vartype::BINARY);
        auto y = cqm.add_variable(Vartype::BINARY);
        auto i = cqm.add_variable(Vartype::INTEGER);
        auto j = cqm.add_variable(Vartype::INTEGER);
        // auto z = cqm.add_variable(Vartype::BINARY);

        cqm.objective.set_linear(x, 1);
        cqm.objective.set_linear(y, 2);
        cqm.objective.set_linear(i, 3);
        cqm.objective.set_linear(j, 4);

        auto& const0 = cqm.constraint_ref(cqm.add_constraint());
        const0.set_linear(i, 3);
        const0.set_quadratic(x, j, 2);
        const0.set_quadratic(i, j, 5);

        CHECK(cqm.objective.num_variables() == 4);
        CHECK(const0.num_variables() == 3);

        WHEN("we substitute a variable with a 0 multiplier") {
            cqm.substitute_variable(x, 0, 0);

            THEN("the biases are updated in the objective and constraint") {
                REQUIRE(cqm.objective.num_variables() == 4);
                CHECK(cqm.objective.linear(x) == 0);
                CHECK(cqm.objective.linear(y) == 2);
                CHECK(cqm.objective.linear(i) == 3);
                CHECK(cqm.objective.linear(j) == 4);

                auto& const0 = cqm.constraint_ref(0);
                REQUIRE(const0.num_variables() == 3);
                REQUIRE(const0.num_interactions() == 2);
                CHECK(const0.linear(i) == 3);
                CHECK(const0.quadratic_at(x, j) == 0);
                CHECK(const0.quadratic_at(i, j) == 5);
            }
        }

        // WHEN("we fix a variable") {
        //     cqm.fix_variable(x, 0);

        //     THEN("everything is updated correctly") {
        //         REQUIRE(cqm.num_variables() == 3);

        //         REQUIRE(const0.num_variables() == 2);
        //         REQUIRE(const0.linear(i-1) == 3);
        //     }
        // }
    }

    GIVEN("A constraint with one-hot constraints") {
        auto cqm = ConstrainedQuadraticModel<double>();
        cqm.add_variables(Vartype::BINARY, 10);
        auto c0 = cqm.add_linear_constraint({0, 1, 2, 3, 4}, {1, 1, 1, 1, 1}, Sense::EQ, 1);
        auto c1 = cqm.add_linear_constraint({5, 6, 7, 8, 9}, {2, 2, 2, 2, 2}, Sense::EQ, 2);

        THEN("the constraints can be tests for one-hotness") {
            CHECK(cqm.constraint_ref(c0).is_onehot());
            CHECK(cqm.constraint_ref(c1).is_onehot());
            CHECK(cqm.constraint_ref(c0).is_disjoint(cqm.constraint_ref(c1)));
        }

        WHEN("we change a linear bias") {
            cqm.constraint_ref(c0).set_linear(0, 1.5);

            THEN("it's no longer one-hot") { CHECK(!cqm.constraint_ref(c0).is_onehot()); }
        }

        WHEN("we add an overlapping variable") {
            cqm.constraint_ref(c0).set_linear(5, 1);
            THEN("they are no longer disjoint") {
                CHECK(cqm.constraint_ref(c0).is_onehot());
                CHECK(cqm.constraint_ref(c1).is_onehot());
                CHECK(!cqm.constraint_ref(c0).is_disjoint(cqm.constraint_ref(c1)));
            }
        }
    }
}

TEST_CASE("Bug 0") {
    GIVEN("A CQM with a single constraint") {
        auto cqm = dimod::ConstrainedQuadraticModel<double>();
        cqm.add_variables(Vartype::BINARY, 5);
        cqm.add_linear_constraint({0, 3, 1, 2}, {1, 2, 3, 4}, Sense::GE, -1);

        WHEN("we start removing variables") {
            cqm.remove_variable(0);
            CHECK(cqm.constraint_ref(0).variables() == std::vector<int>{2, 0, 1});
            cqm.remove_variable(2);
            CHECK(cqm.constraint_ref(0).variables() == std::vector<int>{0, 1});
            cqm.remove_variable(1);
            CHECK(cqm.constraint_ref(0).variables() == std::vector<int>{0});
        }
    }
}

TEST_CASE("Test constraints property") {
    GIVEN("a CQM with one variable and 10 constraints") {
        auto cqm = ConstrainedQuadraticModel<double>();
        auto x = cqm.add_variable(Vartype::BINARY);
        for (int i = 0; i < 10; ++i) {
            cqm.add_linear_constraint({x}, {static_cast<double>(i)}, Sense::EQ, i);
        }

        THEN("range-based for loops work over the constraints") {
            int i = 0;
            for (auto& c : cqm.constraints) {
                CHECK(c.linear(x) == i);
                c.set_linear(x, i - 1);  // can modify
                ++i;
            }
        }

        THEN("const iteration functions") {
            int i = 0;
            for (auto it = cqm.constraints.cbegin(); it != cqm.constraints.cend(); ++it, ++i) {
                CHECK(it->linear(x) == i);
                // it->set_linear(x, i-1);  // raises compiler error
            }
        }

        THEN("we can access the constraints by index") {
            for (int i = 0; i < 10; ++i) {
                CHECK(cqm.constraints[i].linear(x) == i);
                CHECK(cqm.constraints.at(i).linear(x) == i);

                cqm.constraints[i].set_linear(x, i + 1);
                CHECK(cqm.constraints[i].linear(x) == i + 1);

                cqm.constraints.at(i).set_linear(x, i - 1);
                CHECK(cqm.constraints[i].linear(x) == i - 1);

                CHECK_THROWS_AS(cqm.constraints.at(100), std::out_of_range);
            }
        }

        AND_GIVEN("a const reference to the cqm") {
            const dimod::ConstrainedQuadraticModel<double>& const_cqm = cqm;

            THEN("range-based for loops work over the constraints") {
                int i = 0;
                for (auto& c : const_cqm.constraints) {
                    CHECK(c.linear(x) == i);
                    // c.set_linear(x, i-1);  // raises compiler error
                    ++i;
                }
            }

            THEN("we can access the constraints by index") {
                for (int i = 0; i < 10; ++i) {
                    CHECK(const_cqm.constraints[i].linear(x) == i);
                    CHECK(const_cqm.constraints.at(i).linear(x) == i);

                    CHECK_THROWS_AS(const_cqm.constraints.at(100), std::out_of_range);
                }
            }
        }
    }
}

TEST_CASE("Test Constraint::scale()") {
    GIVEN("A CQM with several constraints") {
        auto cqm = dimod::ConstrainedQuadraticModel<double>();
        cqm.add_variables(Vartype::BINARY, 1);
        auto c0 = cqm.add_linear_constraint({0}, {4}, Sense::EQ, 2);
        auto c1 = cqm.add_linear_constraint({0}, {4}, Sense::LE, 2);
        auto c2 = cqm.add_linear_constraint({0}, {4}, Sense::GE, 2);

            cqm.constraint_ref(c0).set_offset(8);
            cqm.constraint_ref(c1).set_offset(8);
            cqm.constraint_ref(c2).set_offset(8);

        WHEN("we scale the constraints by a positive number") {
            cqm.constraint_ref(c0).scale(.5);
            cqm.constraint_ref(c1).scale(.5);
            cqm.constraint_ref(c2).scale(.5);

            THEN("the biases are scaled and everything else stays the same") {
                for (auto c = c0; c <= c2; ++c) {
                    CHECK(cqm.constraint_ref(c).offset() == 4);
                    CHECK(cqm.constraint_ref(c).linear(0) == 2);
                    CHECK(cqm.constraint_ref(c).rhs() == 1);
                }

                CHECK(cqm.constraint_ref(c0).sense() == Sense::EQ);
                CHECK(cqm.constraint_ref(c1).sense() == Sense::LE);
                CHECK(cqm.constraint_ref(c2).sense() == Sense::GE);
            }
        }

        WHEN("we scale the constraints by a negative number") {
            cqm.constraint_ref(c0).scale(-.5);
            cqm.constraint_ref(c1).scale(-.5);
            cqm.constraint_ref(c2).scale(-.5);

            THEN("the biases are scaled and some of the signs flip") {
                for (auto c = c0; c <= c2; ++c) {
                    CHECK(cqm.constraint_ref(c).offset() == -4);
                    CHECK(cqm.constraint_ref(c).linear(0) == -2);
                    CHECK(cqm.constraint_ref(c).rhs() == -1);
                }

                CHECK(cqm.constraint_ref(c0).sense() == Sense::EQ);
                CHECK(cqm.constraint_ref(c1).sense() == Sense::GE);  // flipped
                CHECK(cqm.constraint_ref(c2).sense() == Sense::LE);  // flipped
            }
        }
    }
}

TEST_CASE("Test CQM.add_constraint()") {
    GIVEN("A CQM and a BQM") {
        auto cqm = dimod::ConstrainedQuadraticModel<double>();
        cqm.add_variables(Vartype::BINARY, 5);

        auto bqm = dimod::BinaryQuadraticModel<double>(3, Vartype::BINARY);
        bqm.set_linear(0, -1);
        bqm.set_linear(1, -2);
        bqm.set_linear(2, -3);
        bqm.set_quadratic(0, 2, 1.5);
        bqm.set_offset(4);

        WHEN("we add the BQM as a constraint") {
            cqm.add_constraint(bqm, Sense::EQ, 1, std::vector<int>{4, 2, 0});

            THEN("it was copied correctly") {
                auto& constraint = cqm.constraint_ref(0);

                CHECK(constraint.variables() == std::vector<int>{4, 2, 0});
                CHECK(constraint.linear(4) == -1);
                CHECK(constraint.linear(0) == -3);
                CHECK(constraint.quadratic_at(4, 0) == 1.5);
                CHECK(constraint.offset() == 4);
                CHECK(constraint.sense() == Sense::EQ);
                CHECK(constraint.rhs() == 1);
            }
        }

        WHEN("we move the BQM as a constraint") {
            std::vector<int> mapping = {4, 2, 0};
            cqm.add_constraint(std::move(bqm), Sense::LE, 2, std::move(mapping));

            THEN("it was moved correctly") {
                auto& constraint = cqm.constraint_ref(0);

                CHECK(constraint.variables() == std::vector<int>{4, 2, 0});
                CHECK(constraint.linear(4) == -1);
                CHECK(constraint.linear(0) == -3);
                CHECK(constraint.quadratic_at(4, 0) == 1.5);
                CHECK(constraint.offset() == 4);
                CHECK(constraint.sense() == Sense::LE);
                CHECK(constraint.rhs() == 2);

                CHECK(bqm.num_variables() == 0);  // moved
                CHECK(mapping.size() == 0);       // moved
            }
        }
    }
}

TEST_CASE("Test CQM copy assignment") {
    GIVEN("A CQM") {
        auto cqm = dimod::ConstrainedQuadraticModel<double>();
        auto s = cqm.add_variable(Vartype::SPIN);
        auto x = cqm.add_variable(Vartype::BINARY);
        auto i = cqm.add_variable(Vartype::INTEGER);
        auto t = cqm.add_variable(Vartype::SPIN);

        cqm.objective.set_quadratic(s, i, 1);
        cqm.objective.set_quadratic(t, x, 1);
        cqm.objective.set_quadratic(s, t, 1);

        auto constraint = cqm.new_constraint();
        constraint.set_linear(s, 1);
        constraint.set_linear(t, 1);
        constraint.set_quadratic(s, t, 1);
        constraint.set_sense(Sense::LE);
        constraint.set_rhs(5);
        cqm.add_constraint(std::move(constraint));

        WHEN("we copy it using copy assignment operator") {
            auto cqm2 = cqm;

            AND_WHEN("we mutate the copy") {
                cqm2.objective.set_linear(s, 10);
                cqm2.constraint_ref(0).set_linear(s, 10);

                THEN("the original is not affected") {
                    CHECK(cqm.objective.linear(s) == 0);
                    CHECK(cqm.constraint_ref(0).linear(s) == 1);
                }
            }
        }
    }
}

TEST_CASE("Test CQM::change_vartype()") {
    GIVEN("A CQM with several different vartypes") {
        auto cqm = dimod::ConstrainedQuadraticModel<double>();
        auto s = cqm.add_variable(Vartype::SPIN);
        auto x = cqm.add_variable(Vartype::BINARY);
        auto i = cqm.add_variable(Vartype::INTEGER);
        auto t = cqm.add_variable(Vartype::SPIN);

        cqm.objective.set_quadratic(s, i, 1);
        cqm.objective.set_quadratic(t, x, 1);
        cqm.objective.set_quadratic(s, t, 1);

        auto constraint = cqm.new_constraint();
        constraint.set_linear(s, 1);
        constraint.set_linear(t, 1);
        constraint.set_quadratic(s, t, 1);
        constraint.set_sense(Sense::LE);
        constraint.set_rhs(5);
        cqm.add_constraint(std::move(constraint));

        std::vector<int> sample = {-1, 1, 105, +1};

        auto objective_energy = cqm.objective.energy(sample.begin());
        auto constraint_energy = cqm.constraint_ref(0).energy(sample.begin());

        THEN("when we change the vartype of the spin variables") {
            cqm.change_vartype(Vartype::BINARY, s);
            cqm.change_vartype(Vartype::BINARY, t);

            sample[s] = (sample[s] + 1) / 2;
            sample[t] = (sample[t] + 1) / 2;

            CHECK(objective_energy == cqm.objective.energy(sample.begin()));
            CHECK(constraint_energy == cqm.constraint_ref(0).energy(sample.begin()));
        }
    }
}

TEST_CASE("Test CQM.constraint_weak_ptr()") {
    GIVEN("A CQM with several constraints") {
        auto cqm = dimod::ConstrainedQuadraticModel<double>();
        cqm.add_variables(Vartype::BINARY, 10);
        cqm.add_linear_constraint({0, 1, 2}, {0, 1, 2}, Sense::EQ, 0);
        cqm.add_linear_constraint({1, 2, 3}, {1, 2, 3}, Sense::LE, 1);
        cqm.add_linear_constraint({2, 3, 4}, {2, 3, 4}, Sense::GE, 2);

        WHEN("we get a weak_ptr referencing the third constraint") {
            auto wk_ptr = cqm.constraint_weak_ptr(2);
            REQUIRE(!wk_ptr.expired());
            CHECK(wk_ptr.lock()->linear(4) == 4);
            cqm.remove_constraint(0);
            CHECK(wk_ptr.lock()->linear(4) == 4);  // should still work
            cqm.remove_constraint(1);
            CHECK(wk_ptr.expired());
        }

        WHEN("we get a weak_ptr referencing a third constraint from a const version") {
            auto wk_ptr = static_cast<const dimod::ConstrainedQuadraticModel<double>&>(cqm)
                                  .constraint_weak_ptr(2);
            REQUIRE(!wk_ptr.expired());
            CHECK(wk_ptr.lock()->linear(4) == 4);
            cqm.remove_constraint(0);
            CHECK(wk_ptr.lock()->linear(4) == 4);  // should still work
            cqm.remove_constraint(1);
            CHECK(wk_ptr.expired());
        }
    }
}

TEST_CASE("Test Expression::add_quadratic()") {
    GIVEN("A CQM with two variables with vartypes") {
        auto cqm = dimod::ConstrainedQuadraticModel<double>();
        auto i = cqm.add_variable(Vartype::INTEGER);
        auto x = cqm.add_variable(Vartype::BINARY);

        auto c0 = cqm.add_linear_constraint({i, x}, {0, 0}, Sense::EQ, 1);
        auto c1 = cqm.add_linear_constraint({x, i}, {0, 0}, Sense::LE, 2);

        WHEN("we add self-loops") {
            cqm.constraint_ref(c0).add_quadratic(i, i, 1.5);
            cqm.constraint_ref(c0).add_quadratic(x, x, 2.5);
            cqm.constraint_ref(c1).add_quadratic(i, i, 1.5);
            cqm.constraint_ref(c1).add_quadratic(x, x, 2.5);

            THEN("the are applied correctly") {
                CHECK(cqm.constraint_ref(c0).linear(i) == 0);
                CHECK(cqm.constraint_ref(c0).linear(x) == 2.5);
                CHECK(cqm.constraint_ref(c0).quadratic(i, i) == 1.5);
                CHECK(cqm.constraint_ref(c0).quadratic(x, x) == 0);

                CHECK(cqm.constraint_ref(c1).linear(i) == 0);
                CHECK(cqm.constraint_ref(c1).linear(x) == 2.5);
                CHECK(cqm.constraint_ref(c1).quadratic(i, i) == 1.5);
                CHECK(cqm.constraint_ref(c1).quadratic(x, x) == 0);
            }
        }
    }
}

}  // namespace dimod
