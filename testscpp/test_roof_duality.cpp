// Copyright 2021 D-Wave Systems Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <dimod/quadratic_model.h>

#include "catch2/catch.hpp"
#include "dwave-preprocessing/fix_variables.hpp"

namespace fix_variables_ {

TEST_CASE("Tests for fixQuboVariables", "[roofduality]") {
    SECTION("Test simple case") {
        float Q[9] = {22, -4, 0, 0, 0, 4, 0, 0, -2};

        int num_vars = 3;
        auto bqm = dimod::BinaryQuadraticModel<float, int>(Q, num_vars, dimod::Vartype::BINARY);

        // Checking fixQuboVariables(PosiformInfo, ..)
        PosiformInfo<dimod::BinaryQuadraticModel<float, int>, capacity_type> posiform(bqm);
        std::vector<std::pair<int, int>> fixed_vars;
        fixQuboVariables(posiform, num_vars, true, fixed_vars);

        // Checking fixQuboVariables(BinaryQuadraticModel, ..)
        auto result = fixQuboVariables(bqm, true);
        auto lower_bound = result.first;
        auto fixed_vars2 = result.second;

        REQUIRE(lower_bound == -2);
        REQUIRE(fixed_vars == fixed_vars2);
        REQUIRE(fixed_vars.size() == static_cast<std::size_t>(num_vars));

        for (auto var : fixed_vars) {
            if (var.first == 2) {
                REQUIRE(var.second == 1);
            } else {
                REQUIRE(var.second == 0);
            }
        }
    }

    SECTION("Test offset") {
        float Q[9] = {22, -4, 0, 0, 0, 4, 0, 0, -2};

        int num_vars = 3;
        auto bqm = dimod::BinaryQuadraticModel<float, int>(Q, num_vars, dimod::Vartype::BINARY);

        auto bqm_offset = -5;
        auto result = fixQuboVariables(bqm, true, bqm_offset);
        auto lower_bound = result.first;
        REQUIRE(lower_bound == (-2 + bqm_offset));
    }

    SECTION("Test empty case") {
        float Q[] = {};
        auto bqm = dimod::BinaryQuadraticModel<float, int>(Q, 0, dimod::Vartype::BINARY);

        auto result = fixQuboVariables(bqm, true);
        auto lower_bound = result.first;
        auto fixed_vars = result.second;

        REQUIRE(lower_bound == 0);
        REQUIRE(fixed_vars.size() == 0);

        result = fixQuboVariables(bqm, false);
        lower_bound = result.first;
        fixed_vars = result.second;

        REQUIRE(lower_bound == 0);
        REQUIRE(fixed_vars.size() == 0);
    }

    SECTION("Test zero bias case") {
        float Q[9] = {1, 0, 0, 0, 0, 0, 0, 0, 0};

        int num_vars = 3;
        auto bqm = dimod::BinaryQuadraticModel<float, int>(Q, num_vars, dimod::Vartype::BINARY);

        auto result = fixQuboVariables(bqm, true);
        auto lower_bound = result.first;
        auto fixed_vars = result.second;

        REQUIRE(lower_bound == 0);
        REQUIRE(fixed_vars.size() == 1);
        REQUIRE(fixed_vars[0].first == 0);
        REQUIRE(fixed_vars[0].second == 0);

        result = fixQuboVariables(bqm, false);
        lower_bound = result.first;
        fixed_vars = result.second;

        REQUIRE(lower_bound == 0);
        REQUIRE(fixed_vars.size() == 3);
        for (auto var : fixed_vars) {
            if (var.first == 0) {
                REQUIRE(var.second == 0);
            } else {
                // variables that didn't contribute should be set to 1
                REQUIRE(var.second == 1);
            }
        }
    }

    SECTION("Test all zero bias case") {
        float Q[4] = {0, 0, 0, 0};

        int num_vars = 2;
        auto bqm = dimod::BinaryQuadraticModel<float, int>(Q, num_vars, dimod::Vartype::BINARY);

        auto result = fixQuboVariables(bqm, true);
        auto lower_bound = result.first;
        auto fixed_vars = result.second;

        REQUIRE(lower_bound == 0);
        REQUIRE(fixed_vars.size() == 0);

        result = fixQuboVariables(bqm, false);
        lower_bound = result.first;
        fixed_vars = result.second;

        REQUIRE(lower_bound == 0);
        REQUIRE(fixed_vars.size() == static_cast<std::size_t>(num_vars));

        for (auto var : fixed_vars) {
            REQUIRE(var.second == 1);
        }
    }

    SECTION("Test from previously found bug (BugSAPI1311)") {
        float Q[4] = {2.2, -4.0, 0, 2.0};

        int num_vars = 2;
        auto bqm = dimod::BinaryQuadraticModel<float, int>(Q, num_vars, dimod::Vartype::BINARY);

        for (auto mode : {true, false}) {
            auto result = fixQuboVariables(bqm, mode);
            auto lower_bound = result.first;
            auto fixed_vars = result.second;

            REQUIRE(lower_bound == 0);
            REQUIRE(fixed_vars.size() == 2);

            // checking order of variables
            REQUIRE(fixed_vars[0].first == 0);
            REQUIRE(fixed_vars[1].first == 1);

            // checking fixed values
            REQUIRE(fixed_vars[0].second == 0);
            REQUIRE(fixed_vars[1].second == 0);
        }
    }
}

TEST_CASE("Tests for PosiformInfo", "[roofduality]") {
    SECTION("Test basic case") {
        float Q[9] = {22, -4, 0, 0, 0, 4, 0, 0, -2};

        int num_vars = 3;
        auto bqm = dimod::BinaryQuadraticModel<float, int>(Q, num_vars, dimod::Vartype::BINARY);

        PosiformInfo<dimod::BinaryQuadraticModel<float, int>, capacity_type> posiform(bqm);
        REQUIRE(posiform.getNumVariables() == num_vars);
        REQUIRE(posiform.getNumLinear() == 2);

        REQUIRE(posiform.getNumQuadratic(0) == 1);
        REQUIRE(posiform.getNumQuadratic(1) == 2);
        REQUIRE(posiform.getNumQuadratic(2) == 1);

        REQUIRE(posiform.getLinear(0) > 0);
        REQUIRE(posiform.getLinear(1) == 0);
        REQUIRE(posiform.getLinear(2) < 0);

        auto span = posiform.getQuadratic(0);
        REQUIRE(std::distance(span.first, span.second) == 1);
        REQUIRE(span.first->v == 1);      // neighbor var
        REQUIRE(span.first->bias == -4);  // quad bias

        span = posiform.getQuadratic(1);
        REQUIRE(std::distance(span.first, span.second) == 1);
        REQUIRE(span.first->v == 2);     // neighbor var
        REQUIRE(span.first->bias == 4);  // quad bias

        span = posiform.getQuadratic(2);
        REQUIRE(std::distance(span.first, span.second) == 0);

        for (int var = 0; var < num_vars; var++) {
            int pos_var = posiform.mapVariableQuboToPosiform(var);
            if (pos_var > 0) {
                REQUIRE(posiform.mapVariablePosiformToQubo(pos_var) == var);
            }
        }
    }

    SECTION("Test zero bias case") {
        float Q[9] = {1, 0, 0, 0, 0, 0, 0, 0, 0};

        int num_vars = 3;
        auto bqm = dimod::BinaryQuadraticModel<float, int>(Q, num_vars, dimod::Vartype::BINARY);
        PosiformInfo<dimod::BinaryQuadraticModel<float, int>, capacity_type> posiform(bqm);

        REQUIRE(posiform.getNumVariables() == 1);

        REQUIRE(posiform.getLinear(0) > 0);

        auto span = posiform.getQuadratic(0);
        REQUIRE(std::distance(span.first, span.second) == 0);

        for (int var = 0; var < num_vars; var++) {
            int pos_var = posiform.mapVariableQuboToPosiform(var);
            if (pos_var > 0) {
                REQUIRE(posiform.mapVariablePosiformToQubo(pos_var) == var);
            }
        }
    }
}

}  // namespace fix_variables_
