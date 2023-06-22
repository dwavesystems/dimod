# distutils: include_dirs = extern/
# distutils: sources = extern/filereaderlp/reader.cpp

# Copyright 2022 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os.path

from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from dimod.constrained.cyconstrained cimport cyConstrainedQuadraticModel, make_cqm
from dimod.libcpp.constrained_quadratic_model cimport ConstrainedQuadraticModel as cppCQM
from dimod.quadratic.cyqm.cyqm_float64 cimport bias_type, index_type


cdef extern from "filereaderlp/model.hpp":
    cdef cppclass Variable:
        string name

    cdef cppclass Expression:
        string name

    cdef cppclass Constraint:
        Expression* expr

    cdef cppclass Model:
        vector[Variable*] variables
        vector[Constraint*] constraints

cdef extern from "filereaderlp/reader.hpp":
    Model readinstance(string filename) except+


# We do a funny there whereby we implement the C++ code here rather than in
# a dedicated .cpp file.
# The reason we use C++ rather than Cython is we do a lot of iterating over
# vectors of pointers which is quite hard to read in Cython
# The reason we do it here is because we're wrapping a library that has both
# .h and .cpp files, so we would no longer be a header-only library if we
# included that in dimod/include/
# In the future, we should reconsider the requirement to be header-only but
# for now this keeps things contained and on this side of the compilation
# barrier.
cdef extern from *: 
    """
    #include <limits>
    #include <unordered_map>

    #include "filereaderlp/model.hpp"
    #include "dimod/constrained_quadratic_model.h"
    #include "dimod/vartypes.h"

    dimod::Vartype variable_type_to_vartype(VariableType type) {
        switch (type) {
            case VariableType::CONTINUOUS :
                return dimod::Vartype::REAL;
            case VariableType::BINARY :
                return dimod::Vartype::BINARY;
            case VariableType::GENERAL :
                return dimod::Vartype::INTEGER;
            default :
                throw std::domain_error("unsupported vartype");
        }
    }

    template<class bias_type, class index_type>
    void copy_expression(const Expression& source, dimod::Expression<bias_type, index_type>& target,
                         const std::unordered_map<Variable*, index_type>& variable_mapping,
                         const bool is_objective = false) {

        for (auto& term_ptr : source.linterms) {
            const index_type v = variable_mapping.at(term_ptr->var.get());

            target.add_linear(v, term_ptr->coef);
        }

        // need to correct for the LP file's stupid handling of quadratic terms in the objective
        const bias_type mul = (is_objective) ? .5 : 1;

        for (auto& term_ptr : source.quadterms) {
            const index_type u = variable_mapping.at(term_ptr->var1.get());
            const index_type v = variable_mapping.at(term_ptr->var2.get());

            target.add_quadratic(u, v, mul * term_ptr->coef);
        }

        target.add_offset(source.offset);
    }

    template<class bias_type, class index_type>
    dimod::ConstrainedQuadraticModel<bias_type, index_type> model_to_cqm(const Model& model) {
        auto cqm = dimod::ConstrainedQuadraticModel<bias_type, index_type>();

        // Copy the variables into the CQM, keeping track of the indices
        std::unordered_map<Variable*, index_type> variable_mapping;

        for (const auto& v : model.variables) {
            auto vartype = variable_type_to_vartype(v->type);

            bias_type lb = v->lowerbound;
            bias_type ub = v->upperbound;

            const bias_type min_bound = dimod::vartype_info<bias_type>::min(vartype);
            const bias_type max_bound = dimod::vartype_info<bias_type>::max(vartype);

            if (lb < min_bound) {
                lb = min_bound;
            } else if (lb > max_bound) {
                lb = max_bound;
            }

            if (ub < min_bound) {
                ub = min_bound;
            } else if (ub > max_bound) {
                ub = max_bound;
            }

            variable_mapping.emplace(v.get(), cqm.add_variable(vartype, lb, ub));
        }

        // Copy the objective
        copy_expression(*(model.objective), cqm.objective, variable_mapping, true);

        // Then the constraints
        for (const auto& constraint_ptr : model.constraints) {
            auto constraint = cqm.new_constraint();

            copy_expression(*(constraint_ptr->expr), constraint, variable_mapping);

            // handle the bounds
            if (constraint_ptr->lowerbound == constraint_ptr->upperbound) {
                // Equality
                constraint.set_sense(dimod::Sense::EQ);
                constraint.set_rhs(constraint_ptr->lowerbound);
            } else if (constraint_ptr->lowerbound == -std::numeric_limits<double>::infinity()) {
                // Less than or equal to
                constraint.set_sense(dimod::Sense::LE);
                constraint.set_rhs(constraint_ptr->upperbound);
            } else if (constraint_ptr->upperbound == +std::numeric_limits<double>::infinity()) {
                // Greater than or equal to
                constraint.set_sense(dimod::Sense::GE);
                constraint.set_rhs(constraint_ptr->lowerbound);
            } else {
                throw std::domain_error("unexpected constraint sense");
            }

            cqm.add_constraint(std::move(constraint));
        }

        // Handle maximization
        if (model.sense == ObjectiveSense::MAX) {
            cqm.objective.scale(-1);
        }

        return cqm;
    }
    """
    cppCQM[bias_type, index_type] model_to_cqm[bias_type, index_type](const Model&) except+


def cyread_lp_file(object filename):
    """Create a constrained quadratic model from the given LP file."""

    if not os.path.isfile(filename):
        raise ValueError(f"no file named {filename}")

    # A Model as returned by the LP-file reader
    cdef Model model = readinstance(<string>(filename.encode()))

    # Convert to a C++ CQM
    cdef cppCQM[bias_type, index_type] cppcqm = model_to_cqm[bias_type, index_type](model)

    # Create the Python/Cython CQM, from the C++ one (using a move to avoid the copy)
    cdef cyConstrainedQuadraticModel cqm = make_cqm(move(cppcqm))

    # Relabel the variables
    variable_mapping = dict()
    for i in range(model.variables.size()):
        variable_mapping[i] = model.variables[i].name.decode()
    assert(len(variable_mapping) == cqm.num_variables())
    cqm.relabel_variables(variable_mapping)

    # Relabel the constraints
    constraint_mapping = dict()
    for i in range(model.constraints.size()):
        if model.constraints[i].expr.name.size():
            constraint_mapping[i] = model.constraints[i].expr.name.decode()
    cqm.relabel_constraints(constraint_mapping)

    return cqm
